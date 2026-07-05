import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    RANDOM_SEED, RAW_DATA_PATH, PROCESSED_DATA_DIR, MODELS_DIR,
    LEAKAGE_COLUMNS, DROP_COLUMNS
)

class CustomTargetEncoder:
    """
    Codificador de objetivos personalizado para variables con alta cardinalidad.
    Calcula la probabilidad media de la clase objetivo para cada categoría en el conjunto de entrenamiento,
    con regularización por suavizado (smoothing) para evitar el sobreajuste.
    """
    def __init__(self, cols, smoothing=10.0):
        self.cols = cols
        self.smoothing = smoothing
        self.mapping = {}
        self.global_mean = 0.0

    def fit(self, X, y):
        self.global_mean = y.mean()
        X_temp = X.copy()
        X_temp['target'] = y
        
        for col in self.cols:
            stats = X_temp.groupby(col)['target'].agg(['count', 'mean'])
            counts = stats['count']
            means = stats['mean']
            
            # Fórmula de suavizado: (count * mean + smoothing * global_mean) / (count + smoothing)
            smooth = (counts * means + self.smoothing * self.global_mean) / (counts + self.smoothing)
            self.mapping[col] = smooth.to_dict()
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.cols:
            mapping = self.mapping[col]
            X_out[col] = X_out[col].map(mapping).fillna(self.global_mean)
        return X_out

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


def load_raw_data(file_path=RAW_DATA_PATH):
    """
    Carga el dataset de personas desaparecidas desde el archivo Excel.
    """
    print(f"[ETL] Cargando datos desde {file_path}...")
    # Cargar la hoja '1' que contiene los registros principales
    df = pd.read_excel(file_path, sheet_name='1')
    return df


def clean_data(df):
    """
    Realiza la limpieza inicial de los datos:
    - Elimina duplicados.
    - Corrige coordenadas geográficas (reemplaza comas por puntos decimales).
    - Convierte variables numéricas y maneja valores faltantes.
    - Crea la variable objetivo binaria: 1 para ENCONTRADO, 0 para DESAPARECIDO/FALLECIDO.
    """
    print("[ETL] Limpiando datos...")
    
    # 1. Eliminar duplicados
    initial_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  - Eliminados {initial_len - len(df)} registros duplicados.")
    
    # 2. Variable objetivo binaria
    # Queremos predecir si la persona será ENCONTRADA (1) o NO ENCONTRADA (Fallecido o Desaparecido) (0)
    # Justificación: Ambos estados (Fallecido/Desaparecido) representan un desenlace no exitoso de búsqueda.
    df['target'] = (df['situacion_actual'] == 'ENCONTRADO').astype(int)
    print("  - Creada variable objetivo 'target' (1: Encontrado, 0: No Encontrado).")
    print(f"    Distribución: {df['target'].value_counts(normalize=True).to_dict()}")
    
    # 3. Limpieza de edad
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    # Imputar con la mediana de edad
    edad_median = df['edad'].median()
    df['edad'] = df['edad'].fillna(edad_median)
    print(f"  - Imputada edad faltante con la mediana ({edad_median} años).")
    
    # 4. Limpieza de coordenadas de desaparición
    for col in ['latitud_desaparicion', 'longitud_desaparicion']:
        # Reemplazar comas por puntos
        df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Imputar nulos de coordenadas con la mediana general
        coord_median = df[col].median()
        df[col] = df[col].fillna(coord_median)
    print("  - Coordenadas de desaparición convertidas a flotante e imputadas.")
    
    # 5. Eliminar variables de data leakage y no deseadas
    cols_to_drop = LEAKAGE_COLUMNS + DROP_COLUMNS + ['situacion_actual']
    # Nos aseguramos de conservar la columna 'fecha_desaparicion' para ingeniería de variables temporal antes de eliminarla
    cols_to_drop_final = [c for c in cols_to_drop if c in df.columns]
    
    # Conservamos una copia de las columnas temporalmente para la ingeniería de variables
    df_clean = df.copy()
    
    # Eliminar data leakage
    df_clean = df_clean.drop(columns=cols_to_drop_final)
    print(f"  - Eliminadas variables que producen Data Leakage o irrelevantes: {cols_to_drop_final}")
    
    return df_clean, df['fecha_desaparicion']


def engineer_features(df, fecha_desaparicion_col):
    """
    Ingeniería de características:
    - Extrae año, mes, día, día de la semana y trimestre de fecha_desaparicion.
    - Calcula la antigüedad (días desde la desaparición hasta la fecha máxima en el dataset).
    """
    print("[ETL] Realizando ingeniería de características...")
    
    # Convertir a datetime si no lo está
    fechas = pd.to_datetime(fecha_desaparicion_col)
    
    # Nuevas variables temporales
    df['desaparicion_anio'] = fechas.dt.year
    df['desaparicion_mes'] = fechas.dt.month
    df['desaparicion_dia'] = fechas.dt.day
    df['desaparicion_dia_semana'] = fechas.dt.dayofweek
    df['desaparicion_trimestre'] = fechas.dt.quarter
    
    # Antigüedad: días transcurridos entre la desaparición y el final del rango del dataset (31 de diciembre de 2025)
    max_date = pd.to_datetime('2025-12-31')
    df['antiguedad_dias'] = (max_date - fechas).dt.days
    
    # Eliminar la fecha original para el modelado
    if 'fecha_desaparicion' in df.columns:
        df = df.drop(columns=['fecha_desaparicion'])
        
    print(f"  - Creadas variables temporales y de antigüedad. Columnas actuales: {df.columns.tolist()}")
    return df


def split_data(df):
    """
    Divide los datos de forma estratificada:
    - Entrenamiento: 70%
    - Validación: 15%
    - Prueba: 15%
    """
    print("[ETL] Dividiendo el dataset...")
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Primero: 70% train y 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    
    # Segundo: Dividir el 30% temp en 50% val y 50% test (15% y 15% del total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"  - Dimensiones: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_categorical(X_train, X_val, X_test, y_train):
    """
    Codifica las variables categóricas:
    - Ordinal Encoding para rango_edad.
    - One-Hot Encoding para variables de baja cardinalidad (sexo, etnia, zona).
    - Custom Target Encoding para alta cardinalidad (provincia, canton, nacionalidad).
    """
    print("[ETL] Codificando variables categóricas...")
    
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    
    # 1. Ordinal Encoding para rango_edad
    # Mapeo manual para asegurar que se respete el orden de edad lógico
    rango_edad_mapping = {
        'NIÑOS(AS)': 0,
        'ADOLESCENTES': 1,
        'ADULTO': 2,
        'ADULTO MAYOR': 3,
        'SIN_DATO': 1  # Imputar con la moda (Adolescentes)
    }
    
    for df_temp in [X_train, X_val, X_test]:
        df_temp['rango_edad'] = df_temp['rango_edad'].map(rango_edad_mapping).fillna(1).astype(int)
    print("  - Codificación ordinal aplicada a 'rango_edad'.")
    
    # 2. One-Hot Encoding para variables de baja cardinalidad (sexo, etnia, zona)
    ohe_cols = ['sexo', 'etnia', 'zona']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Ajustar y transformar en entrenamiento
    ohe_train = ohe.fit_transform(X_train[ohe_cols])
    ohe_col_names = ohe.get_feature_names_out(ohe_cols)
    
    # Convertir a dataframes
    df_ohe_train = pd.DataFrame(ohe_train, columns=ohe_col_names, index=X_train.index)
    df_ohe_val = pd.DataFrame(ohe.transform(X_val[ohe_cols]), columns=ohe_col_names, index=X_val.index)
    df_ohe_test = pd.DataFrame(ohe.transform(X_test[ohe_cols]), columns=ohe_col_names, index=X_test.index)
    
    # Concatenar y eliminar las originales
    X_train = pd.concat([X_train.drop(columns=ohe_cols), df_ohe_train], axis=1)
    X_val = pd.concat([X_val.drop(columns=ohe_cols), df_ohe_val], axis=1)
    X_test = pd.concat([X_test.drop(columns=ohe_cols), df_ohe_test], axis=1)
    print(f"  - One-Hot Encoding aplicado a: {ohe_cols}. Creadas {len(ohe_col_names)} columnas.")
    
    # 3. Custom Target Encoding para alta cardinalidad (provincia, canton, nacionalidad)
    high_card_cols = ['provincia', 'canton', 'nacionalidad']
    target_encoder = CustomTargetEncoder(cols=high_card_cols, smoothing=10.0)
    
    # Ajustar en entrenamiento y transformar en todos
    X_train = target_encoder.fit_transform(X_train, y_train)
    X_val = target_encoder.transform(X_val)
    X_test = target_encoder.transform(X_test)
    print(f"  - Target Encoding aplicado a: {high_card_cols}.")
    
    # Guardar codificadores para el pipeline de inferencia
    joblib.dump(ohe, os.path.join(MODELS_DIR, 'one_hot_encoder.joblib'))
    joblib.dump(target_encoder, os.path.join(MODELS_DIR, 'target_encoder.joblib'))
    
    return X_train, X_val, X_test


def compare_scalers(X_train, y_train):
    """
    Compara científicamente StandardScaler y MinMaxScaler utilizando validación cruzada y regresión logística.
    Selecciona el escalador que maximiza el F1-score medio.
    """
    print("[ETL] Comparando escaladores (StandardScaler vs MinMaxScaler)...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # StandardScaler
    scaler_std = StandardScaler()
    X_train_std = scaler_std.fit_transform(X_train)
    model = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
    score_std = cross_val_score(model, X_train_std, y_train, cv=skf, scoring='f1', n_jobs=-1).mean()
    
    # MinMaxScaler
    scaler_minmax = MinMaxScaler()
    X_train_minmax = scaler_minmax.fit_transform(X_train)
    score_minmax = cross_val_score(model, X_train_minmax, y_train, cv=skf, scoring='f1', n_jobs=-1).mean()
    
    print(f"  - StandardScaler F1-Score: {score_std:.5f}")
    print(f"  - MinMaxScaler F1-Score: {score_minmax:.5f}")
    
    best_scaler = StandardScaler() if score_std >= score_minmax else MinMaxScaler()
    print(f"  - Seleccionado: {best_scaler.__class__.__name__}")
    
    return best_scaler


def balance_data(X_train, y_train, method='smote'):
    """
    Aplica técnicas de balanceo de clases:
    - 'smote': Remuestreo sintético para la clase minoritaria.
    - 'under': Submuestreo aleatorio de la clase mayoritaria.
    - 'weight': No remuestrea, pero calcula los pesos de clase para utilizarlos en el entrenamiento de la MLP.
    """
    print(f"[ETL] Aplicando balanceo de clases método: '{method}'...")
    
    if method == 'smote':
        sm = SMOTE(random_state=RANDOM_SEED)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"  - SMOTE aplicado. Dimensiones originales: {X_train.shape}, Nuevas: {X_res.shape}")
        print(f"    Distribución de clases: {pd.Series(y_res).value_counts().to_dict()}")
        return X_res, y_res, None
        
    elif method == 'under':
        rus = RandomUnderSampler(random_state=RANDOM_SEED)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        print(f"  - RandomUnderSampler aplicado. Dimensiones originales: {X_train.shape}, Nuevas: {X_res.shape}")
        print(f"    Distribución de clases: {pd.Series(y_res).value_counts().to_dict()}")
        return X_res, y_res, None
        
    elif method == 'weight':
        # Calcular class weights
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        total = neg_count + pos_count
        
        # Fórmula: weight = total / (classes * class_count)
        weight_0 = total / (2.0 * neg_count)
        weight_1 = total / (2.0 * pos_count)
        
        class_weights = {0: weight_0, 1: weight_1}
        print(f"  - Calculados Class Weights: {class_weights}")
        return X_train, y_train, class_weights
        
    else:
        print("  - Ningún método de balanceo aplicado.")
        return X_train, y_train, None


def run_etl(balance_method='smote'):
    """
    Ejecuta el pipeline de ETL completo:
    Carga -> Limpieza -> Ingeniería de Variables -> Split -> Encoding -> Escalamiento -> Balanceo -> Almacenamiento.
    """
    print("\n=== INICIANDO PIPELINE ETL ===")
    
    # 1. Carga
    df_raw = load_raw_data()
    
    # 2. Limpieza
    df_clean, fecha_desaparicion_col = clean_data(df_raw)
    
    # 3. Ingeniería de variables
    df_features = engineer_features(df_clean, fecha_desaparicion_col)
    
    # 4. Dividir datos
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_features)
    
    # 5. Codificar categóricas
    X_train_enc, X_val_enc, X_test_enc = encode_categorical(X_train, X_val, X_test, y_train)
    
    # 6. Comparar y ajustar Escalador
    scaler = compare_scalers(X_train_enc, y_train)
    
    # Ajustar escalador en entrenamiento y transformar
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc), columns=X_train_enc.columns, index=X_train_enc.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_enc), columns=X_val_enc.columns, index=X_val_enc.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_enc), columns=X_test_enc.columns, index=X_test_enc.index)
    
    # Guardar escalador
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    print(f"  - Escalador guardado en {os.path.join(MODELS_DIR, 'scaler.joblib')}")
    
    # 7. Balanceo de clases (Solo en el conjunto de entrenamiento para evitar Data Leakage)
    X_train_bal, y_train_bal, class_weights = balance_data(X_train_scaled, y_train, method=balance_method)
    
    # 8. Guardar conjuntos de datos procesados
    X_train_bal.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
    X_val_scaled.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_val.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
    
    pd.Series(y_train_bal).to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
    pd.Series(y_val).to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_val.csv'), index=False)
    pd.Series(y_test).to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)
    
    class_weights_path = os.path.join(MODELS_DIR, 'class_weights.joblib')
    if class_weights:
        joblib.dump(class_weights, class_weights_path)
        print(f"  - Pesos de clase guardados en {class_weights_path}")
    else:
        if os.path.exists(class_weights_path):
            os.remove(class_weights_path)
        
    print("[ETL] ¡Pipeline ETL completado con éxito! Archivos guardados en data/processed/\n")
    return X_train_bal, X_val_scaled, X_test_scaled, y_train_bal, y_val, y_test

if __name__ == '__main__':
    run_etl(balance_method='smote')
