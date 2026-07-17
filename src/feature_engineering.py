import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.inspection import permutation_importance

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shap
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RANDOM_SEED, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR

def select_features(n_features_to_select=10):
    """
    Realiza la selección de variables por consenso utilizando 5 métodos científicos:
    1. Análisis de Correlación (filtro preliminar)
    2. Importancia por Permutación (Permutation Importance)
    3. Valores SHAP (SHAP Feature Importance)
    4. RFE (Recursive Feature Elimination)
    5. Información Mutua (Mutual Information)
    """
    print("\n=== INICIANDO PIPELINE SELECCIÓN DE VARIABLES ===")
    
    # 1. Cargar conjuntos de datos preprocesados y escalados
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).values.ravel()
    
    # Reducimos tamaño de muestra para acelerar cálculos costosos (SHAP, Permutation, RFE)
    # Tomamos una muestra estratificada de 3,000 registros de manera compatible con pandas 2.x
    samples = []
    for val in np.unique(y_train):
        mask = (y_train == val)
        class_subset = X_train[mask]
        class_sample = class_subset.sample(min(len(class_subset), 2500), random_state=RANDOM_SEED)
        samples.append(class_sample)
        
    X_sample = pd.concat(samples, axis=0)
    y_sample = y_train[X_sample.index]
    
    feature_names = X_train.columns.tolist()
    votes = pd.DataFrame(index=feature_names)
    votes['Correlation'] = 1  # Por defecto todas pasan este filtro, a menos que sean redundantes
    
    # --- MÉTODO 1: Análisis de Correlación ---
    print("  - Ejecutando Método 1: Análisis de Correlación (multicolinealidad)...")
    # Usar X_sample (muestra ya preparada) para el cálculo de correlación:
    # resultado prácticamente idéntico al de X_train completo pero mucho más rápido
    corr_matrix = X_sample.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    for col in to_drop:
        votes.loc[col, 'Correlation'] = 0
    print(f"    Variables eliminadas por alta correlación (>0.9): {to_drop}")
    
    # Entrenar un Random Forest Proxy para Permutation Importance, RFE y SHAP
    print("  - Entrenando modelo Random Forest Proxy...")
    rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1, max_depth=12)
    rf.fit(X_sample, y_sample)
    
    # --- MÉTODO 2: Permutation Importance ---
    print("  - Ejecutando Método 2: Importancia por Permutación (Permutation Importance)...")
    perm_importance = permutation_importance(rf, X_sample, y_sample, n_repeats=3, random_state=RANDOM_SEED, n_jobs=-1)
    perm_importances_df = pd.DataFrame({'feature': feature_names, 'importance': perm_importance.importances_mean})
    perm_importances_df = perm_importances_df.sort_values(by='importance', ascending=False)
    # Seleccionar top N variables
    top_perm = perm_importances_df.head(n_features_to_select)['feature'].tolist()
    votes['Permutation'] = 0
    votes.loc[top_perm, 'Permutation'] = 1
    
    # --- MÉTODO 3: SHAP (Shapley Additive exPlanations) ---
    if HAS_PLOTTING:
        print("  - Ejecutando Método 3: Valores SHAP...")
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)
        
        # Manejar salida multiclase o binaria de SHAP
        if isinstance(shap_values, list):
            # Para binario, a veces devuelve lista de dos clases, tomamos la de clase 1
            shap_val_array = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_val_array = shap_values[:, :, 1]
        else:
            shap_val_array = shap_values
            
        shap_importances = np.abs(shap_val_array).mean(axis=0)
        shap_df = pd.DataFrame({'feature': feature_names, 'importance': shap_importances})
        shap_df = shap_df.sort_values(by='importance', ascending=False)
        top_shap = shap_df.head(n_features_to_select)['feature'].tolist()
        votes['SHAP'] = 0
        votes.loc[top_shap, 'SHAP'] = 1
        
        # Guardar gráfico SHAP Summary estático
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_val_array, X_sample, plot_type="bar", show=False)
        plt.title('Importancia Global de Variables mediante SHAP', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, '11_shap_summary_bar.png'), dpi=300)
        plt.close()
    else:
        print("  - [Feature Selection] Omitiendo Método 3 (SHAP) por incompatibilidad de directiva de seguridad.")
    
    # --- MÉTODO 4: RFE (Recursive Feature Elimination) ---
    print("  - Ejecutando Método 4: RFE (Recursive Feature Elimination)...")
    rfe_estimator = RandomForestClassifier(n_estimators=30, random_state=RANDOM_SEED, n_jobs=-1, max_depth=8)
    rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_features_to_select, step=2)
    rfe.fit(X_sample, y_sample)
    top_rfe = [f for f, s in zip(feature_names, rfe.support_) if s]
    votes['RFE'] = 0
    votes.loc[top_rfe, 'RFE'] = 1
    
    # --- MÉTODO 5: Mutual Information ---
    print("  - Ejecutando Método 5: Información Mutua (Mutual Information)...")
    mi_scores = mutual_info_classif(X_sample, y_sample, random_state=RANDOM_SEED)
    mi_df = pd.DataFrame({'feature': feature_names, 'score': mi_scores})
    mi_df = mi_df.sort_values(by='score', ascending=False)
    top_mi = mi_df.head(n_features_to_select)['feature'].tolist()
    votes['Mutual_Info'] = 0
    votes.loc[top_mi, 'Mutual_Info'] = 1
    
    # --- CONCENSO Y SELECCIÓN FINAL ---
    votes['Total_Votes'] = votes.sum(axis=1)
    votes = votes.sort_values(by='Total_Votes', ascending=False)
    
    # Criterio de Selección: Variables seleccionadas por al menos 3 métodos (o 2 si SHAP está inactivo)
    threshold = 3 if HAS_PLOTTING else 2
    selected_features = votes[votes['Total_Votes'] >= threshold].index.tolist()
    
    # Si por alguna razón quedan menos de 5 variables, tomar las top 5 del ranking
    if len(selected_features) < 5:
        selected_features = votes.head(5).index.tolist()
        
    print(f"\n  - Tabla de Votos de Consenso:\n{votes.to_string()}")
    print(f"\n  - Variables Finales Seleccionadas ({len(selected_features)}): {selected_features}")
    
    # Guardar las variables seleccionadas y el DataFrame de votación
    joblib.dump(selected_features, os.path.join(MODELS_DIR, 'selected_features.joblib'))
    votes.to_csv(os.path.join(PROCESSED_DATA_DIR, 'feature_votes.csv'))
    
    # Graficar Consenso de Variables si está disponible
    if HAS_PLOTTING:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=votes['Total_Votes'], y=votes.index, palette='viridis', hue=votes.index, legend=False)
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Umbral de Selección (>= {threshold} votos)')
        plt.title('Selección de Variables por Consenso de Métodos')
        plt.xlabel('Número de Votos')
        plt.ylabel('Característica')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, '12_seleccion_consenso.png'), dpi=300)
        plt.close()
    else:
        print("  - [Feature Selection] Omitiendo gráfico de consenso por incompatibilidad de directiva de seguridad.")
    
    # Generar datasets recortados con solo las variables seleccionadas
    for name in ['X_train', 'X_val', 'X_test']:
        df_full = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{name}.csv'))
        df_sel = df_full[selected_features]
        df_sel.to_csv(os.path.join(PROCESSED_DATA_DIR, f'{name}_selected.csv'), index=False)
        print(f"    Archivo recortado generado: {name}_selected.csv")
        
    print("[Feature Selection] ¡Selección de variables completada con éxito!\n")
    return selected_features

if __name__ == '__main__':
    select_features()
