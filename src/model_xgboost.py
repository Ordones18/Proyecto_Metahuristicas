import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RANDOM_SEED, PROCESSED_DATA_DIR, MODELS_DIR

def train_xgboost_model():
    """
    Entrena el modelo XGBoost Classifier utilizando RandomizedSearchCV
    sobre las variables seleccionadas, aplicando pesos de clase en el fit.
    """
    print("\n=== INICIANDO PIPELINE MODELO XGBOOST ===")
    
    # 1. Cargar conjuntos de datos
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train_selected.csv'))
    X_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_val_selected.csv'))
    
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).values.ravel()
    y_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_val.csv')).values.ravel()
    
    # 2. Cargar pesos de clase
    class_weights_path = os.path.join(MODELS_DIR, 'class_weights.joblib')
    sample_weight = None
    if os.path.exists(class_weights_path):
        class_weights_dict = joblib.load(class_weights_path)
        # Crear vector de pesos por muestra
        sample_weight = np.array([class_weights_dict[y] for y in y_train])
        print(f"  - Aplicando pesos de clase en XGBoost fit. Pesos: {class_weights_dict}")
    else:
        print("  - [Advertencia] No se encontraron pesos de clase. Entrenando sin pesos de muestra.")
        
    # 3. Configurar grilla de búsqueda de hiperparámetros
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    # 4. Inicializar modelo base de XGBoost
    xgb = XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    # 5. Configurar RandomizedSearchCV
    print("  - Ejecutando RandomizedSearchCV (10 iteraciones, 3 folds)...")
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=10,
        scoring='f1_macro',
        cv=3,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    # Ajustar RandomizedSearchCV
    fit_params = {}
    if sample_weight is not None:
        fit_params['sample_weight'] = sample_weight
        
    random_search.fit(X_train, y_train, **fit_params)
    
    best_params = random_search.best_params_
    print(f"  - Mejores parámetros encontrados: {best_params}")
    
    # 6. Entrenar el modelo óptimo final con seguimiento de historial
    print("  - Entrenando el modelo XGBoost óptimo final...")
    best_model = XGBClassifier(
        **best_params,
        random_state=RANDOM_SEED,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    start_time = time.time()
    best_model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    train_time = time.time() - start_time
    print(f"  - Modelo XGBoost final entrenado con éxito en {train_time:.2f} segundos.")
    
    # 7. Guardar modelo, historial y estadísticas
    model_path = os.path.join(MODELS_DIR, 'xgboost_model.joblib')
    joblib.dump(best_model, model_path)
    
    # Extraer historial de logloss
    evals_result = best_model.evals_result()
    history = {
        'loss': evals_result['validation_0']['logloss'],
        'val_loss': evals_result['validation_1']['logloss']
    }
    
    joblib.dump(history, os.path.join(MODELS_DIR, 'xgboost_history.joblib'))
    joblib.dump({'train_time': train_time}, os.path.join(MODELS_DIR, 'xgboost_stats.joblib'))
    
    print(f"  - Modelo XGBoost guardado en {model_path}")
    print("[XGBoost] ¡Entrenamiento de XGBoost completado con éxito!\n")
    return best_model

if __name__ == '__main__':
    train_xgboost_model()
