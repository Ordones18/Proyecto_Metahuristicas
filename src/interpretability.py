import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

try:
    import shap
    from lime import lime_tabular
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RANDOM_SEED, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR

def run_interpretability_pipeline():
    """
    Ejecuta el análisis de interpretabilidad del Modelo Híbrido entrenado:
    1. Importancia global de características con SHAP (KernelExplainer).
    2. Explicaciones locales detalladas con LIME para casos de ejemplo.
    """
    print("\n=== INICIANDO PIPELINE DE INTERPRETABILIDAD Y EXPLICABILIDAD ===")
    
    # 1. Cargar datos seleccionados
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train_selected.csv'))
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test_selected.csv'))
    
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')).values.ravel()
    
    # 2. Cargar el modelo híbrido (optimizado)
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'mlp_hybrid.keras'))
    
    feature_names = X_train.columns.tolist()
    background_data = X_train.sample(n=min(50, len(X_train)), random_state=RANDOM_SEED)
    
    # Guardar los objetos SHAP para la aplicación Streamlit (los necesita el Predictor)
    joblib.dump({
        'background_data': background_data,
        'feature_names': feature_names
    }, os.path.join(MODELS_DIR, 'shap_explainer_data.joblib'))
    
    if not HAS_PLOTTING:
        print("  - [Interpretability] Omitiendo visualizaciones estáticas SHAP/LIME por incompatibilidad de directivas de seguridad (Pillow DLL block).")
        print("[Interpretabilidad] Pipeline completado (modo simplificado para la Web).\n")
        return
        
    # 3. ANÁLISIS GLOBAL CON SHAP
    print("  - Calculando Valores SHAP globales (KernelExplainer)...")
    # KernelExplainer es agnóstico del modelo. Para optimizar tiempo, tomamos:
    # 50 muestras de fondo (background) y explicamos 20 muestras representativas del test.
    explain_data = X_test.sample(n=min(20, len(X_test)), random_state=RANDOM_SEED)
    
    # Definir función predictora para SHAP (debe retornar matriz de probabilidades de forma (N, 1) o (N,))
    def model_predict_func(x):
        # Convertir a flotante y predecir
        return model.predict(x, verbose=0).astype(np.float64)
        
    explainer = shap.KernelExplainer(model_predict_func, background_data)
    shap_values = explainer.shap_values(explain_data)
    
    # Extraer el array de SHAP values (puede ser de 2D o lista)
    if isinstance(shap_values, list):
        shap_val_array = shap_values[0]
    else:
        shap_val_array = shap_values
        
    # Guardar gráfico de resumen SHAP (beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_val_array, explain_data, show=False)
    plt.title('Distribución de Valores SHAP en Modelo Híbrido MLP+GA', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '18_shap_beeswarm.png'), dpi=300)
    plt.close()
    print("    - Gráfico SHAP beeswarm guardado.")
    
    # El archivo shap_explainer_data.joblib ya fue guardado al inicio del pipeline.
    
    # 4. ANÁLISIS LOCAL CON LIME
    print("  - Configurando LIME Tabular Explainer...")
    # Crear el explicador de LIME
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=['No Encontrado', 'Encontrado'],
        mode='classification',
        random_state=RANDOM_SEED
    )
    
    # Guardar explicador LIME para inferencias individuales en Streamlit no es posible
    # por contener lambdas no serializables. Se recreará al vuelo en la app.
    # joblib.dump(lime_explainer, os.path.join(MODELS_DIR, 'lime_explainer.joblib'))

    
    # Explicar un caso de ejemplo (por ejemplo, el primer caso en explain_data)
    sample_idx = explain_data.index[0]
    sample_instance = X_test.loc[sample_idx]
    true_label = y_test[X_test.index.get_loc(sample_idx)]
    
    print(f"    - Explicando caso individual con LIME (Índice original: {sample_idx}, Clase Real: {true_label})...")
    
    # LIME necesita una función que retorne las probabilidades para ambas clases (N, 2)
    def lime_predict_func(x):
        prob_class_1 = model.predict(x, verbose=0).astype(np.float64)
        prob_class_0 = 1.0 - prob_class_1
        return np.hstack((prob_class_0, prob_class_1))
        
    exp = lime_explainer.explain_instance(
        data_row=sample_instance.values,
        predict_fn=lime_predict_func,
        num_features=5
    )
    
    # Guardar la explicación local como imagen
    fig = exp.as_pyplot_figure()
    plt.title(f'Explicación Local LIME (Predicción individual - Clase Real: {true_label})', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '19_lime_explanation_sample.png'), dpi=300)
    plt.close()
    print("    - Explicación local de LIME guardada como gráfico.")
    
    print("[Interpretabilidad] ¡Pipeline de explicabilidad e interpretabilidad finalizado!\n")

if __name__ == '__main__':
    run_interpretability_pipeline()
