import streamlit as st
import os
import sys
import time

# Asegurar que el directorio raíz está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import RAW_DATA_PATH, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR



# Estilos CSS premium (Tema Oscuro, tipografía moderna, bordes redondeados y gradientes)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B, #8A2387);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Consola de Entrenamiento del Pipeline</h1>', unsafe_allow_html=True)
st.write("Controle y ejecute las fases científicas del pipeline de Inteligencia Artificial.")

# Ruta para el archivo de logs persistentes
log_file_path = os.path.join(LOGS_DIR, 'pipeline_run.log')

# Verificar si el archivo Excel está disponible
if not os.path.exists(RAW_DATA_PATH):
    st.error(f"Archivo de datos no encontrado en: {RAW_DATA_PATH}. Por favor, coloque el archivo Excel del dataset en la carpeta raíz del proyecto para poder entrenar.")
else:
    st.success(f"Dataset localizado correctamente: {os.path.basename(RAW_DATA_PATH)}")
    
    with st.container(border=True):
        st.subheader("Configuración del Entrenamiento")
        st.write("El pipeline está configurado para utilizar **Class Weights (Pesos de Clase)** como método de balanceo debido a su rigurosidad científica (sin inventar datos sintéticos ni desechar registros históricos).")
        balance_method = "weight"
    
    # Contenedor para el log anterior (se limpia al iniciar un nuevo entrenamiento)
    log_placeholder = st.empty()
    
    # Mostrar logs anteriores si existen para persistencia
    if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
        with log_placeholder.container():
            st.subheader("Bitácora del Último Entrenamiento Realizado")
            with open(log_file_path, 'r', encoding='utf-8') as f:
                old_logs = f.read()
            st.code(old_logs, language="text")
        
    if st.button("Iniciar Pipeline Completo de Entrenamiento", use_container_width=True):
        # Limpiar inmediatamente el log anterior en la UI para evitar acumulación visual
        log_placeholder.empty()
        
        st.write("---")
        st.info("Iniciando pipeline de entrenamiento. Esto puede tomar unos minutos...")
        
        # Contenedores para logs dinámicos
        status_box = st.empty()
        log_box = st.empty()
        
        # Redirect stdout to both file and Streamlit placeholder
        import io
        class StreamlitRedirect(io.StringIO):
            def __init__(self, placeholder, log_file):
                super().__init__()
                self.placeholder = placeholder
                self.log_file = log_file
                self.text = ""
            def write(self, s):
                self.text += s
                self.placeholder.code(self.text)
                self.log_file.write(s)
                self.log_file.flush()
            def flush(self):
                pass
                
        # Abrir el log para escritura
        log_file = open(log_file_path, 'w', encoding='utf-8')
        redirect = StreamlitRedirect(log_box, log_file)
        old_stdout = sys.stdout
        sys.stdout = redirect
        
        try:
            # FASE 1: ETL
            status_box.info("Ejecutando Fase 1: ETL...")
            from src.etl import run_etl
            run_etl(balance_method=balance_method)
            
            # FASE 2: EDA
            status_box.info("Ejecutando Fase 2: Análisis Exploratorio de Datos (EDA)...")
            from src.eda import generate_eda_plots
            generate_eda_plots()
            
            # FASE 3: Selección de Variables
            status_box.info("Ejecutando Fase 3: Selección de Variables...")
            from src.feature_engineering import select_features
            select_features(n_features_to_select=10)
            
            # FASE 4: Modelo Base MLP
            status_box.info("Ejecutando Fase 4: Entrenamiento de Modelo Base (MLP)...")
            from src.model_base import train_base_model
            train_base_model()
            
            # FASE 5: Modelo Híbrido MLP + GA
            status_box.info("Ejecutando Fase 5: Optimización con Algoritmo Genético (Modelo Híbrido)...")
            from src.model_hybrid import train_hybrid_model
            train_hybrid_model()
            
            # FASE 5.5: Modelo XGBoost (Challenger)
            status_box.info("Ejecutando Fase 5.5: Entrenamiento de Modelo XGBoost (Challenger)...")
            from src.model_xgboost import train_xgboost_model
            train_xgboost_model()
            
            # FASE 6: Evaluación y Comparación
            status_box.info("Ejecutando Fase 6: Evaluación y Comparación...")
            from src.evaluation import evaluate_models
            evaluate_models()
            
            # FASE 7: Explicabilidad SHAP/LIME
            status_box.info("Ejecutando Fase 7: Explicabilidad e Interpretabilidad...")
            from src.interpretability import run_interpretability_pipeline
            run_interpretability_pipeline()
            
            sys.stdout = old_stdout
            log_file.close()
            status_box.success("¡Pipeline de entrenamiento completado con éxito! Todos los modelos y reportes han sido generados. Ya puede realizar predicciones y comparar modelos.")
            
            # Recargar la página para actualizar el contenedor de logs persistentes
            st.rerun()
            
        except Exception as e:
            sys.stdout = old_stdout
            if not log_file.closed:
                log_file.write(f"\n[ERROR CRITICO]: {str(e)}\n")
                log_file.close()
            status_box.error(f"Ocurrió un error crítico durante el entrenamiento: {e}")
            st.exception(e)
