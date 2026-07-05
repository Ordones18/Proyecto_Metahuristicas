import os
import sys
import time
import logging

# Configurar el registro de logs
os.makedirs('outputs/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("outputs/logs/pipeline_run.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Agregar src al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.etl import run_etl
from src.eda import generate_eda_plots
from src.feature_engineering import select_features
from src.model_base import train_base_model
from src.model_hybrid import train_hybrid_model
from src.evaluation import evaluate_models
from src.interpretability import run_interpretability_pipeline

def main():
    start_time = time.time()
    logging.info("======================================================================")
    logging.info("INICIANDO PROYECTO DE INVESTIGACIÓN: MLP + ALGORITMO GENÉTICO")
    logging.info("======================================================================")
    
    try:
        # FASE 1: ETL
        logging.info(">>> Iniciando Fase 1: ETL...")
        phase_start = time.time()
        run_etl(balance_method='smote')
        logging.info(f"Fase 1 completada con éxito en {time.time() - phase_start:.2f} segundos.\n")
        
        # FASE 2: EDA
        logging.info(">>> Iniciando Fase 2: Análisis Exploratorio de Datos (EDA)...")
        phase_start = time.time()
        generate_eda_plots()
        logging.info(f"Fase 2 completada con éxito en {time.time() - phase_start:.2f} segundos.\n")
        
        # FASE 3: Selección de Variables
        logging.info(">>> Iniciando Fase 3: Selección de Variables...")
        phase_start = time.time()
        select_features(n_features_to_select=10)
        logging.info(f"Fase 3 completada con éxito en {time.time() - phase_start:.2f} segundos.\n")
        
        # FASE 4: Modelo Base MLP
        logging.info(">>> Iniciando Fase 4: Entrenamiento de Modelo Base (MLP)...")
        phase_start = time.time()
        train_base_model()
        logging.info(f"Fase 4 completada con éxito en {time.time() - phase_start:.2f} segundos.\n")
        
        # FASE 5: Modelo Híbrido MLP + GA
        logging.info(">>> Iniciando Fase 5: Optimización con Algoritmo Genético (Modelo Híbrido)...")
        phase_start = time.time()
        train_hybrid_model()
        logging.info(f"Fase 5 completada con éxito en {time.time() - phase_start:.2f} segundos.\n")
        
        # FASE 6: Evaluación y Comparación
        logging.info(">>> Iniciando Fase 6: Evaluación y Comparación de Modelos...")
        phase_start = time.time()
        evaluate_models()
        logging.info(f"Fase 6 completada con éxito en {time.time() - phase_start:.2f} segundos.\n")
        
        # FASE 7: Interpretabilidad (SHAP / LIME)
        logging.info(">>> Iniciando Fase 7: Explicabilidad e Interpretabilidad...")
        phase_start = time.time()
        run_interpretability_pipeline()
        logging.info(f"Fase 7 completada con éxito en {time.time() - phase_start:.2f} segundos.\n")
        
        total_time = time.time() - start_time
        logging.info("======================================================================")
        logging.info(f"PIPELINE COMPLETO FINALIZADO CON ÉXITO EN {total_time/60:.2f} MINUTOS")
        logging.info("======================================================================")
        
    except Exception as e:
        logging.exception("Ocurrió un error crítico durante la ejecución del pipeline:")
        sys.exit(1)

if __name__ == '__main__':
    main()
