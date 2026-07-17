import os
import sys
import time
import logging

# ── GPU: Configurar ANTES de importar TensorFlow o cualquier módulo src ──────
# TensorFlow exige que set_memory_growth se llame antes de que se cree
# cualquier tensor o sesión en GPU. Hacerlo aquí garantiza el orden correcto.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_enable_xla_devices=false')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')  # Fallback si set_memory_growth falla

import tensorflow as tf

_gpus_early = tf.config.list_physical_devices('GPU')
if _gpus_early:
    try:
        for _gpu in _gpus_early:
            tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass  # Contexto ya inicializado; TF_FORCE_GPU_ALLOW_GROWTH actúa como fallback
# ─────────────────────────────────────────────────────────────────────────────

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
from src.model_pso import train_pso_model
from src.evaluation import evaluate_models
from src.interpretability import run_interpretability_pipeline


def _log_gpu_status():
    """Reporta en el log el estado actual de GPU/VRAM (la config ya se aplicó al inicio)."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpu_names = [g.name for g in gpus]
        logging.info(f"[GPU] ✓ TensorFlow detectó {len(gpus)} GPU(s): {gpu_names}")
        logging.info("[GPU] ✓ Memoria dinámica (memory_growth) activa — VRAM asignada bajo demanda.")
        logging.info("[GPU] ✓ El entrenamiento se ejecutará en GPU (CUDA).")
    else:
        logging.warning("[GPU] ✗ TensorFlow NO detectó ninguna GPU. El entrenamiento usará CPU.")
        logging.warning("[GPU]   Verifica LD_LIBRARY_PATH y que tensorflow[and-cuda] esté instalado.")


def main():
    start_time = time.time()
    logging.info("======================================================================")
    logging.info("INICIANDO PROYECTO DE INVESTIGACIÓN: MLP + ALGORITMO GENÉTICO")
    logging.info("======================================================================")

    # Reportar estado GPU (configuración ya aplicada al inicio del módulo)
    _log_gpu_status()

    try:
        # FASE 1: ETL
        logging.info(">>> Iniciando Fase 1: ETL...")
        phase_start = time.time()
        run_etl(balance_method='weight')
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

        # FASE 5.5: Modelo Híbrido MLP + PSO
        logging.info(">>> Iniciando Fase 5.5: Optimización con Enjambre de Partículas (MLP + PSO)...")
        phase_start = time.time()
        train_pso_model()
        logging.info(f"Fase 5.5 completada con éxito en {time.time() - phase_start:.2f} segundos.\n")

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
