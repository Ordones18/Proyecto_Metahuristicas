import os
import sys

# Deshabilitar backend interactivo de matplotlib para evitar problemas con tkinter en hilos secundarios
import matplotlib
matplotlib.use('Agg')

import time
import shutil
import pandas as pd

# Agregar el directorio raíz al path
sys.path.append(os.getcwd())

from src.etl import run_etl
from src.feature_engineering import select_features
from src.model_base import train_base_model
from src.model_hybrid import train_hybrid_model
from src.evaluation import evaluate_models
from src.interpretability import run_interpretability_pipeline

def run_for_method(method):
    print(f"\n==========================================")
    print(f"RUNNING BENCHMARK FOR METHOD: {method}")
    print(f"==========================================\n")
    
    # 1. Run ETL
    run_etl(balance_method=method)
    
    # 2. Run Feature Selection (to update X_train_selected.csv to match the current balance method)
    select_features(n_features_to_select=10)
    
    # 3. Train Base Model
    train_base_model()
    
    # 4. Train Hybrid Model
    train_hybrid_model()
    
    # 5. Evaluate Models
    evaluate_models()
    
    # 6. Run Interpretability
    run_interpretability_pipeline()
    
    # Copy results to benchmark folder
    dest_dir = f"outputs/benchmark/{method}"
    os.makedirs(dest_dir, exist_ok=True)
    
    # Files to copy
    files_to_copy = [
        "outputs/reports/model_comparison_metrics.csv",
        "outputs/reports/statistical_comparison.txt",
        "outputs/logs/pipeline_run.log"
    ]
    for f in files_to_copy:
        if os.path.exists(f):
            shutil.copy(f, os.path.join(dest_dir, os.path.basename(f)))
            
    # Also copy models
    models_dir = "models"
    dest_models = os.path.join(dest_dir, "models")
    os.makedirs(dest_models, exist_ok=True)
    for model_file in os.listdir(models_dir):
        if model_file.endswith(".keras") or model_file.endswith(".joblib"):
            shutil.copy(os.path.join(models_dir, model_file), os.path.join(dest_models, model_file))
            
    print(f"\nCompleted benchmark for: {method}\n")

def main():
    methods = ["under", "weight", "smote"]
    for m in methods:
        run_for_method(m)
        
    print("\nBENCHMARKS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
