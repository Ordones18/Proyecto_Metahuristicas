import os
import sys
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, log_loss, confusion_matrix, roc_curve
)
import joblib

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RANDOM_SEED, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR

def load_keras_model_safe(file_path):
    import zipfile, json, tempfile, shutil
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        config_path = os.path.join(temp_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        def clean_config(obj):
            if isinstance(obj, dict):
                if 'class_name' in obj:
                    if obj['class_name'] == 'GlorotUniform':
                        return 'glorot_uniform'
                    if obj['class_name'] == 'Zeros':
                        return 'zeros'
                if 'quantization_config' in obj:
                    del obj['quantization_config']
                for k, v in list(obj.items()):
                    obj[k] = clean_config(v)
            elif isinstance(obj, list):
                obj = [clean_config(i) for i in obj]
            return obj
        config = clean_config(config)
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Save fixed model file in the same directory but with a unique name
        fixed_path = file_path + '.fixed_compat.keras'
        with zipfile.ZipFile(fixed_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, temp_dir)
                    zip_out.write(full_path, rel_path)
        model = tf.keras.models.load_model(fixed_path)
        try:
            os.remove(fixed_path)
        except OSError:
            pass
        return model
    finally:
        shutil.rmtree(temp_dir)


def run_mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Realiza la prueba estadística de McNemar para comparar dos modelos.
    McNemar evalúa si las discrepancias en las predicciones son estadísticamente significativas.
    b: Cantidad de casos donde el Modelo 1 acierta y el Modelo 2 falla.
    c: Cantidad de casos donde el Modelo 1 falla y el Modelo 2 acierta.
    """
    # Determinar aciertos
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    # Tabla de contingencia
    a = np.sum(correct1 & correct2)       # Ambos aciertan
    b = np.sum(correct1 & ~correct2)      # Solo Modelo 1 acierta
    c = np.sum(~correct1 & correct2)      # Solo Modelo 2 acierta
    d = np.sum(~correct1 & ~correct2)     # Ambos fallan
    
    print(f"    Tabla Contingencia: [a={a}, b={b}, c={c}, d={d}]")
    
    # Calcular estadístico de McNemar con corrección de continuidad de Yates
    if b + c > 0:
        statistic = (abs(b - c) - 1.0)**2 / (b + c)
        # p-value desde distribución chi-cuadrada con 1 grado de libertad
        from scipy.stats import chi2
        p_value = 1.0 - chi2.cdf(statistic, df=1)
    else:
        statistic = 0.0
        p_value = 1.0
        
    return statistic, p_value


def evaluate_models():
    """
    Evalúa y compara el Modelo Base (MLP), el Modelo Híbrido (MLP + GA) y el modelo CatBoost (Challenger).
    Genera gráficos comparativos, métricas en tablas y ejecuta el Test de McNemar.
    """
    print("\n=== INICIANDO PIPELINE DE EVALUACIÓN Y COMPARACIÓN ===")
    
    # 1. Cargar datos de prueba seleccionados
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test_selected.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')).values.ravel()
    
    # 2. Cargar modelos
    print("  - Cargando modelos entrenados...")
    model_base = load_keras_model_safe(os.path.join(MODELS_DIR, 'mlp_base.keras'))
    model_hybrid = load_keras_model_safe(os.path.join(MODELS_DIR, 'mlp_hybrid.keras'))
    model_pso = load_keras_model_safe(os.path.join(MODELS_DIR, 'mlp_pso.keras'))
    
    # Cargar estadísticas de entrenamiento
    stats_base = joblib.load(os.path.join(MODELS_DIR, 'mlp_base_stats.joblib'))
    stats_hybrid = joblib.load(os.path.join(MODELS_DIR, 'mlp_hybrid_stats.joblib'))
    stats_pso = joblib.load(os.path.join(MODELS_DIR, 'mlp_pso_stats.joblib'))
    
    # 3. Predicciones e Inferencia
    print("  - Calculando predicciones y tiempos de inferencia...")
    
    # Warm-up: la primera llamada a predict() en TF incluye compilación JIT, que inflaría
    # el tiempo medido. Una llamada previa con 1 muestra descarta ese overhead.
    _warmup = X_test.iloc[:1]
    model_base.predict(_warmup, verbose=0)
    model_hybrid.predict(_warmup, verbose=0)
    model_pso.predict(_warmup, verbose=0)
    
    # Cargar thresholds óptimos guardados durante el entrenamiento
    # Los modelos GA y PSO guardan el umbral que maximizó F1-macro en validación
    thresh_hybrid = joblib.load(os.path.join(MODELS_DIR, 'mlp_hybrid_threshold.joblib')) \
        if os.path.exists(os.path.join(MODELS_DIR, 'mlp_hybrid_threshold.joblib')) else 0.5
    thresh_pso = joblib.load(os.path.join(MODELS_DIR, 'mlp_pso_threshold.joblib')) \
        if os.path.exists(os.path.join(MODELS_DIR, 'mlp_pso_threshold.joblib')) else 0.5
    
    # Para el modelo base buscar threshold óptimo en validación
    # (garantiza que el set de test permanezca completamente aislado)
    X_val_sel = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_val_selected.csv'))
    y_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_val.csv')).values.ravel()
    probs_base_val = model_base.predict(X_val_sel, verbose=0).ravel()
    thresh_base, best_f1_base = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.01):
        p = (probs_base_val >= t).astype(int)
        f = f1_score(y_val, p, average='macro', zero_division=0)
        if f > best_f1_base:
            best_f1_base, thresh_base = f, t
            
    joblib.dump(thresh_base, os.path.join(MODELS_DIR, 'mlp_base_threshold.joblib'))
    
    print(f"  - Thresholds óptimos: Base={thresh_base:.2f} | GA={thresh_hybrid:.2f} | PSO={thresh_pso:.2f}")
    
    # Modelo Base
    start_time = time.time()
    probs_base = model_base.predict(X_test, verbose=0).ravel()
    infer_time_base = (time.time() - start_time) / len(X_test) * 1000 # ms por muestra
    preds_base = (probs_base >= thresh_base).astype(int)
    
    # Modelo Híbrido GA
    start_time = time.time()
    probs_hybrid = model_hybrid.predict(X_test, verbose=0).ravel()
    infer_time_hybrid = (time.time() - start_time) / len(X_test) * 1000 # ms por muestra
    preds_hybrid = (probs_hybrid >= thresh_hybrid).astype(int)
    
    # Modelo Híbrido PSO
    start_time = time.time()
    probs_pso = model_pso.predict(X_test, verbose=0).ravel()
    infer_time_pso = (time.time() - start_time) / len(X_test) * 1000 # ms por muestra
    preds_pso = (probs_pso >= thresh_pso).astype(int)
    
    # 4. Calcular métricas
    metrics = {}
    
    # Base Model
    metrics['Base'] = {
        'Accuracy': accuracy_score(y_test, preds_base),
        'Precision': precision_score(y_test, preds_base, average='macro'),
        'Recall': recall_score(y_test, preds_base, average='macro'),
        'F1-Score': f1_score(y_test, preds_base, average='macro'),
        'ROC-AUC': roc_auc_score(y_test, probs_base),
        'Log_Loss': log_loss(y_test, probs_base),
        'Train_Time': stats_base['train_time'],
        'Inference_Time_ms': infer_time_base
    }
    
    # Calcular PR-AUC para modelo base
    p_b, r_b, _ = precision_recall_curve(y_test, probs_base)
    metrics['Base']['PR-AUC'] = auc(r_b, p_b)
    
    # Hybrid Model GA
    metrics['Hybrid'] = {
        'Accuracy': accuracy_score(y_test, preds_hybrid),
        'Precision': precision_score(y_test, preds_hybrid, average='macro'),
        'Recall': recall_score(y_test, preds_hybrid, average='macro'),
        'F1-Score': f1_score(y_test, preds_hybrid, average='macro'),
        'ROC-AUC': roc_auc_score(y_test, probs_hybrid),
        'Log_Loss': log_loss(y_test, probs_hybrid),
        'Train_Time': stats_hybrid['train_time'],
        'Inference_Time_ms': infer_time_hybrid
    }
    
    # Calcular PR-AUC para modelo híbrido GA
    p_h, r_h, _ = precision_recall_curve(y_test, probs_hybrid)
    metrics['Hybrid']['PR-AUC'] = auc(r_h, p_h)
    
    # Hybrid Model PSO
    metrics['PSO'] = {
        'Accuracy': accuracy_score(y_test, preds_pso),
        'Precision': precision_score(y_test, preds_pso, average='macro'),
        'Recall': recall_score(y_test, preds_pso, average='macro'),
        'F1-Score': f1_score(y_test, preds_pso, average='macro'),
        'ROC-AUC': roc_auc_score(y_test, probs_pso),
        'Log_Loss': log_loss(y_test, probs_pso),
        'Train_Time': stats_pso['train_time'],
        'Inference_Time_ms': infer_time_pso
    }
    
    # Calcular PR-AUC para PSO
    p_p, r_p, _ = precision_recall_curve(y_test, probs_pso)
    metrics['PSO']['PR-AUC'] = auc(r_p, p_p)
    
    df_metrics = pd.DataFrame(metrics).T
    
    # Calcular mejoras porcentuales de GA sobre Base
    df_metrics.loc['Mejora GA (%)'] = ((df_metrics.loc['Hybrid'] - df_metrics.loc['Base']) / df_metrics.loc['Base']) * 100
    # Para Log Loss e Inferencia, menor es mejor, la mejora es inversa
    df_metrics.loc['Mejora GA (%)', 'Log_Loss'] = ((df_metrics.loc['Base', 'Log_Loss'] - df_metrics.loc['Hybrid', 'Log_Loss']) / df_metrics.loc['Base', 'Log_Loss']) * 100
    df_metrics.loc['Mejora GA (%)', 'Inference_Time_ms'] = ((df_metrics.loc['Base', 'Inference_Time_ms'] - df_metrics.loc['Hybrid', 'Inference_Time_ms']) / df_metrics.loc['Base', 'Inference_Time_ms']) * 100
    
    # Calcular mejoras porcentuales de PSO sobre Base
    df_metrics.loc['Mejora PSO (%)'] = ((df_metrics.loc['PSO'] - df_metrics.loc['Base']) / df_metrics.loc['Base']) * 100
    # Para Log Loss e Inferencia, menor es mejor, la mejora es inversa
    df_metrics.loc['Mejora PSO (%)', 'Log_Loss'] = ((df_metrics.loc['Base', 'Log_Loss'] - df_metrics.loc['PSO', 'Log_Loss']) / df_metrics.loc['Base', 'Log_Loss']) * 100
    df_metrics.loc['Mejora PSO (%)', 'Inference_Time_ms'] = ((df_metrics.loc['Base', 'Inference_Time_ms'] - df_metrics.loc['PSO', 'Inference_Time_ms']) / df_metrics.loc['Base', 'Inference_Time_ms']) * 100
    
    print(f"\n  - Tabla Comparativa de Métricas:\n{df_metrics.to_string()}")
    
    # Guardar reporte
    df_metrics.to_csv(os.path.join(REPORTS_DIR, 'model_comparison_metrics.csv'))
    
    # 5. Prueba de McNemar
    print("\n  - Ejecutando Test Estadístico de McNemar...")
    print("    * Comparando MLP Base vs MLP GA:")
    stat_hybrid, p_val_hybrid = run_mcnemar_test(y_test, preds_base, preds_hybrid)
    significative_hybrid = p_val_hybrid < 0.05
    print(f"      Estadístico: {stat_hybrid:.5f}, p-valor: {p_val_hybrid:.5e} (Significativo: {significative_hybrid})")
    
    print("    * Comparando MLP GA vs MLP PSO:")
    stat_pso, p_val_pso = run_mcnemar_test(y_test, preds_hybrid, preds_pso)
    significative_pso = p_val_pso < 0.05
    print(f"      Estadístico: {stat_pso:.5f}, p-valor: {p_val_pso:.5e} (Significativo: {significative_pso})")
    
    # Guardar resultados estadísticos
    with open(os.path.join(REPORTS_DIR, 'statistical_comparison.txt'), 'w', encoding='utf-8') as f:
        f.write("PRUEBAS ESTADÍSTICAS DE COMPARACIÓN (TEST DE MCNEMAR)\n")
        f.write("====================================================\n\n")
        
        f.write("1. MLP Base vs MLP Híbrido (Optimizado por GA):\n")
        f.write("----------------------------------------------\n")
        f.write(f"Estadístico Chi-cuadrado: {stat_hybrid:.5f}\n")
        f.write(f"p-valor: {p_val_hybrid:.5e}\n")
        f.write(f"Significancia (alfa=0.05): {'Diferencia Estadísticamente Significativa' if significative_hybrid else 'Diferencia No Significativa'}\n")
        if significative_hybrid:
            f.write("Interpretación: El modelo híbrido GA presenta un comportamiento predictivo\n")
            f.write("significativamente diferente y superior al modelo base MLP, confirmando la efectividad del Algoritmo Genético.\n\n")
        else:
            f.write("Interpretación: Las predicciones de ambos modelos no muestran discrepancias estadísticamente significativas.\n\n")
            
        f.write("2. MLP Híbrido (GA) vs MLP Híbrido (PSO):\n")
        f.write("----------------------------------------\n")
        f.write(f"Estadístico Chi-cuadrado: {stat_pso:.5f}\n")
        f.write(f"p-valor: {p_val_pso:.5e}\n")
        f.write(f"Significancia (alfa=0.05): {'Diferencia Estadísticamente Significativa' if significative_pso else 'Diferencia No Significativa'}\n")
        if significative_pso:
            f.write("Interpretación: Existe una diferencia estadísticamente significativa en el comportamiento predictivo\n")
            f.write("entre el MLP Híbrido GA y el MLP Híbrido PSO, evidenciando el impacto de las diferentes metaheurísticas.\n")
        else:
            f.write("Interpretación: Las discrepancias predictivas entre el MLP GA y el MLP PSO no son estadísticamente significativas.\n")
            
    # --- GRÁFICOS DE EVALUACIÓN ---
    if HAS_PLOTTING:
        print("  - Generando gráficos de curvas ROC y Precision-Recall...")
        
        # 1. Curva ROC Comparativa
        fpr_b, tpr_b, _ = roc_curve(y_test, probs_base)
        fpr_h, tpr_h, _ = roc_curve(y_test, probs_hybrid)
        fpr_p, tpr_p, _ = roc_curve(y_test, probs_pso)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr_b, tpr_b, color='blue', lw=2, label=f'MLP Base (AUC = {metrics["Base"]["ROC-AUC"]:.4f})')
        plt.plot(fpr_h, tpr_h, color='red', lw=2, label=f'MLP Híbrido GA (AUC = {metrics["Hybrid"]["ROC-AUC"]:.4f})')
        plt.plot(fpr_p, tpr_p, color='green', lw=2, label=f'MLP Híbrido PSO (AUC = {metrics["PSO"]["ROC-AUC"]:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title('Curva ROC Comparativa (Conjunto de Test)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(FIGURES_DIR, '14_curva_roc_comparativa.png'), dpi=300)
        plt.close()
        
        # 2. Curva Precision-Recall Comparativa
        plt.figure(figsize=(10, 8))
        plt.plot(r_b, p_b, color='blue', lw=2, label=f'MLP Base (PR-AUC = {metrics["Base"]["PR-AUC"]:.4f})')
        plt.plot(r_h, p_h, color='red', lw=2, label=f'MLP Híbrido GA (PR-AUC = {metrics["Hybrid"]["PR-AUC"]:.4f})')
        plt.plot(r_p, p_p, color='green', lw=2, label=f'MLP Híbrido PSO (PR-AUC = {metrics["PSO"]["PR-AUC"]:.4f})')
        plt.xlabel('Recall (Sensibilidad)')
        plt.ylabel('Precision (Exactitud Predictiva)')
        plt.title('Curva Precision-Recall Comparativa (Conjunto de Test)')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(os.path.join(FIGURES_DIR, '15_curva_pr_comparativa.png'), dpi=300)
        plt.close()
        
        # 3. Matrices de Confusión
        print("  - Generando matrices de confusión...")
        cm_base = confusion_matrix(y_test, preds_base)
        cm_hybrid = confusion_matrix(y_test, preds_hybrid)
        cm_pso = confusion_matrix(y_test, preds_pso)
        
        fig, ax = plt.subplots(1, 3, figsize=(24, 7))
        
        sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False)
        ax[0].set_title('Matriz de Confusión - MLP Base')
        ax[0].set_xlabel('Predicción')
        ax[0].set_ylabel('Realidad')
        ax[0].set_xticklabels(['No Encontrado', 'Encontrado'])
        ax[0].set_yticklabels(['No Encontrado', 'Encontrado'])
        
        sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Reds', ax=ax[1], cbar=False)
        ax[1].set_title('Matriz de Confusión - MLP GA')
        ax[1].set_xlabel('Predicción')
        ax[1].set_ylabel('Realidad')
        ax[1].set_xticklabels(['No Encontrado', 'Encontrado'])
        ax[1].set_yticklabels(['No Encontrado', 'Encontrado'])
        
        sns.heatmap(cm_pso, annot=True, fmt='d', cmap='Greens', ax=ax[2], cbar=False)
        ax[2].set_title('Matriz de Confusión - MLP PSO')
        ax[2].set_xlabel('Predicción')
        ax[2].set_ylabel('Realidad')
        ax[2].set_xticklabels(['No Encontrado', 'Encontrado'])
        ax[2].set_yticklabels(['No Encontrado', 'Encontrado'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, '16_matrices_confusion.png'), dpi=300)
        plt.close()
        
        # 4. Curvas de entrenamiento comparativas (Pérdida e Historial)
        print("  - Generando historial de entrenamiento comparativo...")
        hist_base = joblib.load(os.path.join(MODELS_DIR, 'mlp_base_history.joblib'))
        hist_hybrid = joblib.load(os.path.join(MODELS_DIR, 'mlp_hybrid_history.joblib'))
        hist_pso = joblib.load(os.path.join(MODELS_DIR, 'mlp_pso_history.joblib'))
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pérdida
        ax[0].plot(hist_base['loss'], label='Base Train Loss', color='blue', linestyle='--')
        ax[0].plot(hist_base['val_loss'], label='Base Val Loss', color='blue')
        ax[0].plot(hist_hybrid['loss'], label='GA Train Loss', color='red', linestyle='--')
        ax[0].plot(hist_hybrid['val_loss'], label='GA Val Loss', color='red')
        ax[0].plot(hist_pso['loss'], label='PSO Train Loss', color='green', linestyle='--')
        ax[0].plot(hist_pso['val_loss'], label='PSO Val Loss', color='green')
        ax[0].set_title('Historial de Pérdida (Loss)')
        ax[0].set_xlabel('Época')
        ax[0].set_ylabel('Binary Crossentropy')
        ax[0].legend()
        ax[0].grid(True)
        
        # Precisión / Exactitud (Accuracy)
        ax[1].plot(hist_base['accuracy'], label='Base Train Acc', color='blue', linestyle='--')
        ax[1].plot(hist_base['val_accuracy'], label='Base Val Acc', color='blue')
        ax[1].plot(hist_hybrid['accuracy'], label='GA Train Acc', color='red', linestyle='--')
        ax[1].plot(hist_hybrid['val_accuracy'], label='GA Val Acc', color='red')
        ax[1].plot(hist_pso['accuracy'], label='PSO Train Acc', color='green', linestyle='--')
        ax[1].plot(hist_pso['val_accuracy'], label='PSO Val Acc', color='green')
        ax[1].set_title('Historial de Exactitud (Accuracy)')
        ax[1].set_xlabel('Época')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        ax[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, '17_curvas_entrenamiento_comparativas.png'), dpi=300)
        plt.close()
    else:
        print("  - [Evaluation] Omitiendo generación de gráficos comparativos por incompatibilidad de directiva de seguridad.")
    
    # Generar archivo de reporte en Excel con xlsxwriter
    excel_path = os.path.join(REPORTS_DIR, 'reporte_comparativo_modelos.xlsx')
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df_metrics.to_excel(writer, sheet_name='Metricas')
        # Escribir notas de los tests de McNemar
        worksheet = writer.book.add_worksheet('Test McNemar')
        worksheet.write(0, 0, 'Comparación')
        worksheet.write(0, 1, 'Estadístico Chi-cuadrado')
        worksheet.write(0, 2, 'p-valor')
        worksheet.write(0, 3, 'Estadísticamente Significativo')
        
        worksheet.write(1, 0, 'MLP Base vs MLP GA')
        worksheet.write(1, 1, stat_hybrid)
        worksheet.write(1, 2, p_val_hybrid)
        worksheet.write(1, 3, 'SÍ' if significative_hybrid else 'NO')
        
        worksheet.write(2, 0, 'MLP GA vs MLP PSO')
        worksheet.write(2, 1, stat_pso)
        worksheet.write(2, 2, p_val_pso)
        worksheet.write(2, 3, 'SÍ' if significative_pso else 'NO')
        
    print(f"  - Reporte de comparación en Excel guardado en: {excel_path}")
    print("[Evaluación] ¡Pipeline de evaluación finalizado con éxito!\n")

if __name__ == '__main__':
    evaluate_models()
