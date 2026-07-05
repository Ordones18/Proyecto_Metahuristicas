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
import matplotlib.pyplot as plt
import seaborn as sns

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RANDOM_SEED, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR

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
    Evalúa y compara el Modelo Base (MLP) y el Modelo Híbrido (MLP + GA).
    Genera gráficos comparativos, métricas en tablas y ejecuta el Test de McNemar.
    """
    print("\n=== INICIANDO PIPELINE DE EVALUACIÓN Y COMPARACIÓN ===")
    
    # 1. Cargar datos de prueba seleccionados
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test_selected.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')).values.ravel()
    
    # 2. Cargar modelos
    print("  - Cargando modelos entrenados...")
    model_base = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'mlp_base.keras'))
    model_hybrid = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'mlp_hybrid.keras'))
    
    # Cargar estadísticas de entrenamiento
    stats_base = joblib.load(os.path.join(MODELS_DIR, 'mlp_base_stats.joblib'))
    stats_hybrid = joblib.load(os.path.join(MODELS_DIR, 'mlp_hybrid_stats.joblib'))
    
    # 3. Predicciones e Inferencia
    print("  - Calculando predicciones y tiempos de inferencia...")
    
    # Modelo Base
    start_time = time.time()
    probs_base = model_base.predict(X_test, verbose=0).ravel()
    infer_time_base = (time.time() - start_time) / len(X_test) * 1000 # ms por muestra
    preds_base = (probs_base >= 0.5).astype(int)
    
    # Modelo Híbrido
    start_time = time.time()
    probs_hybrid = model_hybrid.predict(X_test, verbose=0).ravel()
    infer_time_hybrid = (time.time() - start_time) / len(X_test) * 1000 # ms por muestra
    preds_hybrid = (probs_hybrid >= 0.5).astype(int)
    
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
    
    # Hybrid Model
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
    
    # Calcular PR-AUC para modelo híbrido
    p_h, r_h, _ = precision_recall_curve(y_test, probs_hybrid)
    metrics['Hybrid']['PR-AUC'] = auc(r_h, p_h)
    
    df_metrics = pd.DataFrame(metrics).T
    
    # Calcular mejoras porcentuales
    df_metrics.loc['Mejora (%)'] = ((df_metrics.loc['Hybrid'] - df_metrics.loc['Base']) / df_metrics.loc['Base']) * 100
    # Para Log Loss e Inferencia, menor es mejor, la mejora es inversa
    df_metrics.loc['Mejora (%)', 'Log_Loss'] = ((df_metrics.loc['Base', 'Log_Loss'] - df_metrics.loc['Hybrid', 'Log_Loss']) / df_metrics.loc['Base', 'Log_Loss']) * 100
    df_metrics.loc['Mejora (%)', 'Inference_Time_ms'] = ((df_metrics.loc['Base', 'Inference_Time_ms'] - df_metrics.loc['Hybrid', 'Inference_Time_ms']) / df_metrics.loc['Base', 'Inference_Time_ms']) * 100
    
    print(f"\n  - Tabla Comparativa de Métricas:\n{df_metrics.to_string()}")
    
    # Guardar reporte
    df_metrics.to_csv(os.path.join(REPORTS_DIR, 'model_comparison_metrics.csv'))
    
    # 5. Prueba de McNemar
    print("\n  - Ejecutando Test Estadístico de McNemar...")
    stat, p_val = run_mcnemar_test(y_test, preds_base, preds_hybrid)
    print(f"    Estadístico de McNemar: {stat:.5f}, p-valor: {p_val:.5e}")
    significative = p_val < 0.05
    print(f"    ¿La diferencia es estadísticamente significativa? {'SÍ' if significative else 'NO'}")
    
    # Guardar resultados estadísticos
    with open(os.path.join(REPORTS_DIR, 'statistical_comparison.txt'), 'w', encoding='utf-8') as f:
        f.write("PRUEBA ESTADÍSTICA DE MCNEMAR\n")
        f.write("=============================\n")
        f.write(f"Estadístico Chi-cuadrado: {stat:.5f}\n")
        f.write(f"p-valor: {p_val:.5e}\n")
        f.write(f"Significancia (alfa=0.05): {'Diferencia Estadísticamente Significativa' if significative else 'Diferencia No Significativa'}\n")
        f.write("\nInterpretación:\n")
        if significative:
            f.write("El modelo híbrido optimizado mediante Algoritmo Genético presenta un comportamiento predictivo\n")
            f.write("significativamente diferente y superior al modelo base MLP, confirmando la efectividad de la metaheurística.\n")
        else:
            f.write("Las predicciones de ambos modelos no muestran discrepancias estadísticamente significativas en el conjunto de prueba.\n")
            
    # --- GRÁFICOS DE EVALUACIÓN ---
    print("  - Generando gráficos de curvas ROC y Precision-Recall...")
    
    # 1. Curva ROC Comparativa
    fpr_b, tpr_b, _ = roc_curve(y_test, probs_base)
    fpr_h, tpr_h, _ = roc_curve(y_test, probs_hybrid)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_b, tpr_b, color='blue', lw=2, label=f'MLP Base (AUC = {metrics["Base"]["ROC-AUC"]:.4f})')
    plt.plot(fpr_h, tpr_h, color='red', lw=2, label=f'MLP Híbrido GA (AUC = {metrics["Hybrid"]["ROC-AUC"]:.4f})')
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
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False)
    ax[0].set_title('Matriz de Confusión - MLP Base')
    ax[0].set_xlabel('Predicción')
    ax[0].set_ylabel('Realidad')
    ax[0].set_xticklabels(['No Encontrado', 'Encontrado'])
    ax[0].set_yticklabels(['No Encontrado', 'Encontrado'])
    
    sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Reds', ax=ax[1], cbar=False)
    ax[1].set_title('Matriz de Confusión - MLP Híbrido (GA)')
    ax[1].set_xlabel('Predicción')
    ax[1].set_ylabel('Realidad')
    ax[1].set_xticklabels(['No Encontrado', 'Encontrado'])
    ax[1].set_yticklabels(['No Encontrado', 'Encontrado'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '16_matrices_confusion.png'), dpi=300)
    plt.close()
    
    # 4. Curvas de entrenamiento comparativas (Pérdida e Historial)
    print("  - Generando historial de entrenamiento comparativo...")
    hist_base = joblib.load(os.path.join(MODELS_DIR, 'mlp_base_history.joblib'))
    hist_hybrid = joblib.load(os.path.join(MODELS_DIR, 'mlp_hybrid_history.joblib'))
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pérdida
    ax[0].plot(hist_base['loss'], label='Base Train Loss', color='blue', linestyle='--')
    ax[0].plot(hist_base['val_loss'], label='Base Val Loss', color='blue')
    ax[0].plot(hist_hybrid['loss'], label='Híbrido Train Loss', color='red', linestyle='--')
    ax[0].plot(hist_hybrid['val_loss'], label='Híbrido Val Loss', color='red')
    ax[0].set_title('Historial de Pérdida (Loss)')
    ax[0].set_xlabel('Época')
    ax[0].set_ylabel('Binary Crossentropy')
    ax[0].legend()
    ax[0].grid(True)
    
    # Precisión / Exactitud (Accuracy)
    ax[1].plot(hist_base['accuracy'], label='Base Train Acc', color='blue', linestyle='--')
    ax[1].plot(hist_base['val_accuracy'], label='Base Val Acc', color='blue')
    ax[1].plot(hist_hybrid['accuracy'], label='Híbrido Train Acc', color='red', linestyle='--')
    ax[1].plot(hist_hybrid['val_accuracy'], label='Híbrido Val Acc', color='red')
    ax[1].set_title('Historial de Exactitud (Accuracy)')
    ax[1].set_xlabel('Época')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '17_curvas_entrenamiento_comparativas.png'), dpi=300)
    plt.close()
    
    # Generar archivo de reporte en Excel con xlsxwriter
    excel_path = os.path.join(REPORTS_DIR, 'reporte_comparativo_modelos.xlsx')
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df_metrics.to_excel(writer, sheet_name='Metricas')
        # Escribir notas del test de McNemar
        worksheet = writer.book.add_worksheet('Test McNemar')
        worksheet.write(0, 0, 'Estadístico Chi-cuadrado')
        worksheet.write(0, 1, stat)
        worksheet.write(1, 0, 'p-valor')
        worksheet.write(1, 1, p_val)
        worksheet.write(2, 0, 'Estadísticamente Significativo')
        worksheet.write(2, 1, 'SÍ' if significative else 'NO')
        
    print(f"  - Reporte de comparación en Excel guardado en: {excel_path}")
    print("[Evaluación] ¡Pipeline de evaluación finalizado con éxito!\n")

if __name__ == '__main__':
    evaluate_models()
