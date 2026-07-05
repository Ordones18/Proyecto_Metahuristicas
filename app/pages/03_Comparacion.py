import streamlit as st
import os
import sys
import pandas as pd

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import REPORTS_DIR, FIGURES_DIR

# Configuración de página
st.set_page_config(page_title="Comparación de Modelos", page_icon="⚖️", layout="wide")

# Estilos CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .metric-delta {
        font-size: 1.1rem;
        color: #4CAF50;
        font-weight: 600;
    }
    .metric-delta-neg {
        font-size: 1.1rem;
        color: #F44336;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 id="comparison_title">Comparación Científica de Modelos</h1>', unsafe_allow_html=True)
st.write("Análisis estadístico y comparativo detallado entre el Modelo Base (MLP) y el Modelo Híbrido Optimizado (MLP + GA).")

# Cargar archivos de métricas
metrics_csv_path = os.path.join(REPORTS_DIR, 'model_comparison_metrics.csv')
mcnemar_txt_path = os.path.join(REPORTS_DIR, 'statistical_comparison.txt')

if not os.path.exists(metrics_csv_path):
    st.error("No se encontraron los datos de evaluación. Debe ejecutar el pipeline completo (`main.py`) para entrenar y evaluar los modelos primero.")
else:
    df_metrics = pd.read_csv(metrics_csv_path, index_index=False)
    # Renombrar primera columna a Modelo
    df_metrics.columns = ['Modelo'] + list(df_metrics.columns[1:])
    
    st.subheader("Cuadro Comparativo de Métricas")
    st.dataframe(df_metrics.style.format(precision=4), use_container_width=True)
    
    # Explicación de mejoras
    st.info("💡 **Nota sobre las métricas:** En *Accuracy, Precision, Recall, F1-Score, ROC-AUC y PR-AUC*, una mejora positiva indica un aumento del desempeño. Para *Log_Loss e Inference_Time_ms*, una mejora positiva indica una reducción del error/tiempo (menor es mejor).")
    
    # Mostrar resultados de McNemar
    st.subheader("Validación Estadística (Test de McNemar)")
    if os.path.exists(mcnemar_txt_path):
        with open(mcnemar_txt_path, 'r', encoding='utf-8') as f:
            mcnemar_content = f.read()
        st.code(mcnemar_content, language="markdown")
        
    # Mostrar gráficos comparativos en pestañas
    st.subheader("Curvas de Rendimiento y Convergencia")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Curvas ROC y PR", 
        "Matrices de Confusión", 
        "Historial de Entrenamiento", 
        "Evolución del Algoritmo Genético"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        roc_img = os.path.join(FIGURES_DIR, '14_curva_roc_comparativa.png')
        pr_img = os.path.join(FIGURES_DIR, '15_curva_pr_comparativa.png')
        
        with col1:
            if os.path.exists(roc_img):
                st.image(roc_img, caption="Curva ROC Comparativa")
            else:
                st.warning("Curva ROC no disponible.")
        with col2:
            if os.path.exists(pr_img):
                st.image(pr_img, caption="Curva Precision-Recall Comparativa")
            else:
                st.warning("Curva Precision-Recall no disponible.")
                
    with tab2:
        cm_img = os.path.join(FIGURES_DIR, '16_matrices_confusion.png')
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Matrices de Confusión en Test set", use_container_width=True)
        else:
            st.warning("Matrices de confusión no disponibles.")
            
    with tab3:
        train_img = os.path.join(FIGURES_DIR, '17_curvas_entrenamiento_comparativas.png')
        if os.path.exists(train_img):
            st.image(train_img, caption="Curvas de Pérdida e Exactitud durante el Entrenamiento", use_container_width=True)
        else:
            st.warning("Gráficos de entrenamiento no disponibles.")
            
    with tab4:
        ga_img = os.path.join(FIGURES_DIR, '13_evolucion_fitness_ga.png')
        if os.path.exists(ga_img):
            st.image(ga_img, caption="Evolución del Fitness Máximo según diferentes Tasas de Mutación", use_container_width=True)
        else:
            st.warning("Gráfico de evolución del GA no disponible.")
