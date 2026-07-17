import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
import tensorflow as tf

# Asegurar que el directorio raíz está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import REPORTS_DIR, MODELS_DIR



# Estilos CSS premium (Tema Oscuro y fuentes modernas)
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
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Comparación Científica de Modelos</h1>', unsafe_allow_html=True)
st.write("Análisis estadístico e interactivo entre el Modelo Base (MLP) y los Modelos Híbridos Optimizados (MLP + GA y MLP + PSO).")

# Cargar archivos de métricas básicos
metrics_csv_path = os.path.join(REPORTS_DIR, 'model_comparison_metrics.csv')
mcnemar_txt_path = os.path.join(REPORTS_DIR, 'statistical_comparison.txt')

# Cargar dinámicamente los recursos pesados/modelos y cachearlos para máxima rapidez
@st.cache_resource
def load_comparison_data():
    # Configurar GPU memory growth para el proceso de Streamlit (independiente de main.py)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_dir, 'models')
    processed_dir = os.path.join(project_dir, 'data', 'processed')
    
    try:
        X_test = pd.read_csv(os.path.join(processed_dir, 'X_test_selected.csv'))
        y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.ravel()
        
        model_base = tf.keras.models.load_model(os.path.join(models_dir, 'mlp_base.keras'))
        model_hybrid = tf.keras.models.load_model(os.path.join(models_dir, 'mlp_hybrid.keras'))
        model_pso = tf.keras.models.load_model(os.path.join(models_dir, 'mlp_pso.keras'))
        
        probs_base = model_base.predict(X_test, verbose=0).ravel()
        probs_hybrid = model_hybrid.predict(X_test, verbose=0).ravel()
        probs_pso = model_pso.predict(X_test, verbose=0).ravel()
        
        preds_base = (probs_base >= 0.5).astype(int)
        preds_hybrid = (probs_hybrid >= 0.5).astype(int)
        preds_pso = (probs_pso >= 0.5).astype(int)
        
        hist_base = joblib.load(os.path.join(models_dir, 'mlp_base_history.joblib'))
        hist_hybrid = joblib.load(os.path.join(models_dir, 'mlp_hybrid_history.joblib'))
        hist_pso = joblib.load(os.path.join(models_dir, 'mlp_pso_history.joblib'))
        
        ga_results = joblib.load(os.path.join(models_dir, 'ga_comparison_results.joblib'))
        pso_results = joblib.load(os.path.join(models_dir, 'pso_comparison_results.joblib'))
        
        return {
            'y_test': y_test,
            'probs_base': probs_base,
            'probs_hybrid': probs_hybrid,
            'probs_pso': probs_pso,
            'preds_base': preds_base,
            'preds_hybrid': preds_hybrid,
            'preds_pso': preds_pso,
            'hist_base': hist_base,
            'hist_hybrid': hist_hybrid,
            'hist_pso': hist_pso,
            'ga_results': ga_results,
            'pso_results': pso_results,
            'success': True
        }
    except Exception as e:
        raise e


try:
    comp_data = load_comparison_data()
    comp_data['success'] = True
except Exception as e:
    comp_data = {'success': False, 'error': str(e)}

if not os.path.exists(metrics_csv_path):
    st.error("No se encontraron los datos de evaluación. Debe ejecutar el pipeline completo de entrenamiento primero.")
else:
    df_metrics = pd.read_csv(metrics_csv_path)
    df_metrics.columns = ['Modelo'] + list(df_metrics.columns[1:])
    
    st.dataframe(df_metrics.style.format(precision=4), width='stretch')
    
    st.info("**Nota sobre las métricas:** En *Accuracy, Precision, Recall, F1-Score, ROC-AUC y PR-AUC*, una mejora positiva indica un aumento del desempeño. Para *Log_Loss e Inference_Time_ms*, una mejora positiva indica una reducción del error/tiempo (menor es mejor).")
    
    st.subheader("Validación Estadística (Test de McNemar)")
    if os.path.exists(mcnemar_txt_path):
        with open(mcnemar_txt_path, 'r', encoding='utf-8') as f:
            mcnemar_content = f.read()
        st.code(mcnemar_content, language="markdown")
        
    st.subheader("Curvas de Rendimiento y Convergencia Interactivas")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Curvas ROC y PR", 
        "Matrices de Confusión", 
        "Historial de Entrenamiento", 
        "Evolución Metaheurística (GA & PSO)"
    ])
    
    if not comp_data.get('success', False):
        st.warning("Los gráficos interactivos no están disponibles temporalmente. Por favor, asegúrese de haber entrenado los modelos.")
    else:
        with tab1:
            col1, col2 = st.columns(2)
            
            # ROC Curve
            fpr_b, tpr_b, _ = roc_curve(comp_data['y_test'], comp_data['probs_base'])
            fpr_h, tpr_h, _ = roc_curve(comp_data['y_test'], comp_data['probs_hybrid'])
            fpr_p, tpr_p, _ = roc_curve(comp_data['y_test'], comp_data['probs_pso'])
            auc_base = auc(fpr_b, tpr_b)
            auc_hybrid = auc(fpr_h, tpr_h)
            auc_pso = auc(fpr_p, tpr_p)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr_b, y=tpr_b, mode='lines', name=f'MLP Base (AUC = {auc_base:.4f})', line=dict(color='#1f77b4', width=3)))
            fig_roc.add_trace(go.Scatter(x=fpr_h, y=tpr_h, mode='lines', name=f'MLP + GA (AUC = {auc_hybrid:.4f})', line=dict(color='#d62728', width=3)))
            fig_roc.add_trace(go.Scatter(x=fpr_p, y=tpr_p, mode='lines', name=f'MLP + PSO (AUC = {auc_pso:.4f})', line=dict(color='#2ca02c', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Límite Aleatorio'))
            fig_roc.update_layout(
                title='Curva ROC Comparativa (Test Set)',
                xaxis_title='Tasa de Falsos Positivos (FPR)',
                yaxis_title='Tasa de Verdaderos Positivos (TPR)',
                template='plotly_dark',
                hovermode='x unified',
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
            )
            
            # PR Curve
            p_b, r_b, _ = precision_recall_curve(comp_data['y_test'], comp_data['probs_base'])
            p_h, r_h, _ = precision_recall_curve(comp_data['y_test'], comp_data['probs_hybrid'])
            p_p, r_p, _ = precision_recall_curve(comp_data['y_test'], comp_data['probs_pso'])
            pr_auc_b = auc(r_b, p_b)
            pr_auc_h = auc(r_h, p_h)
            pr_auc_p = auc(r_p, p_p)
            
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=r_b, y=p_b, mode='lines', name=f'MLP Base (PR-AUC = {pr_auc_b:.4f})', line=dict(color='#1f77b4', width=3)))
            fig_pr.add_trace(go.Scatter(x=r_h, y=p_h, mode='lines', name=f'MLP + GA (PR-AUC = {pr_auc_h:.4f})', line=dict(color='#d62728', width=3)))
            fig_pr.add_trace(go.Scatter(x=r_p, y=p_p, mode='lines', name=f'MLP + PSO (PR-AUC = {pr_auc_p:.4f})', line=dict(color='#2ca02c', width=3)))
            fig_pr.update_layout(
                title='Curva Precision-Recall Comparativa (Test Set)',
                xaxis_title='Recall (Sensibilidad)',
                yaxis_title='Precision (Exactitud)',
                template='plotly_dark',
                hovermode='x unified',
                legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01)
            )
            
            with col1:
                st.plotly_chart(fig_roc, width='stretch')
            with col2:
                st.plotly_chart(fig_pr, width='stretch')
                
        with tab2:
            col1, col2, col3 = st.columns(3)
            
            cm_base = confusion_matrix(comp_data['y_test'], comp_data['preds_base'])
            cm_hybrid = confusion_matrix(comp_data['y_test'], comp_data['preds_hybrid'])
            cm_pso = confusion_matrix(comp_data['y_test'], comp_data['preds_pso'])
            labels = ['No Encontrado', 'Encontrado']
            
            fig_cm_b = px.imshow(
                cm_base, 
                x=labels, 
                y=labels, 
                text_auto=True, 
                color_continuous_scale='Blues',
                labels=dict(x="Predicción", y="Realidad", color="Casos")
            )
            fig_cm_b.update_layout(title='Matriz de Confusión - MLP Base', template='plotly_dark', coloraxis_showscale=False)
            
            fig_cm_h = px.imshow(
                cm_hybrid, 
                x=labels, 
                y=labels, 
                text_auto=True, 
                color_continuous_scale='Reds',
                labels=dict(x="Predicción", y="Realidad", color="Casos")
            )
            fig_cm_h.update_layout(title='Matriz de Confusión - MLP + GA', template='plotly_dark', coloraxis_showscale=False)
            
            fig_cm_p = px.imshow(
                cm_pso, 
                x=labels, 
                y=labels, 
                text_auto=True, 
                color_continuous_scale='Greens',
                labels=dict(x="Predicción", y="Realidad", color="Casos")
            )
            fig_cm_p.update_layout(title='Matriz de Confusión - MLP + PSO', template='plotly_dark', coloraxis_showscale=False)
            
            with col1:
                st.plotly_chart(fig_cm_b, width='stretch')
            with col2:
                st.plotly_chart(fig_cm_h, width='stretch')
            with col3:
                st.plotly_chart(fig_cm_p, width='stretch')
                
        with tab3:
            col1, col2 = st.columns(2)
            hist_b = comp_data['hist_base']
            hist_h = comp_data['hist_hybrid']
            hist_p = comp_data['hist_pso']
            
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=hist_b['loss'], mode='lines', name='MLP Base - Entr.', line=dict(dash='dash', color='#1f77b4')))
            fig_loss.add_trace(go.Scatter(y=hist_b['val_loss'], mode='lines', name='MLP Base - Val.', line=dict(color='#1f77b4', width=2)))
            fig_loss.add_trace(go.Scatter(y=hist_h['loss'], mode='lines', name='MLP + GA - Entr.', line=dict(dash='dash', color='#d62728')))
            fig_loss.add_trace(go.Scatter(y=hist_h['val_loss'], mode='lines', name='MLP + GA - Val.', line=dict(color='#d62728', width=2)))
            fig_loss.add_trace(go.Scatter(y=hist_p['loss'], mode='lines', name='MLP + PSO - Entr.', line=dict(dash='dash', color='#2ca02c')))
            fig_loss.add_trace(go.Scatter(y=hist_p['val_loss'], mode='lines', name='MLP + PSO - Val.', line=dict(color='#2ca02c', width=2)))
            fig_loss.update_layout(
                title='Historial de Pérdida (Loss)',
                xaxis_title='Época',
                yaxis_title='Binary Crossentropy',
                template='plotly_dark',
                hovermode='x unified'
            )
            
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(y=hist_b['accuracy'], mode='lines', name='MLP Base - Entr.', line=dict(dash='dash', color='#1f77b4')))
            fig_acc.add_trace(go.Scatter(y=hist_b['val_accuracy'], mode='lines', name='MLP Base - Val.', line=dict(color='#1f77b4', width=2)))
            fig_acc.add_trace(go.Scatter(y=hist_h['accuracy'], mode='lines', name='MLP + GA - Entr.', line=dict(dash='dash', color='#d62728')))
            fig_acc.add_trace(go.Scatter(y=hist_h['val_accuracy'], mode='lines', name='MLP + GA - Val.', line=dict(color='#d62728', width=2)))
            fig_acc.add_trace(go.Scatter(y=hist_p['accuracy'], mode='lines', name='MLP + PSO - Entr.', line=dict(dash='dash', color='#2ca02c')))
            fig_acc.add_trace(go.Scatter(y=hist_p['val_accuracy'], mode='lines', name='MLP + PSO - Val.', line=dict(color='#2ca02c', width=2)))
            fig_acc.update_layout(
                title='Historial de Exactitud (Accuracy)',
                xaxis_title='Época',
                yaxis_title='Accuracy',
                template='plotly_dark',
                hovermode='x unified'
            )
            
            with col1:
                st.plotly_chart(fig_loss, width='stretch')
            with col2:
                st.plotly_chart(fig_acc, width='stretch')
                
        with tab4:
            ga_res = comp_data['ga_results']
            pso_res = comp_data['pso_results']
            fig_meta = go.Figure()
            for mr, res in ga_res.items():
                fig_meta.add_trace(go.Scatter(y=res['history'], mode='lines+markers', name=f'GA (Mut Rate: {mr})', marker=dict(size=8)))
            for w, res in pso_res.items():
                fig_meta.add_trace(go.Scatter(y=res['history'], mode='lines+markers', name=f'PSO (Inertia w: {w})', marker=dict(size=8)))
            fig_meta.update_layout(
                title='Evolución del Fitness Máximo (GA vs PSO)',
                xaxis_title='Generación / Iteración',
                yaxis_title='Fitness Máximo (Macro F1-Score)',
                template='plotly_dark',
                hovermode='x unified'
            )
            st.plotly_chart(fig_meta, width='stretch')
