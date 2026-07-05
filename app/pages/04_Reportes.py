import streamlit as st
import os
import sys
import pandas as pd
from fpdf import FPDF

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import REPORTS_DIR, FIGURES_DIR



# Estilos CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .download-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 2rem;
    }
    .download-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 id="reports_title">Módulo de Descarga de Reportes</h1>', unsafe_allow_html=True)
st.write("Exporte los resultados de la investigación en formatos profesionales (CSV, Excel, PDF).")

# Rutas de los archivos
metrics_csv_path = os.path.join(REPORTS_DIR, 'model_comparison_metrics.csv')
mcnemar_txt_path = os.path.join(REPORTS_DIR, 'statistical_comparison.txt')
excel_report_path = os.path.join(REPORTS_DIR, 'reporte_comparativo_modelos.xlsx')

def clean_accents(text):
    import unicodedata
    if not isinstance(text, str):
        return text
    text = text.replace('ñ', 'n').replace('Ñ', 'N').replace('¿', '').replace('¡', '')
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def generate_pdf_report(metrics_df, mcnemar_text):
    """
    Genera un archivo PDF académico usando fpdf2.
    """
    mcnemar_text = clean_accents(mcnemar_text)
    pdf = FPDF()
    pdf.add_page()
    
    # Encabezado
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "UNIVERSIDAD NACIONAL DE CHIMBORAZO", ln=True, align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, "Proyecto de Investigacion en IA y Metaheuristicas", ln=True, align="C")
    pdf.cell(0, 5, "================================================", ln=True, align="C")
    pdf.ln(10)
    
    # Título del Reporte
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Prediccion de Personas Desaparecidas en Ecuador (2017-2025)", ln=True)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Reporte Comparativo del Modelo MLP Base vs MLP + GA (Hibrido)", ln=True)
    pdf.ln(5)
    
    # Resumen
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "1. RESUMEN EJECUTIVO", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, "Este reporte resume los hallazgos cientificos obtenidos al aplicar el algoritmo "
                          "de redes neuronales multicapa (MLP) optimizadas mediante Algoritmos Geneticos para la "
                          "prediccion del exito en la localizacion de personas desaparecidas en el Ecuador. "
                          "Los datos utilizados corresponden a los registros oficiales depurados y sin fuga de informacion (leakage).")
    pdf.ln(5)
    
    # Tabla de Métricas
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "2. TABLA COMPARATIVA DE METRICAS", ln=True)
    pdf.set_font("Helvetica", "B", 9)
    
    # Columnas
    pdf.cell(35, 8, "Metrica", border=1)
    pdf.cell(45, 8, "MLP Base", border=1)
    pdf.cell(45, 8, "MLP Hibrido (GA)", border=1)
    pdf.cell(45, 8, "Mejora (%)", border=1, ln=True)
    
    pdf.set_font("Helvetica", "", 9)
    for col in metrics_df.columns:
        if col == 'Modelo':
            continue
        val_base = metrics_df.iloc[0][col]
        val_hyb = metrics_df.iloc[1][col]
        val_delta = metrics_df.iloc[2][col]
        
        pdf.cell(35, 8, str(col), border=1)
        pdf.cell(45, 8, f"{val_base:.4f}" if isinstance(val_base, (int, float)) else str(val_base), border=1)
        pdf.cell(45, 8, f"{val_hyb:.4f}" if isinstance(val_hyb, (int, float)) else str(val_hyb), border=1)
        pdf.cell(45, 8, f"{val_delta:.2f}%" if isinstance(val_delta, (int, float)) else str(val_delta), border=1, ln=True)
        
    pdf.ln(5)
    
    # Conclusiones Estadísticas
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "3. PRUEBA DE SIGNIFICANCIA ESTADISTICA (McNEMAR)", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, mcnemar_text)
    
    # Guardar en PDF binario
    return pdf.output()


# --- Interfaz de Descargas ---
if not os.path.exists(metrics_csv_path):
    st.error("No se encontraron los datos de evaluación. Debe ejecutar el pipeline completo (`main.py`) para generar reportes.")
else:
    metrics_df = pd.read_csv(metrics_csv_path)
    
    # Renombrar para formateo
    metrics_df.columns = ['Modelo'] + list(metrics_df.columns[1:])
    
    mcnemar_text = ""
    if os.path.exists(mcnemar_txt_path):
        with open(mcnemar_txt_path, 'r', encoding='utf-8') as f:
            mcnemar_text = f.read()
            
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown('<div class="download-title" style="font-size: 1.5rem; font-weight: 600; color: #FF4B4B; margin-bottom: 1rem;">Reporte de Métricas (CSV)</div>', unsafe_allow_html=True)
            st.write("Descargue el cuadro comparativo de métricas clave del conjunto de prueba.")
            
            with open(metrics_csv_path, 'rb') as f:
                st.download_button(
                    label="Descargar CSV",
                    data=f,
                    file_name="comparacion_metricas.csv",
                    mime="text/csv"
                )
        
    with col2:
        with st.container(border=True):
            st.markdown('<div class="download-title" style="font-size: 1.5rem; font-weight: 600; color: #FF4B4B; margin-bottom: 1rem;">Reporte Detallado (Excel)</div>', unsafe_allow_html=True)
            st.write("Cuadros de métricas y validaciones en formato de hoja de cálculo Excel.")
            
            if os.path.exists(excel_report_path):
                with open(excel_report_path, 'rb') as f:
                    st.download_button(
                        label="Descargar Excel",
                        data=f,
                        file_name="reporte_modelos.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("Archivo Excel no disponible.")
        
    with col3:
        with st.container(border=True):
            st.markdown('<div class="download-title" style="font-size: 1.5rem; font-weight: 600; color: #FF4B4B; margin-bottom: 1rem;">Reporte Científico (PDF)</div>', unsafe_allow_html=True)
            st.write("Genera y descarga un reporte académico oficial con formato universitario en PDF.")
            
            try:
                pdf_bytes = generate_pdf_report(metrics_df, mcnemar_text)
                st.download_button(
                    label="Descargar PDF",
                    data=bytes(pdf_bytes),
                    file_name="reporte_investigacion_desaparecidos.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Error al generar reporte PDF: {e}")
