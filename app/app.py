import streamlit as st
import os
import sys

# Asegurar que el directorio raíz está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar el layout global una sola vez para toda la aplicación
st.set_page_config(
    page_title="Predicción de Personas Desaparecidas - MLP+GA",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función con el contenido de presentación (Inicio)
def show_inicio():
    # Estilo CSS personalizado premium (Tema Oscuro, tipografía moderna, bordes redondeados y gradientes)
    st.markdown("""
    <style>
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #FF4B4B, #8A2387);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .subtitle {
            font-size: 1.3rem;
            color: #B0B0B0;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(138, 35, 135, 0.2);
            border: 1px solid rgba(138, 35, 135, 0.4);
        }
        
        .feature-header {
            color: #FF4B4B;
            font-weight: 600;
            font-size: 1.4rem;
            margin-bottom: 0.8rem;
        }
        
        .author-badge {
            display: inline-block;
            background: linear-gradient(135deg, #4A0E4E, #1F1C2C);
            color: #E0E0E0;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
    </style>
    """, unsafe_allow_html=True)

    # Título principal (ID único para SEO y pruebas)
    st.markdown('<h1 class="main-title" id="main_title_h1">Predicción de Personas Desaparecidas en Ecuador</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Modelo Híbrido de Inteligencia Artificial: Redes MLP optimizadas mediante Algoritmos Genéticos</p>', unsafe_allow_html=True)

    # Distribución en columnas
    col1, col2 = st.columns([2, 1])

    with col1:
        with st.container(border=True):
            st.markdown('<h3 style="color: #FF4B4B; margin-top: 0; font-weight: 600;">Presentación del Proyecto</h3>', unsafe_allow_html=True)
            st.markdown("""
            Este proyecto de investigación implementa un **sistema inteligente híbrido** para la predicción de localización de personas desaparecidas en Ecuador. Utilizando datos oficiales históricos que abarcan desde el año 2017 hasta el 2025, el sistema predice si una persona reportada como desaparecida será **encontrada (con vida)** o **no encontrada (fallecida o con paradero desconocido)**.
            
            El núcleo tecnológico combina la capacidad de representación de patrones tabulares complejos de los **Perceptrones Multicapa (MLP)** con el poder de exploración y optimización global de los **Algoritmos Genéticos (GA)** para sintonizar automáticamente la arquitectura y los hiperparámetros de entrenamiento.
            """)
            
        with st.container(border=True):
            st.markdown('<h3 style="color: #FF4B4B; margin-top: 0; font-weight: 600;">Objetivos Científicos</h3>', unsafe_allow_html=True)
            st.markdown("""
            * **Optimización Metaheurística**: Diseñar y evaluar un Algoritmo Genético codificado en DEAP que determine de manera automatizada la estructura óptima de la red neuronal.
            * **Mitigación del Data Leakage**: Implementar un pipeline riguroso de ingeniería de variables que elimine variables que sesguen el modelo o introduzcan fuga de información posterior a la localización.
            * **Interpretabilidad de Caja Negra**: Explicar las predicciones locales e importancia global de variables mediante SHAP y LIME para proveer explicaciones confiables a investigadores.
            * **Dashboard de Toma de Decisiones**: Integrar análisis visual interactivo, mapas de calor geográficos y un formulario de predicción individual en tiempo real.
            """)

    with col2:
        with st.container(border=True):
            st.markdown('<h3 style="color: #FF4B4B; margin-top: 0; font-weight: 600;">Arquitectura de la Red</h3>', unsafe_allow_html=True)
            st.markdown("""
            ```mermaid
            graph TD
                A[Datos de Entrada Tabulares] --> B[Pipeline ETL & Target Encoding]
                B --> C[Selección de Variables por Consenso]
                C --> D[Población GA - DEAP]
                D -->|Cromosoma| E[Evaluación Fitness: MLP CV Macro F1]
                E -->|Cromosoma optimizado| F[MLP Híbrido Final]
                F --> G[Predicción de Localización]
            ```
            """)
            
        with st.container(border=True):
            st.markdown('<h3 style="color: #FF4B4B; margin-top: 0; font-weight: 600;">Información Académica</h3>', unsafe_allow_html=True)
            st.markdown("**Autores e Investigadores:**")
            st.markdown('<span class="author-badge">Investigador Principal: Ciencia de Datos</span>', unsafe_allow_html=True)
            st.markdown('<span class="author-badge">Especialidad: Metaheurísticas</span>', unsafe_allow_html=True)
            st.markdown('<span class="author-badge">Universidad Nacional de Chimborazo</span>', unsafe_allow_html=True)

    # Pie de página
    st.write("---")
    st.caption("Desarrollado bajo las normas IEEE para publicaciones de Inteligencia Artificial - 2026")
    st.caption("Para navegar, utilice el menú desplegable a la izquierda.")

# Definir las páginas usando st.Page y Google Material Icons para una UI profesional
inicio_page = st.Page(show_inicio, title="Inicio", icon=":material/home:", default=True)
dashboard_page = st.Page("pages/00_Dashboard.py", title="Dashboard", icon=":material/bar_chart:")
entrenamiento_page = st.Page("pages/01_Entrenamiento.py", title="Entrenamiento", icon=":material/settings:")
prediccion_page = st.Page("pages/02_Prediccion.py", title="Predicción", icon=":material/online_prediction:")
comparacion_page = st.Page("pages/03_Comparacion.py", title="Comparación", icon=":material/balance:")
reportes_page = st.Page("pages/04_Reportes.py", title="Reportes", icon=":material/description:")

# Crear el enrutador de navegación en el orden de flujo óptimo
pg = st.navigation([
    inicio_page, 
    dashboard_page, 
    entrenamiento_page, 
    prediccion_page, 
    comparacion_page, 
    reportes_page
])

# Ejecutar el ruteo de la página activa
pg.run()
