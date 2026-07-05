import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MODELS_DIR

# Configuración de página
st.set_page_config(page_title="Predicción de Localización", page_icon="🔮", layout="wide")

# Estilos CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .result-box {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
        font-size: 1.5rem;
        font-weight: 800;
    }
    .risk-low {
        background-color: rgba(76, 175, 80, 0.15);
        color: #4CAF50;
        border: 2px solid #4CAF50;
    }
    .risk-high {
        background-color: rgba(244, 67, 54, 0.15);
        color: #F44336;
        border: 2px solid #F44336;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 id="predict_title">Módulo de Inferencia en Tiempo Real</h1>', unsafe_allow_html=True)
st.write("Ingrese los datos de la desaparición para estimar la probabilidad de éxito en la localización de la persona.")

# Función para cargar recursos del modelo con caché
@st.cache_resource
def load_prediction_resources():
    try:
        model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'mlp_hybrid.keras'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        ohe = joblib.load(os.path.join(MODELS_DIR, 'one_hot_encoder.joblib'))
        target_encoder = joblib.load(os.path.join(MODELS_DIR, 'target_encoder.joblib'))
        selected_features = joblib.load(os.path.join(MODELS_DIR, 'selected_features.joblib'))
        from lime import lime_tabular
        # Recrear explicador LIME al vuelo leyendo X_train_selected.csv
        project_dir = os.path.dirname(MODELS_DIR)
        X_train_explain = pd.read_csv(os.path.join(project_dir, 'data', 'processed', 'X_train_selected.csv'))
        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train_explain.values,
            feature_names=selected_features,
            class_names=['No Encontrado', 'Encontrado'],
            mode='classification',
            random_state=42
        )
        return model, scaler, ohe, target_encoder, selected_features, lime_explainer, None
    except Exception as e:
        return None, None, None, None, None, None, str(e)


model, scaler, ohe, target_encoder, selected_features, lime_explainer, error_msg = load_prediction_resources()

if error_msg:
    st.error(f"Error al cargar los modelos de Inteligencia Artificial. Asegúrese de haber ejecutado el pipeline de entrenamiento (`main.py`) antes de realizar predicciones.\nDetalle: {error_msg}")
else:
    # Formulario en columnas
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Datos de la Persona")
            edad = st.number_input("Edad de la persona (Años)", min_value=0, max_value=120, value=25)
            sexo = st.selectbox("Sexo", ["MUJER", "HOMBRE"])
            etnia = st.selectbox("Etnia", ["MESTIZO/A", "INDIGENA", "AFRO", "MONTUBIO/A", "MULATO/A", "BLANCO/A", "OTROS", "SIN_DATO"])
            nacionalidad = st.selectbox("Nacionalidad", ["ECUADOR", "VENEZUELA", "COLOMBIA", "PERU", "ESPAÑA", "ESTADOS UNIDOS", "OTROS"])
            
        with col2:
            st.subheader("Ubicación del Suceso")
            provincia = st.selectbox("Provincia", ["PICHINCHA", "GUAYAS", "MANABÍ", "AZUAY", "EL ORO", "LOS RÍOS", "CHIMBORAZO", "TUNGURAHUA", "SANTA ELENA", "ESMERALDAS", "LOJA"])
            canton = st.text_input("Cantón (Ej: GUAYAQUIL, QUITO, ALAUSI)", value="QUITO").upper()
            zona = st.selectbox("Zona Policial", ["ZONA 9", "ZONA 8", "ZONA 1", "ZONA 2", "ZONA 3", "ZONA 4", "ZONA 5", "ZONA 6", "ZONA 7"])
            
            # Coordenadas geográficas
            latitud = st.number_input("Latitud de Desaparición", value=-0.1806, format="%.5f")
            longitud = st.number_input("Longitud de Desaparición", value=-78.4678, format="%.5f")
            
        with col3:
            st.subheader("Temporalidad")
            anio = st.number_input("Año de Desaparición", min_value=2017, max_value=2030, value=2026)
            mes = st.slider("Mes", 1, 12, 1)
            dia = st.slider("Día del Mes", 1, 31, 15)
            dia_semana = st.selectbox("Día de la Semana", ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])
            
            # Mapear día semana a numérico (0=Lunes, 6=Domingo)
            dia_semana_map = {"Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3, "Viernes": 4, "Sábado": 5, "Domingo": 6}
            dia_semana_num = dia_semana_map[dia_semana]
            
            # Calcular trimestre y antigüedad ficticia (días transcurridos desde desaparición hasta 31-12-2025)
            trimestre = (mes - 1) // 3 + 1
            # Antigüedad en días relativa a 31-12-2025 (si el año ingresado es posterior, será negativo)
            import datetime
            fecha_input = datetime.date(int(anio), int(mes), int(dia))
            ref_date = datetime.date(2025, 12, 31)
            antiguedad_dias = (ref_date - fecha_input).days
            
        submit_btn = st.form_submit_button("Realizar Predicción")
        
    if submit_btn:
        # Preprocesar datos de entrada
        # Determinar el rango de edad
        if edad <= 12:
            rango_edad = "NIÑOS(AS)"
        elif edad <= 17:
            rango_edad = "ADOLESCENTES"
        elif edad <= 64:
            rango_edad = "ADULTO"
        else:
            rango_edad = "ADULTO MAYOR"
            
        # Crear DataFrame crudo
        raw_input = pd.DataFrame([{
            'edad': edad,
            'rango_edad': rango_edad,
            'sexo': sexo,
            'etnia': etnia,
            'nacionalidad': nacionalidad,
            'provincia': provincia,
            'canton': canton,
            'zona': zona,
            'latitud_desaparicion': latitud,
            'longitud_desaparicion': longitud,
            'desaparicion_anio': anio,
            'desaparicion_mes': mes,
            'desaparicion_dia': dia,
            'desaparicion_dia_semana': dia_semana_num,
            'desaparicion_trimestre': trimestre,
            'antiguedad_dias': antiguedad_dias
        }])
        
        # 1. Codificación Ordinal
        rango_edad_mapping = {'NIÑOS(AS)': 0, 'ADOLESCENTES': 1, 'ADULTO': 2, 'ADULTO MAYOR': 3}
        raw_input['rango_edad'] = raw_input['rango_edad'].map(rango_edad_mapping).fillna(1).astype(int)
        
        # 2. One-Hot Encoding
        ohe_cols = ['sexo', 'etnia', 'zona']
        ohe_features = ohe.transform(raw_input[ohe_cols])
        ohe_col_names = ohe.get_feature_names_out(ohe_cols)
        df_ohe = pd.DataFrame(ohe_features, columns=ohe_col_names, index=raw_input.index)
        
        # Concatenar y eliminar
        encoded_input = pd.concat([raw_input.drop(columns=ohe_cols), df_ohe], axis=1)
        
        # 3. Target Encoding
        encoded_input = target_encoder.transform(encoded_input)
        
        # 4. Escalamiento
        scaled_input = pd.DataFrame(scaler.transform(encoded_input), columns=encoded_input.columns)
        
        # 5. Seleccionar variables
        final_input = scaled_input[selected_features]
        
        # Realizar Inferencia
        pred_prob = model.predict(final_input, verbose=0)[0][0]
        
        # Mostrar resultado
        st.subheader("Resultados de la Inferencia")
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            if pred_prob >= 0.5:
                st.markdown(f'<div class="result-box risk-low">LOCALIZACIÓN EXITOSA ESTIMADA<br><br>Probabilidad: {pred_prob*100:.2f}%<br>Confianza: Alta</div>', unsafe_allow_html=True)
                risk_level = "Bajo"
            else:
                st.markdown(f'<div class="result-box risk-high">RIESGO DE NO LOCALIZACIÓN<br><br>Probabilidad de Éxito: {pred_prob*100:.2f}%<br>Nivel de Riesgo: Crítico</div>', unsafe_allow_html=True)
                risk_level = "Alto"
                
        with col_res2:
            # Explicación con LIME en tiempo real
            st.subheader("Explicación Local del Modelo (LIME)")
            
            # Predictora para LIME (retorna dos clases)
            def predict_fn(x):
                prob = model.predict(x, verbose=0).astype(np.float64)
                return np.hstack((1.0 - prob, prob))
                
            with st.spinner("Generando explicación del caso..."):
                exp = lime_explainer.explain_instance(
                    data_row=final_input.iloc[0],
                    predict_fn=predict_fn,
                    num_features=5
                )
                
                # Renderizar figura LIME
                fig = exp.as_pyplot_figure()
                plt.title("Contribución de Variables a la Predicción")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
