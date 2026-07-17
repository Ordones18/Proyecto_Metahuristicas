import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MODELS_DIR



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
        model_base = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'mlp_base.keras'))
        model_hybrid = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'mlp_hybrid.keras'))
        model_pso = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'mlp_pso.keras'))
        
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        ohe = joblib.load(os.path.join(MODELS_DIR, 'one_hot_encoder.joblib'))
        target_encoder = joblib.load(os.path.join(MODELS_DIR, 'target_encoder.joblib'))
        selected_features = joblib.load(os.path.join(MODELS_DIR, 'selected_features.joblib'))
        
        from lime import lime_tabular
        project_dir = os.path.dirname(MODELS_DIR)
        X_train_explain = pd.read_csv(os.path.join(project_dir, 'data', 'processed', 'X_train_selected.csv'))
        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train_explain.values,
            feature_names=selected_features,
            class_names=['No Encontrado', 'Encontrado'],
            mode='classification',
            random_state=42
        )
        
        import json
        with open(os.path.join(MODELS_DIR, 'parroquia_coords.json'), 'r', encoding='utf-8') as f:
            parroquia_coords = json.load(f)
            
        return {
            'mlp_base': model_base,
            'mlp_hybrid': model_hybrid,
            'mlp_pso': model_pso,
            'scaler': scaler,
            'ohe': ohe,
            'target_encoder': target_encoder,
            'selected_features': selected_features,
            'lime_explainer': lime_explainer,
            'parroquia_coords': parroquia_coords
        }, None
    except Exception as e:
        return None, str(e)


resources, error_msg = load_prediction_resources()

if error_msg:
    st.error(f"Error al cargar los modelos de Inteligencia Artificial. Asegúrese de haber ejecutado el pipeline de entrenamiento (`main.py`) antes de realizar predicciones.\nDetalle: {error_msg}")
else:
    scaler = resources['scaler']
    ohe = resources['ohe']
    target_encoder = resources['target_encoder']
    selected_features = resources['selected_features']
    lime_explainer = resources['lime_explainer']
    
    # Formulario interactivo sin st.form para soportar selectores dependientes en tiempo real
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
        
        # Cargar mapeo de parroquias y coordenadas
        mapping = resources['parroquia_coords']
        cantones_disponibles = sorted(list(mapping.keys()))
        
        # Intentar preseleccionar QUITO
        default_canton_idx = cantones_disponibles.index("QUITO") if "QUITO" in cantones_disponibles else 0
        canton = st.selectbox("Cantón", cantones_disponibles, index=default_canton_idx)
        
        # Cargar las parroquias del cantón seleccionado
        parroquias_disponibles = sorted(list(mapping[canton].keys())) if canton in mapping else []
        if not parroquias_disponibles:
            parroquias_disponibles = ["SIN DATO"]
            
        parroquia = st.selectbox("Parroquia donde desapareció", parroquias_disponibles)
        
        # Asignar coordenadas automáticamente tras bambalinas
        if canton in mapping and parroquia in mapping[canton]:
            latitud, longitud = mapping[canton][parroquia]
        else:
            latitud, longitud = -0.1806, -78.4678 # Default Quito
            
        zona = st.selectbox("Zona Policial", ["ZONA 9", "ZONA 8", "ZONA 1", "ZONA 2", "ZONA 3", "ZONA 4", "ZONA 5", "ZONA 6", "ZONA 7"])
        
        # Mostrar coordenadas calculadas discretamente
        st.info(f"📍 Coordenadas Parroquia: Lat: {latitud:.4f}, Lon: {longitud:.4f}")
        
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
        try:
            fecha_input = datetime.date(int(anio), int(mes), int(dia))
            ref_date = datetime.date(2025, 12, 31)
            antiguedad_dias = (ref_date - fecha_input).days
        except ValueError:
            st.error("La fecha seleccionada es inválida (por ejemplo, el día no existe en el mes seleccionado). Por favor verifique el día y el mes.")
            st.stop()
        
    submit_btn = st.button("Realizar Predicción Simultánea")
        
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
        
        # Asegurar que el orden de las columnas sea idéntico al visto en el fit del scaler
        encoded_input = encoded_input[scaler.feature_names_in_]
        
        # 4. Escalamiento
        scaled_input = pd.DataFrame(scaler.transform(encoded_input), columns=encoded_input.columns)
        
        # 5. Seleccionar variables
        final_input = scaled_input[selected_features]
        
        # Realizar Inferencia en los 3 modelos a la vez
        # Modelo 1: MLP Base
        model_base = resources['mlp_base']
        pred_base = model_base.predict(final_input, verbose=0)[0][0]
        
        # Modelo 2: MLP Híbrido (GA)
        model_ga = resources['mlp_hybrid']
        pred_ga = model_ga.predict(final_input, verbose=0)[0][0]
        
        # Modelo 3: MLP Híbrido (PSO)
        model_pso = resources['mlp_pso']
        pred_pso = model_pso.predict(final_input, verbose=0)[0][0]
        
        # Función predictora para LIME (utiliza el modelo principal GA)
        def predict_fn_ga(x):
            prob = model_ga.predict(x, verbose=0).astype(np.float64)
            return np.hstack((1.0 - prob, prob))
        
        # Mostrar resultado
        st.subheader("Resultados de la Inferencia Simultánea")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.markdown("<h4 style='text-align: center; color: #510A32;'>MLP Base</h4>", unsafe_allow_html=True)
            if pred_base >= 0.5:
                st.markdown(f'<div class="result-box risk-low">LOCALIZACIÓN EXITOSA<br><br>Probabilidad: {pred_base*100:.2f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box risk-high">RIESGO NO LOCALIZACIÓN<br><br>Probabilidad: {pred_base*100:.2f}%</div>', unsafe_allow_html=True)
                
        with col_res2:
            st.markdown("<h4 style='text-align: center; color: #FF4B4B;'>MLP Híbrido (GA)</h4>", unsafe_allow_html=True)
            if pred_ga >= 0.5:
                st.markdown(f'<div class="result-box risk-low">LOCALIZACIÓN EXITOSA<br><br>Probabilidad: {pred_ga*100:.2f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box risk-high">RIESGO NO LOCALIZACIÓN<br><br>Probabilidad: {pred_ga*100:.2f}%</div>', unsafe_allow_html=True)
                
        with col_res3:
            st.markdown("<h4 style='text-align: center; color: #2ca02c;'>MLP Híbrido (PSO)</h4>", unsafe_allow_html=True)
            if pred_pso >= 0.5:
                st.markdown(f'<div class="result-box risk-low">LOCALIZACIÓN EXITOSA<br><br>Probabilidad: {pred_pso*100:.2f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box risk-high">RIESGO NO LOCALIZACIÓN<br><br>Probabilidad: {pred_pso*100:.2f}%</div>', unsafe_allow_html=True)
                
        # Explicación LIME del modelo principal debajo
        st.markdown("---")
        st.subheader("Explicabilidad Local (LIME) del Modelo Principal (MLP Híbrido GA)")
        
        with st.spinner("Generando explicación del caso con LIME..."):
            exp = lime_explainer.explain_instance(
                data_row=final_input.iloc[0].values,
                predict_fn=predict_fn_ga,
                num_features=5
            )
            
            # Renderizar explicación local con Plotly
            exp_list = exp.as_list()
            df_exp = pd.DataFrame(exp_list, columns=['Variable', 'Contribucion'])
            df_exp['Efecto'] = df_exp['Contribucion'].apply(
                lambda x: 'Favorece Localización (Positivo)' if x > 0 else 'Favorece No Localización (Negativo)'
            )
            fig = px.bar(
                df_exp,
                x='Contribucion',
                y='Variable',
                orientation='h',
                color='Efecto',
                color_discrete_map={
                    'Favorece Localización (Positivo)': '#4CAF50',
                    'Favorece No Localización (Negativo)': '#F44336'
                },
                title="Contribución de Variables a la Predicción (MLP Híbrido GA)"
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
