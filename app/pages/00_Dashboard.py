import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import RAW_DATA_PATH, RANDOM_SEED



# Estilos CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .kpi-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .kpi-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FF4B4B;
    }
    .kpi-title {
        font-size: 1rem;
        color: #B0B0B0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #8A2387;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 id="dashboard_title">Dashboard de Visualización Histórica</h1>', unsafe_allow_html=True)
st.write("Exploración dinámica de las estadísticas oficiales de personas desaparecidas en Ecuador.")

# Cargar datos para el dashboard (con caché para evitar demoras en la navegación de Streamlit)
@st.cache_data
def get_dashboard_data():
    df = pd.read_excel(RAW_DATA_PATH, sheet_name='1')
    
    # Procesamiento básico rápido
    df['edad_num'] = pd.to_numeric(df['edad'], errors='coerce')
    df['edad_num'] = df['edad_num'].fillna(df['edad_num'].median())
    
    df['lat_clean'] = pd.to_numeric(df['latitud_desaparicion'].astype(str).str.replace(',', '.'), errors='coerce')
    df['lon_clean'] = pd.to_numeric(df['longitud_desaparicion'].astype(str).str.replace(',', '.'), errors='coerce')
    df['lat_clean'] = df['lat_clean'].fillna(df['lat_clean'].median())
    df['lon_clean'] = df['lon_clean'].fillna(df['lon_clean'].median())
    
    df['fecha_dt'] = pd.to_datetime(df['fecha_desaparicion'])
    df['anio'] = df['fecha_dt'].dt.year
    df['mes'] = df['fecha_dt'].dt.month
    df['trimestre'] = df['fecha_dt'].dt.quarter
    
    df['target_bin'] = df['situacion_actual'].map({
        'ENCONTRADO': 'Encontrado',
        'FALLECIDO': 'No Encontrado',
        'DESAPARECIDO': 'No Encontrado'
    })
    return df

df = get_dashboard_data()

# --- FILTROS EN LA BARRA LATERAL ---
st.sidebar.header("Filtros Dinámicos")

# Filtro de Año
anios = sorted(df['anio'].dropna().unique())
sel_anios = st.sidebar.multiselect("Año de Desaparición", anios, default=anios)

# Filtro de Provincia
provincias = sorted(df['provincia'].dropna().unique())
sel_provincias = st.sidebar.multiselect("Provincia", provincias, default=provincias[:6]) # Preseleccionar 6 provs por defecto

# Filtro de Sexo
sexos = df['sexo'].unique().tolist()
sel_sexos = st.sidebar.multiselect("Sexo", sexos, default=sexos)

# Rango de Edad
min_age, max_age = int(df['edad_num'].min()), int(df['edad_num'].max())
sel_rango_edad = st.sidebar.slider("Rango de Edad", min_age, max_age, (min_age, max_age))

# Aplicar filtros
df_filtered = df[
    (df['anio'].isin(sel_anios)) &
    (df['provincia'].isin(sel_provincias)) &
    (df['sexo'].isin(sel_sexos)) &
    (df['edad_num'].between(sel_rango_edad[0], sel_rango_edad[1]))
]

# --- KPIs ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_casos = len(df_filtered)
encontrados_pct = (df_filtered['target_bin'] == 'Encontrado').sum() / total_casos * 100 if total_casos > 0 else 0
no_encontrados_pct = 100 - encontrados_pct if total_casos > 0 else 0
edad_prom = df_filtered['edad_num'].mean() if total_casos > 0 else 0

with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Registros Filtrados</div>
        <div class="kpi-val">{total_casos:,}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Personas Encontradas</div>
        <div class="kpi-val" style="color: #4CAF50;">{encontrados_pct:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">No Encontrados</div>
        <div class="kpi-val" style="color: #FF5252;">{no_encontrados_pct:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Edad Promedio</div>
        <div class="kpi-val" style="color: #FFEB3B;">{edad_prom:.1f} años</div>
    </div>
    """, unsafe_allow_html=True)

# --- MAPA Y GRÁFICOS ---
col_map, col_chart = st.columns([1.2, 1])

with col_map:
    st.markdown('<div class="section-title">Mapa de Calor de Desapariciones</div>', unsafe_allow_html=True)
    
    if len(df_filtered) > 0:
        # Filtrar coordenadas válidas para Ecuador
        map_coords = df_filtered[
            (df_filtered['lat_clean'].between(-5.0, 2.0)) & 
            (df_filtered['lon_clean'].between(-82.0, -75.0))
        ][['lat_clean', 'lon_clean']].dropna()
        
        # Muestrear si hay muchos puntos para mantener fluida la renderización
        if len(map_coords) > 5000:
            map_coords = map_coords.sample(n=5000, random_state=RANDOM_SEED)
            
        m = folium.Map(location=[-1.8312, -78.1834], zoom_start=6, tiles='CartoDB dark_matter')
        heat_data = [[row['lat_clean'], row['lon_clean']] for idx, row in map_coords.iterrows()]
        
        if heat_data:
            HeatMap(heat_data, radius=10, blur=7, max_zoom=10).add_to(m)
            
        st_folium(m, height=450, width=700, returned_objects=[])
    else:
        st.warning("No hay coordenadas disponibles para el filtro actual.")

with col_chart:
    st.markdown('<div class="section-title">Tendencia de Desapariciones</div>', unsafe_allow_html=True)
    
    # Serie de tiempo anual-mensual
    df_ts = df_filtered.groupby(['anio', 'mes']).size().reset_index(name='casos')
    df_ts['fecha_str'] = df_ts['anio'].astype(str) + '-' + df_ts['mes'].astype(str).str.zfill(2)
    
    fig_line = px.line(
        df_ts, x='fecha_str', y='casos',
        title="Historial de Casos Mensuales",
        labels={'fecha_str': 'Fecha', 'casos': 'Frecuencia'},
        color_discrete_sequence=['#FF4B4B']
    )
    fig_line.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_line, use_container_width=True)

# --- SEGUNDA FILA DE GRÁFICOS ---
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown('<div class="section-title">Distribución por Rango de Edad y Sexo</div>', unsafe_allow_html=True)
    if len(df_filtered) > 0:
        fig_bar = px.histogram(
            df_filtered, x='rango_edad', color='sexo',
            barmode='group',
            category_orders={'rango_edad': ['NIÑOS(AS)', 'ADOLESCENTES', 'ADULTO', 'ADULTO MAYOR']},
            color_discrete_map={'MUJER': '#FF4081', 'HOMBRE': '#2196F3', 'SIN_DATO': '#9E9E9E'}
        )
        fig_bar.update_layout(template="plotly_dark", height=350, xaxis_title="Rango de Edad", yaxis_title="Casos")
        st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    st.markdown('<div class="section-title">Etnia de las Personas Desaparecidas</div>', unsafe_allow_html=True)
    if len(df_filtered) > 0:
        df_etnia = df_filtered['etnia'].value_counts().reset_index(name='casos')
        df_etnia.columns = ['etnia', 'casos']
        fig_pie = px.pie(
            df_etnia, values='casos', names='etnia',
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        fig_pie.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
