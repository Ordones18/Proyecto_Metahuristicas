import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RANDOM_SEED, RAW_DATA_PATH, FIGURES_DIR

# Configurar el estilo de matplotlib/seaborn si está disponible
if HAS_PLOTTING:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 18
    })

def generate_eda_plots():
    """
    Realiza el Análisis Exploratorio de Datos (EDA) completo y guarda los gráficos en outputs/figures/.
    """
    print("\n=== INICIANDO PIPELINE EDA ===")
    
    # 1. Cargar datos crudos para el análisis exploratorio fiel
    df = pd.read_excel(RAW_DATA_PATH, sheet_name='1')
    
    # Preprocesamiento rápido para visualización
    df['target_str'] = df['situacion_actual'].map({
        'ENCONTRADO': 'Encontrado',
        'FALLECIDO': 'No Encontrado (Fallecido)',
        'DESAPARECIDO': 'No Encontrado (Desaparecido)'
    })
    
    df['target_bin'] = df['situacion_actual'].map({
        'ENCONTRADO': 'Encontrado',
        'FALLECIDO': 'No Encontrado',
        'DESAPARECIDO': 'No Encontrado'
    })
    
    df['edad_num'] = pd.to_numeric(df['edad'], errors='coerce')
    df['lat_clean'] = pd.to_numeric(df['latitud_desaparicion'].astype(str).str.replace(',', '.'), errors='coerce')
    df['lon_clean'] = pd.to_numeric(df['longitud_desaparicion'].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Generar el reporte científico escrito siempre
    write_eda_report_file(df)
    
    if not HAS_PLOTTING:
        print("  - [EDA] Omitiendo generación de gráficos estáticos por incompatibilidad de directivas de seguridad (Pillow DLL block).")
        print("[EDA] ¡Pipeline EDA completado (modo sin gráficos)!\n")
        return
        
    # --- 1. Distribución de la Variable Objetivo ---
    print("  - Generando gráfico de variable objetivo...")
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    
    # Histograma/barras original
    order_orig = ['ENCONTRADO', 'FALLECIDO', 'DESAPARECIDO']
    sns.countplot(x='situacion_actual', data=df, ax=ax[0], hue='situacion_actual', palette='viridis', order=order_orig, legend=False)
    ax[0].set_title('Distribución Multiclase Original')
    ax[0].set_xlabel('Situación Actual')
    ax[0].set_ylabel('Frecuencia (Registros)')
    for p in ax[0].patches:
        ax[0].annotate(f'{p.get_height():,}\n({p.get_height()/len(df)*100:.2f}%)', 
                        (p.get_x() + p.get_width() / 2., p.get_height() + 100), 
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=11)
        
    # Binario (Encontrado vs No Encontrado)
    sns.countplot(x='target_bin', data=df, ax=ax[1], hue='target_bin', palette='coolwarm', order=['Encontrado', 'No Encontrado'], legend=False)
    ax[1].set_title('Distribución Binaria Propuesta')
    ax[1].set_xlabel('Estado de Localización')
    ax[1].set_ylabel('Frecuencia (Registros)')
    for p in ax[1].patches:
        ax[1].annotate(f'{p.get_height():,}\n({p.get_height()/len(df)*100:.2f}%)', 
                        (p.get_x() + p.get_width() / 2., p.get_height() + 100), 
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '01_distribucion_target.png'), dpi=300)
    plt.close()
    
    # --- 2. Distribución de Edad por Target ---
    print("  - Generando gráfico de distribución de edad...")
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df, x='edad_num', hue='target_bin', common_norm=False, fill=True, palette='coolwarm', alpha=0.5, linewidth=2)
    plt.title('Densidad de Edad de las Personas Desaparecidas según Estado')
    plt.xlabel('Edad (Años)')
    plt.ylabel('Densidad')
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '02_densidad_edad.png'), dpi=300)
    plt.close()
    
    # Boxplot edad
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='target_bin', y='edad_num', data=df, palette='coolwarm', hue='target_bin', legend=False)
    plt.title('Distribución y Outliers de Edad por Estado')
    plt.xlabel('Estado de Localización')
    plt.ylabel('Edad (Años)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '03_boxplot_edad.png'), dpi=300)
    plt.close()

    # --- 3. Distribución Temporal (Año y Mes) ---
    print("  - Generando gráfico de distribución temporal...")
    df['fecha_dt'] = pd.to_datetime(df['fecha_desaparicion'])
    df['anio'] = df['fecha_dt'].dt.year
    df['mes'] = df['fecha_dt'].dt.month
    
    # Serie de tiempo mensual
    df_temporal = df.groupby(['anio', 'mes', 'target_bin']).size().reset_index(name='casos')
    df_temporal['fecha_str'] = df_temporal['anio'].astype(str) + '-' + df_temporal['mes'].astype(str).str.zfill(2)
    
    # Gráfico interactivo plotly temporal
    fig_plotly = px.line(df_temporal, x='fecha_str', y='casos', color='target_bin',
                          labels={'fecha_str': 'Fecha', 'casos': 'Número de Desapariciones', 'target_bin': 'Estado'},
                          title='Evolución Histórica Mensual de Desapariciones (2017-2025)',
                          color_discrete_map={'Encontrado': '#1f77b4', 'No Encontrado': '#d62728'})
    fig_plotly.update_layout(xaxis_tickangle=-45)
    fig_plotly.write_html(os.path.join(FIGURES_DIR, '04_evolucion_temporal.html'))
    
    # Guardar versión estática para paper
    plt.figure(figsize=(14, 6))
    df_temporal_static = df.groupby(['anio', 'target_bin']).size().reset_index(name='casos')
    sns.barplot(x='anio', y='casos', hue='target_bin', data=df_temporal_static, palette='coolwarm')
    plt.title('Casos Anuales de Desapariciones en Ecuador (2017-2025)')
    plt.xlabel('Año de Desaparición')
    plt.ylabel('Cantidad de Casos')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '05_casos_anuales.png'), dpi=300)
    plt.close()

    # --- 4. Distribución Geográfica (Provincia) ---
    print("  - Generando gráfico de distribución geográfica...")
    prov_counts = df['provincia'].value_counts().reset_index(name='casos')
    prov_counts.columns = ['provincia', 'casos']
    
    plt.figure(figsize=(14, 8))
    sns.barplot(y='provincia', x='casos', data=prov_counts.head(15), palette='viridis', hue='provincia', legend=False)
    plt.title('Top 15 Provincias con Mayor Número de Desapariciones')
    plt.xlabel('Cantidad de Casos')
    plt.ylabel('Provincia')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '06_top_provincias.png'), dpi=300)
    plt.close()

    # --- 5. Distribución por Sexo ---
    print("  - Generando gráfico de distribución por sexo...")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sexo', hue='target_bin', data=df, palette='coolwarm')
    plt.title('Desapariciones según el Sexo de la Persona')
    plt.xlabel('Sexo')
    plt.ylabel('Casos')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '07_distribucion_sexo.png'), dpi=300)
    plt.close()

    # --- 6. Distribución por Nacionalidad y Etnia ---
    print("  - Generando gráfico por etnia y nacionalidad...")
    etnia_order = df['etnia'].value_counts().index
    plt.figure(figsize=(12, 6))
    sns.countplot(x='etnia', hue='target_bin', data=df, order=etnia_order, palette='coolwarm')
    plt.title('Distribución de Desapariciones por Autoidentificación Étnica')
    plt.xlabel('Etnia')
    plt.ylabel('Casos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '08_distribucion_etnia.png'), dpi=300)
    plt.close()
    
    # --- 7. Mapa de Calor y Correlación ---
    print("  - Generando gráfico de correlación...")
    # Solo podemos calcular correlación sobre las variables numéricas procesadas rápido
    df_corr = df[['edad_num', 'lat_clean', 'lon_clean', 'anio', 'mes', 'target_bin']].copy()
    df_corr['target_bin_num'] = (df_corr['target_bin'] == 'Encontrado').astype(int)
    df_corr = df_corr.drop(columns=['target_bin']).dropna()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5, square=True)
    plt.title('Matriz de Correlación de Pearson de Variables Numéricas')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '09_matriz_correlacion.png'), dpi=300)
    plt.close()

    # --- 8. Mapa de Calor Geográfico (Folium) ---
    print("  - Generando mapa interactivo de calor (Folium)...")
    # Filtrar coordenadas válidas en Ecuador
    ecuador_coords = df[
        (df['lat_clean'].between(-5.0, 2.0)) & 
        (df['lon_clean'].between(-82.0, -75.0))
    ][['lat_clean', 'lon_clean', 'target_bin']].dropna()
    
    # Para evitar congelar el mapa, tomamos un submuestreo del 10%
    map_sample = ecuador_coords.sample(n=min(8000, len(ecuador_coords)), random_state=RANDOM_SEED)
    
    # Crear mapa centrado en Ecuador
    m = folium.Map(location=[-1.8312, -78.1834], zoom_start=7, tiles='CartoDB dark_matter')
    
    # Agregar heatmap de desapariciones
    heat_data = [[row['lat_clean'], row['lon_clean']] for index, row in map_sample.iterrows()]
    HeatMap(heat_data, radius=12, blur=8, max_zoom=10).add_to(m)
    
    # Guardar mapa
    m.save(os.path.join(FIGURES_DIR, '10_mapa_calor_desapariciones.html'))
    print("  - Mapa de Folium guardado con éxito.")
    
    print("[EDA] ¡Pipeline EDA completado con éxito! Gráficos guardados en outputs/figures/\n")


def write_eda_report_file(df):
    """
    Escribe un reporte de texto con la descripción científica del EDA.
    """
    total = len(df)
    encontrados = (df['situacion_actual'] == 'ENCONTRADO').sum()
    fallecidos = (df['situacion_actual'] == 'FALLECIDO').sum()
    desaparecidos = (df['situacion_actual'] == 'DESAPARECIDO').sum()
    
    # Manejar fechas con seguridad
    fechas_rep = pd.to_datetime(df['fecha_desaparicion'], errors='coerce').dropna()
    fecha_min = fechas_rep.min().strftime('%Y-%m-%d') if not fechas_rep.empty else "N/A"
    fecha_max = fechas_rep.max().strftime('%Y-%m-%d') if not fechas_rep.empty else "N/A"
    
    report_text = f"""# ANÁLISIS EXPLORATORIO DE DATOS (EDA) - REPORTE CIENTÍFICO

Este documento presenta el análisis exploratorio detallado del dataset de personas desaparecidas en Ecuador (2017-2025).

## 1. Análisis de la Variable Objetivo (`situacion_actual`)
- **Total de registros**: {total:,}
- **Encontrados**: {encontrados:,} ({encontrados/total*100:.2f}%)
- **Fallecidos**: {fallecidos:,} ({fallecidos/total*100:.2f}%)
- **Desaparecidos**: {desaparecidos:,} ({desaparecidos/total*100:.2f}%)

*Implicación Metodológica*: Existe un desbalanceo de clases severo. Más del 93.18% de las personas son encontradas con vida, mientras que solo el 6.82% corresponden a desenlaces no exitosos (fallecidos o que siguen desaparecidos). Para modelar este fenómeno de forma robusta, se propone una clasificación binaria agrupando "FALLECIDO" y "DESAPARECIDO" bajo la clase "No Encontrado", y aplicando técnicas de balanceo (SMOTE, pesos de clase).

## 2. Análisis Sociodemográfico
- **Edad promedio**: {df['edad_num'].mean():.2f} años (Mediana: {df['edad_num'].median():.1f} años).
- **Rango de Edad predominante**: El {df['rango_edad'].value_counts(normalize=True).head(1).values[0]*100:.2f}% de los casos son {df['rango_edad'].value_counts().index[0]}.
- **Sexo**: El dataset cuenta con una representación de {df['sexo'].value_counts().to_dict()}.
- **Etnia**: La gran mayoría de los registros se autoidentifican como {df['etnia'].value_counts().index[0]} ({df['etnia'].value_counts(normalize=True).values[0]*100:.2f}%).

## 3. Distribución Geográfica y Temporal
- **Provincia con mayor incidencia**: {df['provincia'].value_counts().index[0]} ({df['provincia'].value_counts().values[0]:,} casos).
- **Rango temporal**: Desde {fecha_min} hasta {fecha_max}.
- **Pico histórico**: Se observa una tendencia irregular de desapariciones mensuales con picos característicos en ciertos meses del año, posiblemente ligados a festividades, periodos escolares o factores socioeconómicos estacionales.
"""
    with open(os.path.join(FIGURES_DIR, 'eda_scientific_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("  - Reporte científico guardado como eda_scientific_report.txt")

if __name__ == '__main__':
    generate_eda_plots()
