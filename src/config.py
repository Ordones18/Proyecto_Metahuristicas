import os

# Semilla aleatoria para reproducibilidad
RANDOM_SEED = 42

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(BASE_DIR, 'mdi_personasdesaparecidas_pm_2017_2025.xlsx')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
REPORTS_DIR = os.path.join(OUTPUTS_DIR, 'reports')
LOGS_DIR = os.path.join(OUTPUTS_DIR, 'logs')

# Crear directorios si no existen
for d in [PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Variables a eliminar por Data Leakage (porque solo están disponibles post-evento o se correlacionan 100% con encontrar a la persona)
LEAKAGE_COLUMNS = [
    'fecha_localizacion', 
    'latitud_localizacion', 
    'longitud_localizacion', 
    'provincia_localizacion', 
    'dias_solucion', 
    'motivo_desaparicion', 
    'motivacion_desaparicion_observada', 
    'estado_desaparecido'
]

# Otras columnas innecesarias o redundantes para el modelado
DROP_COLUMNS = ['Unnamed: 27', 'fecha_denuncia', 'fecha_conocimiento', 'codigo_provincia', 'codigo_canton', 'circuito', 'subcircuito', 'distrito']

# Espacio de búsqueda para el Algoritmo Genético (GA) e hiperparámetros de la MLP
GA_PARAM_SPACE = {
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    'n_layers': [1, 2, 3, 4],
    'neurons_per_layer': [32, 64, 128, 256],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    'batch_size': [32, 64, 128, 256],
    'activation': ['relu', 'elu', 'selu'],
    'optimizer': ['adam', 'rmsprop', 'sgd'],
    'epochs': [20, 35, 50, 75]
}

# Configuración del GA
GA_CONFIG = {
    'pop_size': 16,        # Aumentado para mejor exploración en entrenamiento definitivo
    'ngen': 8,             # Aumentado para asegurar mayor convergencia del fitness
    'cxpb': 0.7,          # Probabilidad de cruce
    'mutpb': 0.2,         # Probabilidad de mutación
    'mut_indpb': 0.15,    # Probabilidad de mutar un gen individual
    'k_tournament': 3,    # Tamaño del torneo para selección
    'elitism_size': 1     # Número de mejores individuos a preservar intactos
}

