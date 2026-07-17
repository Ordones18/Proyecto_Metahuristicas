import os
import sys

# Desactivar compilación XLA para evitar errores de Autotuner en WSL2/GPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import time
import random
import pandas as pd
import numpy as np
import tensorflow as tf

# Configuración de crecimiento de memoria dinámico para GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass
from tensorflow.keras import layers, models, optimizers, callbacks
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score
import joblib

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    RANDOM_SEED, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR,
    GA_PARAM_SPACE, PSO_CONFIG
)

# Fijar semillas para reproducibilidad
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)

# Variables globales para evaluar fitness (se cargarán de forma diferida/lazy)
X_fitness_sample = None
y_fitness_sample = None

def build_and_compile_mlp(input_shape, params):
    """
    Construye y compila el modelo MLP de acuerdo a los hiperparámetros decoded.
    """
    from tensorflow.keras import regularizers
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    l2 = params.get('l2_reg', 0.0)
    for _ in range(params['n_layers']):
        model.add(layers.Dense(
            params['neurons_per_layer'],
            activation=params['activation'],
            kernel_regularizer=regularizers.l2(l2) if l2 > 0.0 else None
        ))
        if params['dropout'] > 0:
            model.add(layers.Dropout(params['dropout']))
            
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Configurar optimizador
    if params['optimizer'] == 'adam':
        opt = optimizers.Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=params['learning_rate'])
    else:
        opt = optimizers.Adam(learning_rate=params['learning_rate'])
        
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'], jit_compile=False)
    return model


def decode_position(position):
    """
    Decodifica la posición continua del PSO en un diccionario de hiperparámetros discretos reales.
    """
    params = {}
    keys = list(GA_PARAM_SPACE.keys())
    for i, key in enumerate(keys):
        space = GA_PARAM_SPACE[key]
        idx = int(np.round(position[i]))
        # Asegurar que el índice esté dentro del rango válido
        idx = max(0, min(idx, len(space) - 1))
        params[key] = space[idx]
    return params


def eval_position(position):
    """
    Función de aptitud (fitness).
    Calcula el F1-Score promedio en validación cruzada de un split estratificado de prueba.
    """
    global X_fitness_sample, y_fitness_sample
    if X_fitness_sample is None or y_fitness_sample is None:
        # Cargar los datos globales para evaluar fitness
        # reset_index garantiza que los índices del DataFrame sean posicionales (0, 1, 2, ...)
        # lo que evita el BUG de usar índices originales del DF sobre un numpy array
        X_train_full = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train_selected.csv')).reset_index(drop=True)
        y_train_full = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).values.ravel()

        # Tomar una muestra representativa fija para evaluar el fitness rápidamente
        samples = []
        for val in np.unique(y_train_full):
            mask = (y_train_full == val)
            class_subset = X_train_full[mask]
            class_sample = class_subset.sample(min(len(class_subset), 1000), random_state=RANDOM_SEED)
            samples.append(class_sample)
            
        df_selected = pd.concat(samples, axis=0)
        X_fitness_sample = df_selected.values
        # Usar get_indexer para garantizar índices posicionales correctos sobre el numpy array
        positional_idx = X_train_full.index.get_indexer(df_selected.index)
        y_fitness_sample = y_train_full[positional_idx]

    params = decode_position(position)
    
    # Usar el mismo split del GA (75% / 25%) para reproducibilidad e igualdad de condiciones
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_fitness_sample, y_fitness_sample, 
        test_size=0.25, 
        random_state=RANDOM_SEED, 
        stratify=y_fitness_sample
    )
    
    # Épocas reducidas para evaluación de fitness: 8 es un balance entre
    # ranking confiable (más que 4) y velocidad de búsqueda
    epochs = min(params['epochs'], 8)
    
    model = build_and_compile_mlp(X_fitness_sample.shape[1], params)
    
    try:
        model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=params['batch_size'],
            verbose=0
        )
        # Threshold sweep: en lugar de usar 0.5 fijo, buscar el umbral que
        # maximiza el Macro F1-Score sobre la muestra de validación de fitness.
        # Crítico para datasets desbalanceados (93/7%).
        probs_ravel = model.predict(X_va, verbose=0).ravel()
        best_score = 0.0
        for thresh in np.arange(0.25, 0.76, 0.05):
            preds_t = (probs_ravel >= thresh).astype(int)
            s = f1_score(y_va, preds_t, average='macro', zero_division=0)
            if s > best_score:
                best_score = s
        score = best_score
    except Exception as e:
        score = 0.0
    finally:
        del model
        K.clear_session()
        
    return score


class Particle:
    """
    Representa una partícula individual del PSO.
    """
    def __init__(self, bounds):
        self.position = np.array([random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.array([random.uniform(-abs(b[1] - b[0])/10.0, abs(b[1] - b[0])/10.0) for b in bounds])
        self.best_position = np.copy(self.position)
        self.best_fitness = -1.0
        self.fitness = -1.0

    def update_position(self, bounds):
        self.position += self.velocity
        for i in range(len(bounds)):
            self.position[i] = max(bounds[i][0], min(self.position[i], bounds[i][1]))


def run_pso_algorithm(w_max=0.9, w_min=0.4, c1=1.5, c2=1.5):
    """
    Ejecuta la optimización por enjambre de partículas (PSO) con
    Linear Inertia Weight Reduction (LIWR): w decrece linealmente de w_max
    (exploración amplia) a w_min (explotación refinada) a lo largo de las iteraciones.
    """
    swarm_size = PSO_CONFIG['swarm_size']
    max_iter = PSO_CONFIG['max_iter']
    
    # Fijar semillas para reproducibilidad interna
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    keys = list(GA_PARAM_SPACE.keys())
    bounds = [(0, len(GA_PARAM_SPACE[k]) - 1) for k in keys]
    
    # Inicializar enjambre
    swarm = [Particle(bounds) for _ in range(swarm_size)]
    gbest_position = None
    gbest_fitness = -1.0
    
    print(f"    - Inicializando PSO: w_max={w_max}, w_min={w_min}, c1={c1}, c2={c2}")
    
    # Evaluar enjambre inicial
    for particle in swarm:
        particle.fitness = eval_position(particle.position)
        particle.best_fitness = particle.fitness
        if particle.fitness > gbest_fitness:
            gbest_fitness = particle.fitness
            gbest_position = np.copy(particle.position)
            
    history = []
    
    # Iterar con inercia decreciente (LIWR)
    for it in range(1, max_iter + 1):
        # Linear Inertia Weight Reduction: exploración amplia al inicio, explotación fina al final
        w_current = w_max - (w_max - w_min) * (it / max_iter)
        
        fitnesses = []
        for particle in swarm:
            # Generar números aleatorios para los coeficientes cognitivos y sociales
            r1 = np.array([random.random() for _ in range(len(bounds))])
            r2 = np.array([random.random() for _ in range(len(bounds))])
            
            # Calcular nueva velocidad con inercia decreciente
            cognitive = c1 * r1 * (particle.best_position - particle.position)
            social = c2 * r2 * (gbest_position - particle.position)
            particle.velocity = w_current * particle.velocity + cognitive + social
            
            # Actualizar posición y evaluar
            particle.update_position(bounds)
            particle.fitness = eval_position(particle.position)
            fitnesses.append(particle.fitness)
            
            # Actualizar mejor personal
            if particle.fitness > particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = np.copy(particle.position)
                
            # Actualizar mejor global
            if particle.fitness > gbest_fitness:
                gbest_fitness = particle.fitness
                gbest_position = np.copy(particle.position)
                
        # Registrar estadísticas de la iteración
        max_fit = np.max(fitnesses)
        avg_fit = np.mean(fitnesses)
        
        history.append(max_fit)
        print(f"      Iter {it:02d} (w={w_current:.3f}): Max Fitness = {max_fit:.5f}, Avg = {avg_fit:.5f}")
        
    return history, gbest_position


def compare_pso_parameters():
    """
    Compara científicamente diferentes configuraciones de PSO con
    inercia decreciente (w_max, w_min) variando el valor de w_min.
    """
    print("  - Comparando diferentes configuraciones de PSO...")
    # Comparar dos estrategias de inercia decreciente: más agresiva vs. más suave
    w_configs = [
        (0.9, 0.4),   # LIWR estándar: decaimiento amplio
        (0.9, 0.6),   # LIWR suave: menos explotación al final
    ]
    results = {}
    best_overall_pos = None
    best_overall_fitness = -1.0
    
    if HAS_PLOTTING:
        plt.figure(figsize=(10, 6))
        
    for w_max, w_min in w_configs:
        key = f"{w_max}->{w_min}"
        start_time = time.time()
        history, best_pos = run_pso_algorithm(w_max=w_max, w_min=w_min, c1=PSO_CONFIG['c1'], c2=PSO_CONFIG['c2'])
        elapsed = time.time() - start_time
        
        best_fitness = eval_position(best_pos)
        results[key] = {
            'history': history,
            'best_pos': best_pos.tolist(),
            'best_fitness': best_fitness,
            'time': elapsed
        }
        
        print(f"    Inercia w {key}: Mejor Fitness = {best_fitness:.5f} (Tiempo: {elapsed:.2f}s)")
        
        if HAS_PLOTTING:
            plt.plot(range(len(history)), history, marker='s', label=f'w: {key}')
            
        if best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_pos = best_pos
            
    if HAS_PLOTTING:
        plt.title('Comparación de la Evolución del Fitness según Pesos de Inercia (PSO)')
        plt.xlabel('Iteración')
        plt.ylabel('Fitness Máximo (Macro F1-Score)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, '13_evolucion_fitness_pso.png'), dpi=300)
        plt.close()
    else:
        print("  - [PSO] Omitiendo gráfico de evolución de fitness por falta de librerías de dibujo.")
        
    # Guardar resultados comparativos
    joblib.dump(results, os.path.join(MODELS_DIR, 'pso_comparison_results.joblib'))
    return best_overall_pos


class CompactEpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, epochs):
        self.epochs = epochs
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = epoch + 1
        if epoch_num == 1 or epoch_num == self.epochs or epoch_num % 5 == 0:
            print(f"    [Epoca {epoch_num:02d}/{self.epochs}] - loss: {logs.get('loss', 0):.4f} - val_loss: {logs.get('val_loss', 0):.4f} - acc: {logs.get('accuracy', 0):.4f} - val_acc: {logs.get('val_accuracy', 0):.4f}")


def train_pso_model():
    """
    Ejecuta PSO -> Encuentra el mejor vector de hiperparámetros ->
    Entrena el MLP final optimizado sobre el dataset completo con Callbacks.
    """
    print("\n=== INICIANDO PIPELINE MODELO HÍBRIDO (MLP + PSO) ===")
    
    # 1. Correr el optimizador PSO
    best_pos = compare_pso_parameters()
    best_params = decode_position(best_pos)
    print(f"\n  - Mejor posición del PSO decodificada: {best_params}")
    
    # Guardar mejores hiperparámetros
    joblib.dump(best_params, os.path.join(MODELS_DIR, 'mlp_pso_params.joblib'))
    
    # 2. Cargar conjuntos de datos completos seleccionados
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train_selected.csv'))
    X_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_val_selected.csv'))
    
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).values.ravel()
    y_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_val.csv')).values.ravel()
    
    # 3. Entrenar el modelo final optimizado sobre todo el dataset de entrenamiento
    model = build_and_compile_mlp(X_train.shape[1], best_params)
    
    # Callbacks
    checkpoint_path = os.path.join(MODELS_DIR, 'mlp_pso_checkpoint.keras')
    epochs = best_params['epochs']
    my_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
        CompactEpochLogger(epochs=epochs)
    ]
    
    # Cargar pesos de clase si existen
    class_weights_path = os.path.join(MODELS_DIR, 'class_weights.joblib')
    if os.path.exists(class_weights_path):
        class_weights = joblib.load(class_weights_path)
        print(f"  - Aplicando pesos de clase en el entrenamiento PSO: {class_weights}")
    else:
        class_weights = None
  
    print("  - Entrenando el Modelo Híbrido final (MLP + PSO) con todo el set de entrenamiento...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=best_params['batch_size'],
        callbacks=my_callbacks,
        class_weight=class_weights,
        verbose=0
    )
    train_time = time.time() - start_time
    print(f"  - Modelo híbrido PSO entrenado en {train_time:.2f} segundos.")
    
    # Buscar el threshold óptimo sobre el set de validación (paso fino de 0.01)
    print("  - Buscando threshold óptimo de clasificación sobre validación...")
    probs_val = model.predict(X_val, verbose=0).ravel()
    best_thresh, best_f1_val = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.01):
        p = (probs_val >= t).astype(int)
        f = f1_score(y_val, p, average='macro', zero_division=0)
        if f > best_f1_val:
            best_f1_val, best_thresh = f, t
    print(f"  - Threshold óptimo PSO: {best_thresh:.2f} (Val F1-macro: {best_f1_val:.5f})")
    joblib.dump(best_thresh, os.path.join(MODELS_DIR, 'mlp_pso_threshold.joblib'))
    
    # Guardar modelo final, historial y estadísticas
    model_path = os.path.join(MODELS_DIR, 'mlp_pso.keras')
    model.save(model_path)
    joblib.dump(history.history, os.path.join(MODELS_DIR, 'mlp_pso_history.joblib'))
    joblib.dump({'train_time': train_time}, os.path.join(MODELS_DIR, 'mlp_pso_stats.joblib'))
    
    print(f"  - Modelo híbrido PSO guardado en {model_path}")
    print("[Modelo Híbrido PSO] ¡Pipeline de MLP + PSO completado con éxito!\n")
    return model, best_params


if __name__ == '__main__':
    train_pso_model()
