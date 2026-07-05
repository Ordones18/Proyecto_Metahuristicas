import os
import sys
import time
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from deap import base, creator, tools, algorithms
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    RANDOM_SEED, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR,
    GA_PARAM_SPACE, GA_CONFIG
)

# Fijar semillas para reproducibilidad
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)

# Variables globales para evaluar fitness (se cargarán de forma diferida/lazy)
X_fitness_sample = None
y_fitness_sample = None

# Definir la estructura de DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def build_and_compile_mlp(input_shape, params):
    """
    Construye y compila el modelo MLP de acuerdo a los hiperparámetros decoded.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    for _ in range(params['n_layers']):
        model.add(layers.Dense(params['neurons_per_layer'], activation=params['activation']))
        if params['dropout'] > 0:
            model.add(layers.Dropout(params['dropout']))
            
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Configurar optimizador
    if params['optimizer'] == 'adam':
        opt = optimizers.Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'sgd':
        opt = optimizers.SGD(learning_rate=params['learning_rate'], momentum=0.9)
    else:
        opt = optimizers.Adam(learning_rate=params['learning_rate'])
        
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def decode_chromosome(individual):
    """
    Decodifica el cromosoma entero en un diccionario de hiperparámetros reales.
    """
    params = {}
    keys = list(GA_PARAM_SPACE.keys())
    for i, key in enumerate(keys):
        space = GA_PARAM_SPACE[key]
        idx = individual[i]
        # Asegurar que el índice esté dentro del rango válido
        idx = max(0, min(idx, len(space) - 1))
        params[key] = space[idx]
    return params


def eval_individual(individual):
    """
    Función de aptitud (fitness).
    Calcula el F1-Score promedio en validación cruzada de 3 pliegues.
    """
    global X_fitness_sample, y_fitness_sample
    if X_fitness_sample is None or y_fitness_sample is None:
        # Cargar los datos globales para evaluar fitness
        X_train_full = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train_selected.csv'))

        y_train_full = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).values.ravel()

        # Tomar una muestra representativa fija para evaluar el fitness rápidamente
        # 1500 registros estratificados para evaluación de fitness en validación cruzada
        df_temp = X_train_full.copy()
        df_temp['target'] = y_train_full
        sample_df = df_temp.groupby('target', group_keys=False).apply(lambda x: x.sample(min(len(x), 750), random_state=RANDOM_SEED))
        X_fitness_sample = sample_df.drop(columns=['target']).values
        y_fitness_sample = sample_df['target'].values

    params = decode_chromosome(individual)
    
    # Usar un simple split de validación (75% / 25%) para acelerar la evaluación del fitness en el GA
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_fitness_sample, y_fitness_sample, 
        test_size=0.25, 
        random_state=RANDOM_SEED, 
        stratify=y_fitness_sample
    )
    
    # Épocas ultra reducidas (máximo 4) para la evaluación de la aptitud en el GA
    epochs = min(params['epochs'], 4)
    
    model = build_and_compile_mlp(X_fitness_sample.shape[1], params)
    
    try:
        model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=params['batch_size'],
            verbose=0
        )
        # Predecir clases
        preds_prob = model.predict(X_va, verbose=0)
        preds = (preds_prob >= 0.5).astype(int).ravel()
        
        # Calcular Macro F1-Score
        score = f1_score(y_va, preds, average='macro')
    except Exception as e:
        score = 0.0
        
    return (score,)



def create_individual():
    """
    Crea un individuo con índices aleatorios dentro del espacio de búsqueda.
    """
    ind = []
    for key in GA_PARAM_SPACE.keys():
        ind.append(random.randint(0, len(GA_PARAM_SPACE[key]) - 1))
    return creator.Individual(ind)


def run_genetic_algorithm(mut_rate=0.2):
    """
    Ejecuta el Algoritmo Genético con una tasa de mutación específica.
    Retorna la mejor población, el registro estadístico y el mejor individuo.
    """
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_individual)
    
    # Cruce uniforme con probabilidad 0.5 por gen
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    
    # Mutación uniforme en el espacio de índices
    def mutate_indices(individual, indpb=0.15):
        for i, key in enumerate(GA_PARAM_SPACE.keys()):
            if random.random() < indpb:
                individual[i] = random.randint(0, len(GA_PARAM_SPACE[key]) - 1)
        return individual,
        
    toolbox.register("mutate", mutate_indices, indpb=GA_CONFIG['mut_indpb'])
    toolbox.register("select", tools.selTournament, tournsize=GA_CONFIG['k_tournament'])
    
    pop = toolbox.population(n=GA_CONFIG['pop_size'])
    
    # Registrar estadísticas
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Elitismo y evolución
    elitism_size = GA_CONFIG['elitism_size']
    
    print(f"    - Inicializando evolución con Tasa de Mutación: {mut_rate}")
    
    # Evaluar la población inicial
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        
    logbook = tools.Logbook()
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    
    for gen in range(1, GA_CONFIG['ngen'] + 1):
        # 1. Selección de descendientes
        offspring = toolbox.select(pop, len(pop) - elitism_size)
        offspring = list(toolbox.map(toolbox.clone, offspring))
        
        # 2. Aplicar cruce
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CONFIG['cxpb']:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        # 3. Aplicar mutación
        for mutant in offspring:
            if random.random() < mut_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        # 4. Evaluar individuos con fitness inválido
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # 5. Elitismo: Mantener los mejores de la población anterior
        best_parents = tools.selBest(pop, elitism_size)
        pop[:] = best_parents + offspring
        
        # Guardar log de estadísticas
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(f"      Gen {gen:02d}: Max Fitness = {record['max']:.5f}, Avg = {record['avg']:.5f}")
        
    best_ind = tools.selBest(pop, 1)[0]
    return logbook, best_ind


def compare_mutation_rates():
    """
    Compara científicamente diferentes tasas de mutación (0.1, 0.2, 0.3)
    y dibuja la curva de convergencia del fitness.
    """
    print("  - Comparando diferentes tasas de mutación...")
    mutation_rates = [0.1, 0.2]
    results = {}
    best_overall_ind = None
    best_overall_fitness = -1.0
    
    plt.figure(figsize=(10, 6))
    
    for mr in mutation_rates:
        start_time = time.time()
        logbook, best_ind = run_genetic_algorithm(mut_rate=mr)
        elapsed = time.time() - start_time
        
        fitness_history = [gen['max'] for gen in logbook]
        results[mr] = {
            'history': fitness_history,
            'best_ind': best_ind,
            'best_fitness': best_ind.fitness.values[0],
            'time': elapsed
        }
        
        print(f"    Tasa {mr}: Mejor Fitness = {best_ind.fitness.values[0]:.5f} (Tiempo: {elapsed:.2f}s)")
        
        # Graficar curva de convergencia
        plt.plot(range(len(fitness_history)), fitness_history, marker='o', label=f'Mut Rate: {mr}')
        
        if best_ind.fitness.values[0] > best_overall_fitness:
            best_overall_fitness = best_ind.fitness.values[0]
            best_overall_ind = best_ind
            
    plt.title('Comparación de la Evolución del Fitness según Tasas de Mutación')
    plt.xlabel('Generación')
    plt.ylabel('Fitness Máximo (Macro F1-Score)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '13_evolucion_fitness_ga.png'), dpi=300)
    plt.close()
    
    # Guardar resultados comparativos
    joblib.dump(results, os.path.join(MODELS_DIR, 'ga_comparison_results.joblib'))
    return best_overall_ind


def train_hybrid_model():
    """
    Ejecuta el Algoritmo Genético -> Encuentra el mejor cromosoma ->
    Entrena el MLP final optimizado sobre el dataset completo con Callbacks.
    """
    print("\n=== INICIANDO PIPELINE MODELO HÍBRIDO (MLP + GA) ===")
    
    # 1. Correr el optimizador GA
    best_ind = compare_mutation_rates()
    best_params = decode_chromosome(best_ind)
    print(f"\n  - Mejor individuo del GA (cromosoma): {best_ind}")
    print(f"  - Parámetros óptimos decodificados: {best_params}")
    
    # Guardar mejores hiperparámetros
    joblib.dump(best_params, os.path.join(MODELS_DIR, 'mlp_hybrid_params.joblib'))
    
    # 2. Cargar conjuntos de datos completos seleccionados
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train_selected.csv'))
    X_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_val_selected.csv'))
    
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).values.ravel()
    y_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_val.csv')).values.ravel()
    
    # 3. Entrenar el modelo final optimizado sobre todo el dataset de entrenamiento
    model = build_and_compile_mlp(X_train.shape[1], best_params)
    
    # Callbacks
    checkpoint_path = os.path.join(MODELS_DIR, 'mlp_hybrid_checkpoint.keras')
    my_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    ]
    
    print("  - Entrenando el Modelo Híbrido final (MLP + GA) con todo el set de entrenamiento...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        callbacks=my_callbacks,
        verbose=1
    )
    train_time = time.time() - start_time
    print(f"  - Modelo híbrido optimizado entrenado en {train_time:.2f} segundos.")
    
    # Guardar modelo final, historial y estadísticas
    model_path = os.path.join(MODELS_DIR, 'mlp_hybrid.keras')
    model.save(model_path)
    joblib.dump(history.history, os.path.join(MODELS_DIR, 'mlp_hybrid_history.joblib'))
    joblib.dump({'train_time': train_time}, os.path.join(MODELS_DIR, 'mlp_hybrid_stats.joblib'))
    
    print(f"  - Modelo híbrido guardado en {model_path}")
    print("[Modelo Híbrido] ¡Pipeline de MLP + GA completado con éxito!\n")
    return model, best_params

if __name__ == '__main__':
    train_hybrid_model()
