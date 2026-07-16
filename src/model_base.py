import os
import sys

# Desactivar compilación XLA para evitar errores de Autotuner en WSL2/GPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import time
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
import joblib

# Asegurar que el directorio src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RANDOM_SEED, PROCESSED_DATA_DIR, MODELS_DIR

# Configurar semilla de Keras/TensorFlow para reproducibilidad
tf.keras.utils.set_random_seed(RANDOM_SEED)

def build_mlp(input_shape, n_layers=2, neurons=128, activation='relu', dropout=0.2, learning_rate=0.001, optimizer_name='adam'):
    """
    Construye y compila un modelo Multi-Layer Perceptron (MLP) parametrizable.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    # Capas ocultas
    for _ in range(n_layers):
        model.add(layers.Dense(neurons, activation=activation))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
            
    # Capa de salida para clasificación binaria (sigmoide)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Seleccionar optimizador
    if optimizer_name.lower() == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        opt = optimizers.Adam(learning_rate=learning_rate)
        
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
        jit_compile=False
    )
    return model


def run_grid_search(X_train, y_train, X_val, y_val):
    """
    Realiza una búsqueda de cuadrícula (Grid Search) sobre configuraciones predefinidas de MLP
    y retorna los mejores hiperparámetros.
    """
    print("  - Buscando arquitectura y configuración óptima para MLP Base...")
    
    # Definimos un espacio reducido de búsqueda para asegurar tiempos rápidos de ejecución en CPU
    configs = [
        {'n_layers': 2, 'neurons': 64, 'activation': 'relu', 'dropout': 0.2, 'learning_rate': 0.001, 'optimizer_name': 'adam'},
        {'n_layers': 2, 'neurons': 128, 'activation': 'relu', 'dropout': 0.3, 'learning_rate': 0.001, 'optimizer_name': 'adam'},
        {'n_layers': 3, 'neurons': 128, 'activation': 'elu', 'dropout': 0.3, 'learning_rate': 0.0005, 'optimizer_name': 'rmsprop'},
        {'n_layers': 3, 'neurons': 256, 'activation': 'selu', 'dropout': 0.4, 'learning_rate': 0.001, 'optimizer_name': 'adam'}
    ]
    
    best_val_auc = -1.0
    best_config = None
    
    for i, config in enumerate(configs):
        print(f"    Evaluando configuración {i+1}/{len(configs)}: {config}")
        
        model = build_mlp(
            input_shape=X_train.shape[1],
            n_layers=config['n_layers'],
            neurons=config['neurons'],
            activation=config['activation'],
            dropout=config['dropout'],
            learning_rate=config['learning_rate'],
            optimizer_name=config['optimizer_name']
        )
        
        # Entrenamiento rápido para evaluar la configuración
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=128,
            callbacks=[early_stopping],
            verbose=0
        )
        
        val_auc = history.history['val_auc'][-1]
        print(f"      Resultado Val AUC: {val_auc:.5f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_config = config
            
    print(f"  - Configuración óptima encontrada: {best_config} (Val AUC: {best_val_auc:.5f})")
    return best_config


class CompactEpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, epochs):
        self.epochs = epochs
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = epoch + 1
        if epoch_num == 1 or epoch_num == self.epochs or epoch_num % 5 == 0:
            print(f"    [Epoca {epoch_num:02d}/{self.epochs}] - loss: {logs.get('loss', 0):.4f} - val_loss: {logs.get('val_loss', 0):.4f} - acc: {logs.get('accuracy', 0):.4f} - val_acc: {logs.get('val_accuracy', 0):.4f}")


def train_base_model():
    """
    Orquesta el entrenamiento del Modelo MLP Base: Búsqueda de hiperparámetros ->
    Entrenamiento final con Callbacks de monitoreo -> Evaluación e inferencia básica -> Almacenamiento.
    """
    print("\n=== INICIANDO PIPELINE MODELO MLP BASE ===")
    
    # 1. Cargar datos recortados (variables seleccionadas)
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train_selected.csv'))
    X_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_val_selected.csv'))
    
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).values.ravel()
    y_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_val.csv')).values.ravel()
    
    # 2. Búsqueda de la mejor configuración
    best_params = run_grid_search(X_train, y_train, X_val, y_val)
    joblib.dump(best_params, os.path.join(MODELS_DIR, 'mlp_base_params.joblib'))
    
    # 3. Construir el modelo final
    model = build_mlp(
        input_shape=X_train.shape[1],
        n_layers=best_params['n_layers'],
        neurons=best_params['neurons'],
        activation=best_params['activation'],
        dropout=best_params['dropout'],
        learning_rate=best_params['learning_rate'],
        optimizer_name=best_params['optimizer_name']
    )
    
    # 4. Callbacks para el entrenamiento final
    checkpoint_path = os.path.join(MODELS_DIR, 'mlp_base_checkpoint.keras')
    epochs = 30
    my_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        CompactEpochLogger(epochs=epochs)
    ]
    
    # Cargar pesos de clase si existen (para balanceo 'weight')
    class_weights_path = os.path.join(MODELS_DIR, 'class_weights.joblib')
    if os.path.exists(class_weights_path):
        class_weights = joblib.load(class_weights_path)
        print(f"  - Aplicando pesos de clase en el entrenamiento: {class_weights}")
    else:
        class_weights = None
        
    # 5. Entrenamiento final completo
    print("  - Iniciando entrenamiento final del Modelo Base MLP...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,  # 30 épocas máximo para MLP Base
        batch_size=64,
        callbacks=my_callbacks,
        class_weight=class_weights,
        verbose=0
    )
    train_time = time.time() - start_time
    print(f"  - Modelo base entrenado con éxito en {train_time:.2f} segundos.")
    
    # Guardar modelo final y el historial de entrenamiento
    model_path = os.path.join(MODELS_DIR, 'mlp_base.keras')
    model.save(model_path)
    joblib.dump(history.history, os.path.join(MODELS_DIR, 'mlp_base_history.joblib'))
    joblib.dump({'train_time': train_time}, os.path.join(MODELS_DIR, 'mlp_base_stats.joblib'))
    
    print(f"  - Modelo base guardado en {model_path}")
    print("[Modelo Base] ¡Entrenamiento de MLP Base completado con éxito!\n")
    return model, best_params

if __name__ == '__main__':
    train_base_model()
