#!/bin/bash

# Configuración y ejecución de Proyecto_Metahuristicas en WSL para soporte de GPU
echo "=========================================================="
echo "    PROYECTO METAHURÍSTICAS: CONFIGURACIÓN EN WSL"
echo "=========================================================="

# ─── 0. Base: libs del driver NVIDIA que WSL2 expone desde Windows ─────────
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Desactivar compilación XLA para evitar errores de Autotuner en WSL2/GPU
export TF_CPP_MIN_LOG_LEVEL=2

# 1. Verificar si python3 y venv están instalados
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 no está instalado en tu WSL. Por favor ejecuta:"
    echo "        sudo apt update && sudo apt install python3 python3-pip python3-venv -y"
    exit 1
fi

# 2. Crear entorno virtual de Linux si no existe
ENV_DIR=".venv_wsl"
if [ ! -d "$ENV_DIR" ]; then
    echo "[WSL] Creando entorno virtual de Python para Linux ($ENV_DIR)..."
    python3 -m venv "$ENV_DIR"
    if [ $? -ne 0 ]; then
        echo "[ERROR] No se pudo crear el entorno virtual. Asegúrate de tener instalado python3-venv."
        exit 1
    fi
fi

# 3. Activar el entorno virtual
source "$ENV_DIR/bin/activate"

# ─── Configurar CUDA libs desde pip (debe ejecutarse DESPUÉS de activar el venv) ─
_NVIDIA_LIBS=$(python3 -c "
import site, os
sp = site.getsitepackages()[0]
d = os.path.join(sp, 'nvidia')
if os.path.isdir(d):
    paths = [os.path.join(d, p, 'lib') for p in os.listdir(d) if os.path.isdir(os.path.join(d, p, 'lib'))]
    print(':'.join(paths))
" 2>/dev/null)
if [ -n "$_NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$_NVIDIA_LIBS:$LD_LIBRARY_PATH"
    echo "[GPU] Librerías CUDA de pip detectadas y configuradas."
else
    echo "[WARN] No se encontraron librerías nvidia-* de pip. Verifica la instalación."
fi

# 4. Actualizar pip e instalar requerimientos
echo "[WSL] Instalando dependencias en el entorno WSL..."
pip install --upgrade pip --quiet

# Instalar TensorFlow y dependencias
pip install -r requirements.txt --quiet

# Verificar si TensorFlow detecta la GPU en WSL
echo "----------------------------------------------------------"
echo "[WSL] Verificando detección de GPU por TensorFlow..."
TF_CPP_MIN_LOG_LEVEL=2 python3 -c "
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print()
print('>>> TensorFlow version:', tf.__version__)
print('>>> Num GPUs disponibles:', len(gpus))
if gpus:
    for g in gpus: print('   ', g)
    print('[OK] GPU lista para entrenamiento.')
else:
    print('[WARN] GPU no detectada. Revisa LD_LIBRARY_PATH o drivers.')
print()
"
echo "----------------------------------------------------------"

# 5. Ofrecer opciones de ejecución
echo "Selecciona una opción para ejecutar:"
echo "1) Ejecutar el pipeline completo de ciencia de datos (main.py)"
echo "2) Lanzar la aplicación web interactiva (Streamlit)"
echo "3) Salir"
read -p "Opción [1-3]: " opt

case $opt in
    1)
        echo "[WSL] Iniciando pipeline de entrenamiento..."
        python3 main.py
        ;;
    2)
        echo "[WSL] Lanzando aplicación Streamlit..."
        streamlit run app/app.py
        ;;
    3)
        echo "Saliendo. Puedes volver a activar este entorno ejecutando: source $ENV_DIR/bin/activate"
        exit 0
        ;;
    *)
        echo "Opción no válida. Saliendo."
        exit 1
        ;;
esac
