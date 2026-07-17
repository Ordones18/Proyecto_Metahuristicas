#!/bin/bash

# Configuración y ejecución de Proyecto_Metahuristicas en WSL para soporte de GPU
echo "=========================================================="
echo "    PROYECTO METAHURÍSTICAS: CONFIGURACIÓN EN WSL"
echo "=========================================================="

# ─── 0. Base: libs del driver NVIDIA que WSL2 expone desde Windows ─────────
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Variables de entorno de TensorFlow para GPU en WSL2
export TF_CPP_MIN_LOG_LEVEL=2
export TF_XLA_FLAGS='--tf_xla_enable_xla_devices=false'
export CUDA_VISIBLE_DEVICES=0         # Usar la primera GPU disponible
export TF_FORCE_GPU_ALLOW_GROWTH=true # Crecimiento dinámico de VRAM

# ─── Caché persistente de kernels JIT-compilados (Blackwell / sm_120) ──────
# Evita recompilar desde PTX en cada corrida; la caché vive dentro del
# proyecto en vez de depender de ~/.nv que puede no persistir en WSL.
export CUDA_CACHE_PATH="$(pwd)/.cuda_cache"
export CUDA_CACHE_MAXSIZE=2147483648  # 2GB de caché, ajustable
mkdir -p "$CUDA_CACHE_PATH"

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

# 4. Actualizar pip e instalar requerimientos (solo si requirements.txt cambió)
echo "[WSL] Verificando dependencias en el entorno WSL..."
pip install --upgrade pip --quiet

REQ_HASH_FILE="$ENV_DIR/.req_hash"
if [ -f "requirements.txt" ]; then
    CURRENT_HASH=$(md5sum requirements.txt | awk '{print $1}')
    if [ ! -f "$REQ_HASH_FILE" ] || [ "$(cat "$REQ_HASH_FILE")" != "$CURRENT_HASH" ]; then
        echo "[WSL] Cambios detectados en requirements.txt, instalando..."
        pip install -r requirements.txt --quiet
        echo "$CURRENT_HASH" > "$REQ_HASH_FILE"
    else
        echo "[WSL] Dependencias sin cambios, se omite instalación."
    fi
else
    echo "[WARN] No se encontró requirements.txt, se omite instalación de dependencias."
fi

# Verificar si TensorFlow detecta la GPU en WSL
echo "----------------------------------------------------------"
echo "[WSL] Verificando detección de GPU por TensorFlow..."
python3 -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print()
print('>>> TensorFlow version:', tf.__version__)
print('>>> Num GPUs disponibles:', len(gpus))

if gpus:
    for g in gpus:
        print('   Dispositivo:', g)
    # Habilitar crecimiento dinámico de memoria
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('[OK] Crecimiento dinámico de VRAM habilitado.')
    except RuntimeError as e:
        print('[WARN] No se pudo configurar memory_growth:', e)

    # Mostrar info de memoria si pynvml está disponible
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                name, total, free = line.split(', ')
                print(f'   GPU: {name.strip()} | VRAM Total: {total.strip()} | Libre: {free.strip()}')
    except Exception:
        pass

    print('[OK] El entrenamiento usará GPU.')
else:
    print('[WARN] TensorFlow NO detectó ninguna GPU.')
    print('       El entrenamiento se realizará en CPU (más lento).')
    print('       Verifica:')
    print('       1) nvidia-smi  ->  debe mostrar tu GPU')
    print('       2) LD_LIBRARY_PATH incluye /usr/lib/wsl/lib')
    print('       3) El paquete tensorflow[and-cuda] está instalado')
print()
"
GPU_STATUS=$?
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
        # Verificar una última vez que haya GPU antes de lanzar el pipeline costoso
        GPU_COUNT=$(python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null)
        if [ "$GPU_COUNT" -gt 0 ] 2>/dev/null; then
            echo "[GPU] ✓ Usando GPU para el entrenamiento (${GPU_COUNT} dispositivo/s detectado/s)."
        else
            echo "[WARN] No se detectó GPU. El pipeline correrá en CPU."
            read -p "¿Deseas continuar de todas formas? [s/N]: " confirm
            if [[ ! "$confirm" =~ ^[sS]$ ]]; then
                echo "Saliendo. Revisa la configuración de CUDA antes de re-intentar."
                exit 1
            fi
        fi
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