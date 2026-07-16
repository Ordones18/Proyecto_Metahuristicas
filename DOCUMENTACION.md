# Documentación Científica y Apoyo Metodológico
Este documento recopila las explicaciones conceptuales, justificaciones metodológicas y fundamentos teóricos de los algoritmos e implementaciones del proyecto. Su propósito es servir como material de referencia para la redacción de la tesis, informes técnicos o defensa académica.

---

## 1. El Modelo Base y el Optimizador Evolutivo (MLP + GA)
El núcleo de la propuesta de investigación es una **arquitectura híbrida** que combina dos niveles de optimización con naturalezas y objetivos diferentes:

### A. Perceptrón Multicapa (MLP)
* **¿Qué es?** Es una red neuronal artificial feedforward que consta de una capa de entrada, una o más capas ocultas y una capa de salida. Es capaz de aprender relaciones no lineales complejas entre las características del caso (edad, sexo, geografía, temporalidad) y la probabilidad de localización.
* **Optimización Local (Ajuste de Pesos):** El entrenamiento clásico de la red neuronal utiliza el algoritmo de **Retropropagación (Backpropagation)** junto con un optimizador (como Adam). Este proceso calcula el gradiente del error (pérdida de entropía cruzada) y ajusta iterativamente los **pesos y sesgos internos** de las conexiones neuronales para que la red "aprenda" de los patrones históricos.

### B. Algoritmo Genético (GA)
* **¿Qué es?** Es una técnica de búsqueda y optimización metaheurística bio-inspirada en la selección natural y la genética.
* **Optimización Global (Evolución de Arquitectura e Hiperparámetros):** Una red neuronal es altamente sensible a sus hiperparámetros. Configurar a mano cuántas neuronas poner, qué tasa de aprendizaje usar o cuántas capas ocultas definir es ineficiente y sesgado. El GA automatiza esto tratando a la configuración como un **cromosoma** (un vector de decisiones). 
* A través de generaciones sucesivas, aplicando operadores de **Selección** (los mejores sobreviven), **Cruce** (intercambio de parámetros entre configuraciones exitosas) y **Mutación** (introducción de cambios aleatorios para explorar nuevas arquitecturas), el GA encuentra de forma inteligente la estructura óptima de la red que maximiza el rendimiento predictivo (medido a través de la métrica robusta `F1-Score Macro`).

### C. Sinergia del Modelo Híbrido (MLP+GA)
El resultado es un sistema robusto que combina la **búsqueda global inteligente** del Algoritmo Genético para estructurar la red ideal, y el **ajuste fino local** de Backpropagation para entrenar dicha red.

---

## 2. El Contrincante Científico: XGBoost + RandomizedSearchCV
Para validar la efectividad de la optimización evolutiva sobre las redes neuronales, el proyecto incorpora una comparación rigurosa con el estándar industrial de aprendizaje supervisado sobre datos tabulares:

### A. XGBoost (Extreme Gradient Boosting)
* **¿Qué es?** Es una implementación optimizada y eficiente del algoritmo de Gradient Boosting. En lugar de utilizar redes neuronales, construye un ensamble secuencial de **árboles de decisión asimétricos**. Cada nuevo árbol se entrena para corregir los errores residuales (gradientes) cometidos por los árboles anteriores.
* **Ventajas en Ciencia de Datos:** Es extremadamente rápido, maneja de forma eficiente datos estructurados y cuenta con un fuerte soporte de regularización matemática integrada para evitar el sobreajuste.

### B. RandomizedSearchCV (Búsqueda Aleatoria con Validación Cruzada)
* **¿Qué es?** Es un método estadístico estándar para la optimización de hiperparámetros. 
* **¿Cómo funciona?** En lugar de probar exhaustivamente todas las combinaciones posibles de parámetros de XGBoost (lo cual sería computacionalmente inviable), selecciona al azar una muestra de configuraciones de una grilla predefinida (por ejemplo, muestreando profundidades del árbol, tasas de aprendizaje y número de estimadores). Evalúa cada configuración utilizando **validación cruzada de K-folds** (en este caso, 3-folds) para asegurar que el rendimiento sea generalizable y no dependa de una partición específica de los datos.

### C. Rigor de la Comparativa
Enfrentar **MLP+GA** contra **XGBoost + RandomizedSearchCV** es el diseño experimental óptimo porque:
* **Comparación Justa:** Evita comparar un modelo optimizado (MLP+GA) contra un modelo base sin optimizar (XGBoost por defecto), lo que invalidaría científicamente los resultados. Ambos algoritmos se evalúan en su "mejor versión posible".
* **Contraste de Paradigmas:** Compara la optimización basada en procesos evolutivos aplicados a modelos biológicos (Redes Neuronales) contra la optimización basada en muestreo aleatorio aplicada a modelos lógicos (Árboles de Decisión).

---

## 3. Justificación del Método de Balanceo: Class Weights (Pesos de Clase)
El dataset original presenta un **desbalanceo severo de clases** (~93.18% de casos localizados vs. ~6.82% no localizados). Para contrarrestar el sesgo del modelo a predecir siempre la clase mayoritaria, se analizó el balanceo de datos. Metodológicamente se consolidó el uso exclusivo de **Class Weights (Pesos de Clase / Ponderación de Costos)** por las siguientes razones:

### A. Descarte de SMOTE (Remuestreo Sintético)
* **El Problema:** SMOTE genera muestras sintéticas en el espacio de características interpolando linealmente los registros de la clase minoritaria y sus vecinos más cercanos.
* **El Riesgo en este dataset:** Al contar con variables geográficas de alta cardinalidad codificadas mediante target encoding y coordenadas de latitud/longitud precisas en Ecuador, crear registros ficticios genera "ruido científico". Los datos sintéticos pueden ubicar personas desaparecidas en coordenadas imposibles (fuera del mapa de Ecuador o en el océano) o inventar combinaciones de cantón/provincia inexistentes, lo que le quita rigurosidad al estudio.

### B. Descarte de Under-sampling (Submuestreo Aleatorio)
* **El Problema:** Elimina aleatoriamente registros de la clase mayoritaria (Encontrados) hasta igualar la proporción 50/50.
* **El Riesgo en este dataset:** Forzar un balanceo 50/50 mediante submuestreo destruiría cerca de **60,000 registros reales** de casos localizados. Esto significaría perder valiosa información histórica sobre patrones temporales (años, meses) y dinámicas demográficas que las redes neuronales necesitan para generalizar de forma correcta.

### C. Adopción de Class Weights (Pesos de Clase)
* **¿Cómo funciona?** No modifica los datos. En su lugar, modifica la función de costo (Binary Cross-entropy) durante el entrenamiento de la red y el ajuste de XGBoost, penalizando mucho más fuertemente los errores cometidos sobre la clase minoritaria (No Encontrado).
* **Valor Metodológico:** Permite entrenar los modelos sobre el 100% de la historia real (los 75,517 registros), sin destruir información valiosa ni fabricar datos artificiales/sintéticos. Es la técnica más limpia, transparente y matemáticamente rigurosa para el contexto de esta investigación.
