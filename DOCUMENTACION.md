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

## 2. El Optimizador de Enjambre de Partículas (MLP + PSO)
Para contrastar y validar la efectividad del Algoritmo Genético, el proyecto incorpora una segunda metaheurística basada en poblaciones para sintonizar los mismos 8 hiperparámetros bajo el mismo espacio de búsqueda combinatorio discreto:

### A. Optimización por Enjambre de Partículas (PSO)
* **¿Qué es?** Es un algoritmo de optimización global bio-inspirado en el comportamiento social y de bandadas de aves o peces (Kennedy & Eberhart, 1995).
* **¿Cómo funciona?** A diferencia del GA que utiliza operadores discretos de cruce y mutación, el PSO inicializa un "enjambre" de partículas en un espacio de búsqueda continuo multidimensional. Cada partícula mantiene una posición y una velocidad. La velocidad se actualiza en cada iteración en función de:
  - **Inercia:** Conserva parte de la velocidad anterior.
  - **Componente Cognitivo:** Atracción hacia la mejor posición histórica que la propia partícula ha visitado (`pbest`).
  - **Componente Social:** Atracción hacia la mejor posición global encontrada por todo el enjambre (`gbest`).
* **Sintonización de MLP:** Aunque el espacio de búsqueda del MLP es de naturaleza discreta y combinatoria, el PSO opera en coordenadas reales continuas. Para resolver esto, las posiciones se redondean y acotan a los índices enteros correspondientes de la grilla de hiperparámetros (`GA_PARAM_SPACE`). Esto permite explotar la dinámica cinético-social del PSO para resolver problemas discretos complejos.

### B. Rigor de la Comparativa (GA vs. PSO)
La comparación directa de **MLP+GA** frente a **MLP+PSO** representa un marco científico riguroso por dos motivos:
* **Entorno Experimental Idéntico:** Ambos algoritmos evalúan exactamente los mismos hiperparámetros discretos (capas, neuronas, activación, dropout, tasa de aprendizaje, batch, optimizador y épocas) usando la misma función de fitness basada en Macro F1-Score bajo validación cruzada rápida, evitando sesgos de implementación.
* **Contraste de Metaheurísticas de Población:** Permite contrastar la exploración/explotación de un algoritmo evolutivo (GA) basado en la supervivencia del más apto frente a la dinámica cooperativa y de seguimiento social de un optimizador de enjambres (PSO).

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
