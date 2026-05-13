# EPH-Pobreza-multidimensional-XAI

# Análisis de los determinantes de la pobreza multidimensional en Argentina: Una perspectiva regional con XAI (2016-2025)

## 📌 Descripción del Proyecto

Este repositorio contiene el desarrollo de mi Trabajo de Fin de Máster (TFM), enfocado en identificar los factores que impulsan la pobreza multidimensional en Argentina. A diferencia de las mediciones tradicionales basadas únicamente en ingresos, este estudio adopta un enfoque holístico e interregional, utilizando algoritmos de **Machine Learning** y técnicas de **Inteligencia Artificial Explicable (XAI)**.

El objetivo central es desmenuzar la "caja negra" de los modelos predictivos para entender cómo varían los determinantes de la vulnerabilidad social según la región geográfica y el contexto macroeconómico.

## 🛠 Metodología y Rigor Estadístico

Para garantizar que el análisis sea apto para la toma de decisiones en políticas públicas, se implementaron los siguientes estándares:

- **Análisis de Correspondencias Múltiples (MCA):** Dado que la pobreza se mide con categorías (ej: tipo de baño, nivel educativo), se utiliza el MCA como un paso previo esencial al modelado con los siguientes objetivos.

  - *Simplificación de datos:* Actúa como un "traductor" que agrupa variables similares, evitando que el modelo se confunda con información redundante.

  - *Selección inteligente de variables:* Los modelos se entrenan con las variables que el MCA identifica como las que realmente explican las diferencias entre hogares en cada año.

  - *Adaptación temporal:* Este análisis se realiza año a año, permitiendo que el modelo detecte si, por ejemplo, la falta de internet pesa más en la pobreza de 2024 que en la de 2016.

<!-- -->

- **Deflactación Regional:** Ajuste de ingresos nominales a pesos constantes utilizando el IPC regional del INDEC, permitiendo la comparabilidad temporal (2016-2025).
- **Ajuste geográfico:** Se aplicaron coeficientes de paridad regional para corregir el sesgo del promedio nacional en las líneas de pobreza.
- **Índice de Pobreza Multidimensional (IPM):** Construcción de una variable target basada en el método Alkire-Foster, integrando dimensiones de vivienda, saneamiento y educación.
- **Muestreo Complejo:** Uso de los factores de expansión (`PONDERA`) de la EPH para asegurar representatividad poblacional.
- **Validación Cruzada por Bloques:** Estrategia de split (Train/Test) estratificada por aglomerados para evitar el data leakage geográfico.
- **Validación Cruzada por Metodología:** Estrategia de split (Train/Test) entre periodos (2016-2024 ; 2025) para evitar el data leakage entre distintas metodologías del INDEC. *Nota:* Las EPH del periodo 2016-2024 se basan en el censo de 2010, mientras que las EPH del periodo 2025- se basan en el censo de 2022.

## 🤖 Modelos Comparados

El entrenamiento de los modelos no usa un listado fijo de variables. Está precedido por una etapa de selección basada en los resultados del MCA. Esto garantiza que cada algoritmo (XGBoost, Random Forest) trabaje solo con los indicadores más informativos y relevantes de cada periodo, mejorando la precisión y evitando el ruido estadístico. Después, se evalúa el desempeño y la interpretabilidad de tres arquitecturas: 1. **Árboles de Decisión (CART):** Para establecer reglas de clasificación base. 2. **Random Forest:** Para capturar interacciones robustas entre variables. 3. **XGBoost:** Optimización mediante gradiente para máxima precisión predictiva.

## 🔍 Análisis XAI (Interpretabilidad)

Utilizando la librería `DALEX`, el proyecto se enfoca en: - **SHAP (Shapley Additive Explanations):** Para cuantificar la contribución global de cada variable a la pobreza. - **LIME / Break-down:** Para explicar predicciones a nivel de hogares individuales (análisis local). - **Partial Dependence Plots (PDP):** Para visualizar la relación marginal entre variables críticas (ej. años de educación) y la probabilidad de pobreza.

## 📂 Estructura del Repositorio

- `data/`: (Excluido vía .gitignore) Microdatos de la EPH y tablas de IPC.
- `src/`: Scripts de R (Descarga, Limpieza, Modelado, XAI).
- `notebooks/`: Análisis Exploratorio de Datos (EDA) y visualizaciones regionales.
- `output/`: Resultados de los modelos y figuras de importancia de variables.

## 🚀 Tecnologías Utilizadas

- **Lenguaje:** R v4.x
- **Librerías Clave:** `tidyverse`, `tidymodels`, `eph`, `dalex`, `xgboost`, `randomForest, FactoMineR (MCA), factoextra`

------------------------------------------------------------------------

**Autor:** Luis Miguel Herrera Corrales\
**Contacto:** [lm.hcor\@gmail.com](mailto:lm.hcor@gmail.com) \| [GitHub](https://github.com/lm-hcor) \| [LinkedIn](https://www.linkedin.com/in/lmhcor)\
**Licencia:** Apache 2.0\
*Nota: Los datos provienen de la Encuesta Permanente de Hogares ([PH) del](https://www.linkedin.com/in/lmhcor)INDEC bajo la Ley de Secreto Estadístico 17.622.*
