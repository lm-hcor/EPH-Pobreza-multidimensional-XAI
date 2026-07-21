# EPH-Pobreza-Multidimensional-XAI 

**App Shiny: https://lm-hcor.shinyapps.io/EPH-Pobreza-multidimensional-XAI/

> **Análisis de los determinantes de la pobreza multidimensional en Argentina: Una perspectiva regional e intertemporal con Inteligencia Artificial Explicable (2016-2025).**

Este repositorio contiene el código fuente, la arquitectura de datos y el pipeline analítico de mi **Trabajo de Fin de Máster (TFM)**. El estudio abandona las mediciones unidimensionales tradicionales basadas exclusivamente en ingresos para proponer un enfoque híbrido de Machine Learning y técnicas avanzadas de **Inteligencia Artificial Explicable (XAI)**, desarmando la "caja negra" de los modelos predictivos para identificar los determinantes estructurales de la vulnerabilidad social en Argentina a nivel federal.

## 🛠️ Metodología y Rigor Estadístico

Para garantizar que el modelo sea apto como herramienta de simulación en políticas públicas, el ecosistema de datos se diseñó bajo estrictos controles macroeconómicos y geométricos:

- **Deflactación Regional e Intertemporal:** Sincronización de los flujos de ingresos nominales (`itcf_real`, `p21_real`) a pesos constantes utilizando el Índice de Precios al Consumidor (IPC) desagregado por las seis grandes regiones del INDEC, mitigando el sesgo del promedio nacional en contextos de alta inflación.

- **Índice de Pobreza Multidimensional (IPM):** Construcción de la variable *target* binaria (`mpi_pobre`) adaptando el método Alkire-Foster sobre las dimensiones de calidad de la vivienda, saneamiento básico, acceso a la salud (`priv_salud`) y educación.

- **Alineación Demográfica Post-Censal:** Reponderación regional de los factores de expansión (`pondera`) del set de entrenamiento para aproximar la estructura demográfica del marco censal 2022, mitigando el *data drift*.

### Prevención de Data Leakage Geográfico y Temporal

- **Block Cross-Validation:** Agrupación por conglomerados urbanos (`grupo_cv`) mediante `group_vfold_cv`. Las unidades de un aglomerado completo se aíslan estrictamente en el pool de entrenamiento o en el de validación, respetando el supuesto de independencia espacial.

- **Chronological Time-Splitting:** Separación tajante entre los datos históricos (2016-2024, basados en el Censo 2010) y el set externo (**Externo 2025**, basado en el Censo 2022). Esto permite aislar y evaluar el impacto real del apagón metodológico y el cambio de marco muestral del INDEC.

## 🤖 Pipeline de Modelado y Optimización de Arquitectura

El pipeline fue optimizado mediante `tidymodels` para ejecutarse eficientemente bajo restricciones estrictas de hardware (**16GB RAM** en entornos Windows 11), reduciendo el tiempo de cómputo a menos de 16 horas.

```         
[Datos Originales] ──> [Imputación/Dummies] ──> [SMOTE-NC Único y Externo] ──> [ANOVA Racing CV (2 Cores)] 
```

### 1. Corrección de Desbalance de Clases (SMOTE-NC Externo)

Debido a la naturaleza minoritaria de la clase pobre multidimensional extrema, se aplica **SMOTE-NC** (Themis) con un `over_ratio = 0.25`.

- **Optimización crítica:** Para evitar el cuello de botella del cálculo repetitivo de distancias de vecinos cercanos dentro de cada fold, SMOTE-NC se ejecuta **una sola vez de forma externa** sobre la matriz de entrenamiento pre-procesada. Las variables técnicas (`codusu`, `pondera`, `grupo_cv`) se reasignan sintéticamente mediante muestreo estratificado para asegurar consistencia regional absoluta.

### 2. Reducción de Dimensionalidad (Análisis de Correspondencias Múltiples)

Se integra un **MCA** dinámico (`FactoMineR`) enfocado en variables de infraestructura no lineales (`v2`, `v11`, `v13`, `iv1`, `iv2`, `ii7`, `ii8`). Las coordenadas latentes resultantes (`mca_dim1`, `mca_dim2`) reemplazan a las variables originales crudas para evitar la colinealidad y la inflación artificial de métricas, excluyendo cualquier variable que induzca *data leakage* directo con el IPM.

### 3. Ajuste por Carreras Competitivas (ANOVA Racing)

En lugar de una búsqueda exhaustiva por grilla, el ajuste de hiperparámetros se realiza mediante **ANOVA Race Tuning** (`tune_race_anova` de `finetune`), evaluando las grillas espaciales mediante paralelismo restrictivo a nivel de resample (`parallel_over = "resamples"`) con 2 workers PSOCK. El algoritmo elimina tempranamente las combinaciones estadísticamente inferiores tras un burn-in de 2 folds, optimizando el área bajo la curva ROC ($AUC\text{-}ROC$).

## 📈 Evaluación del Rendimiento Operativo

Los modelos son evaluados mediante métricas threshold-agnostic y curvas de confiabilidad calibradas bajo la maximización del **F1-Score** (priorizando el balance óptimo entre la precisión fiscal contra filtraciones y la sensibilidad de cobertura social).

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| **Modelo** | **Engine** | **Umbral Calibrado (t)** | **AUC-ROC (Test 2024)** | **AUC-ROC (Externo 2025)** | **Calibración Probabilística** |
| **CART** | `rpart` | $0.404$ | $0.711$ | $0.690$ | Deficiente / Escalonada |
| **Random Forest** | `ranger` | $0.430$ | $0.863$ | $0.838$ | Sesgo de sobreestimación sistemático |
| **XGBoost** | `xgboost` | **0.444** | **0.865** | **0.841** | **Excelente / Alineación Cuasi-Perfecta** |

- **Impacto del Cambio Censal (2025):** El modelo detecta una degradación de performance estructural controlada ($\Delta \approx 0.024$ de $AUC$) en el set externo 2025, demostrando la validez del split cronológico para evaluar la resiliencia de modelos estadísticos frente a quiebres institucionales.

- **Calibración:** **XGBoost se consolida como el modelo definitivo** para asignación presupuestaria, ya que sus probabilidades de salida se alinean de manera directa con las frecuencias empíricas observadas, permitiendo interpretar los *scores* como porcentajes reales de riesgo del hogar.

## 🔍 Análisis XAI (Inteligencia Artificial Explicable)

El repositorio explota la librería `DALEX` y arquitecturas unificadas de SHAP para romper la opacidad de los algoritmos ensembles:

- **SHAP Global Attributions:** Cuantificación del peso de las dimensiones MCA de infraestructura frente a los shocks de ingresos reales por ocupación principal.

- **Break-Down Plots:** Disección de cuatro casos tipos: Máxima y mínima probabilidad de pobreza, y dos casos limitrofes, para observar los determinantes de su clasificación.

## 📂 Estructura del Repositorio

```         
├── data/                            # (Excluido en .gitignore por políticas del INDEC) │   ├── raw/                         # Microdatos crudos de la EPH (trimestrales) │   └── processed/                   # Datasets consolidados con coordenadas MCA e ingresos deflactados ├── output/ │   ├── models/                      # Almacenamiento de objetos serializados (best_xgb.rds, etc.) │   └── results/                     # Tablas de umbrales calibrados y matrices de robustez ├── src/ │   ├── 01_data_download.R           # Automatización de descarga de ondas EPH vía librería 'eph' │   ├── 02_deflactacion_mca.R        # Preprocesamiento, deflactación regional y cómputo de dimensiones MCA │   ├── 06c_optimized_modelling.R    # Pipeline de Machine Learning (SMOTE-NC externo + ANOVA Race) │   ├── 06_robustness_analysis.R     # Pruebas de estrés estructural (Exclusión de variables de educación) │   └── 07_xai_plots.R               # Generación de gráficos SHAP, PDP y diagnósticos DALEX └── README.md 
```

## 🚀 Tecnologías Utilizadas

- **Lenguaje Base:** R v4.x

- **Ecosistema de Modelado:** `tidymodels`, `spatialsample`, `finetune`, `themis`

- **Motores de Clasificación:** `xgboost` (v1.7.5+), `ranger` (Random Forest), `rpart` (CART)

- **Análisis Multivariado:** `FactoMineR` (Multiple Correspondence Analysis)

- **Explicabilidad Visual:** `DALEX`, `DALEXtra`, `ggplot2`

- **Procesamiento de Datos:** `tidyverse` (`dplyr`, `purrr`, `stringr`)

## ⚠️ **Limitaciones del Estudio y Fronteras Analíticas**

No obstante, estos resultados prometedores deben enmarcarse dentro de ciertos límites estructurales y metodológicos descritos a continuación:

- **Solapamiento Conceptual de Predictores:** Aunque las pruebas de robustez demostraron con éxito la resiliencia del modelo —en particular, al soportar la exclusión intencionada de métricas educativas dominantes—, existe un solapamiento conceptual inherente entre ciertos predictores de la EPH y las dimensiones objetivo del método Alkire-Foster que exige una interpretación cautelosa de las atribuciones locales.

- **Sesgo Urbano de la Encuesta Base:** La Encuesta Permanente de Hogares (EPH) restringe la inferencia geográfica del modelo exclusivamente a los principales aglomerados metropolitanos del país. En consecuencia, las dinámicas de privación y vulnerabilidad rurales quedan fuera del bucle predictivo.

- **Naturaleza Conservadora del IPM:** Las definiciones estáticas y estructurales del Índice de Pobreza Multidimensional (IPM) pueden pasar por alto shocks económicos transitorios y de alta velocidad, un desafío que se ve acentuado por el *data drift* y el quiebre de tendencia post-censal de 2025.

- **Necesidad de Datos Dinámicos:** El pipeline actual demuestra la urgencia de orientar futuras líneas de investigación hacia la integración de registros administrativos dinámicos o conjuntos de datos longitudinales que complementen estos hallazgos estructurales y mitiguen la pérdida de vigencia temporal de los coeficientes.

> 💡 **Nota metodológica:** El reconocimiento de estas fronteras analíticas no disminuye la utilidad actual del marco propuesto; por el contrario, define un camino claro hacia la construcción de herramientas de focalización para políticas sociales aún más adaptativas, integrales y de alta precisión.

**Autor:** Luis Miguel Herrera Corrales\
**Contacto:** [lm.hcor\@gmail.com](mailto:lm.hcor@gmail.com) \| [GitHub](https://github.com/lm-hcor) \| [LinkedIn](https://www.linkedin.com/in/lmhcor)\
**Licencia:** Apache 2.0\
*Nota: Los datos provienen de la Encuesta Permanente de Hogares ([PH) del](https://www.linkedin.com/in/lmhcor)INDEC bajo la Ley de Secreto Estadístico 17.622.*
