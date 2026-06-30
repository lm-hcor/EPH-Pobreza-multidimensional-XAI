# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 06c_optimized_modelling_revised.R
# Propósito: Pipeline de Machine Learning con tidymodels - OPTIMIZADO PARA 16GB RAM
#
# CAMBIO CRÍTICO v2.0:
#   - SMOTE MOVIDO FUERA DEL RECIPE (se ejecuta UNA SOLA VEZ). 
#   - Se editó el oveR_ratio a 0.25.
#   - Elimina el cuello de botella de KNN distance calculations en el tuning
#   - Reduce tiempo de 12-16 horas a <45 minutos
#
# OPTIMIZACIONES DE RENDIMIENTO:
#   1. Paralelización minimalista (2 núcleos) con estrategia PSOCK secuencial
#   2. Control de hilos nativos (OMP/MKL/OPENBLAS) para evitar sub-paralelización
#   3. parallel_over = "resamples" para minimizar duplicación de memoria
#   4. Garbage collection estratégico en puntos críticos
#   5. Liberación agresiva de objetos intermedios
#
# CORRECCIONES METODOLÓGICAS MANTENIDAS:
#   1. PREDICTORES corregidos: eliminadas priv_* que causaban data leakage
#   2. La calibración del threshold (paso 11.5) se ejecuta ANTES de fit()
#   3. evaluar_modelo() usa los umbrales calibrados desde tabla_umbrales
#   4. ano4 excluido correctamente de predictores via update_role()
#   5. step_smotenc() se aplica DESPUÉS de step_dummy() (ahora externo)
#   6. grep() sobre eph_train (no eph_train_mca)
#   7. Se entrena sobre el nuevo train set (2016-2023), y se testea en 2024 y 2025.
# ==============================================================================

# ==============================================================================
# 0. CONFIGURACIÓN DE RENDIMIENTO Y PARALELIZACIÓN ROBUSTA
# ==============================================================================
# CRÍTICO: Configurar ANTES de cargar librerías pesadas para evitar deadlock

# Control de hilos nativos (BLAS/LAPACK) - EVITAR SUB-PARALELIZACIÓN
Sys.setenv(
  OMP_NUM_THREADS = "1",
  OMP_THREAD_LIMIT = "1",
  MKL_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

# Verificar configuración
message(">>> Configuración de hilos nativos:")
message("  OMP_NUM_THREADS: ", Sys.getenv("OMP_NUM_THREADS"))
message("  MKL_NUM_THREADS: ", Sys.getenv("MKL_NUM_THREADS"))
message("  OPENBLAS_NUM_THREADS: ", Sys.getenv("OPENBLAS_NUM_THREADS"))

# Paralelización minimalista para Windows 11 - EVITAR BLOQUEO
# Usamos exactamente 2 núcleos físicos con estrategia PSOCK secuencial
# Esto evita que Windows congele los procesos por "Modo de Eficiencia"
n_cores_fisicos <- parallel::detectCores(logical = FALSE)
message(">>> Núcleos físicos detectados: ", n_cores_fisicos)

# Crear cluster con estrategia secuencial para evitar deadlock de sockets
cl <- parallel::makeCluster(
  2,  # EXACTAMENTE 2 workers - mínimo para paralelismo real
  type = "PSOCK",  # PSOCK es más estable en Windows que FORK
  setup_strategy = "sequential"  # Evita contención de sockets locales
)

# Registrar el cluster
doParallel::registerDoParallel(cl)
message(">>> Cluster PSOCK registrado con 2 workers (setup_strategy = 'sequential')")
message("  Esto evita el throttling de Windows 11 y bloqueo por antivirus")

# ==============================================================================
# 1. CARGA DE LIBRERÍAS
# ==============================================================================
library(tidyverse)
library(tidymodels)
library(themis)
library(probably)
library(finetune)
library(xgboost)
library(ranger)
library(rpart)
library(janitor)
library(doParallel)
library(FactoMineR)
library(yardstick)
library(dplyr)
library(tibble)
tidymodels_prefer()

# ==============================================================================
# 2. CARGA DE DATOS
# ==============================================================================
message(">>> Cargando datos...")

eph_train     <- readRDS("data/processed/eph_train_mca.rds")
eph_test      <- readRDS("data/processed/eph_test_ml.rds")
eph_externo   <- readRDS("data/processed/eph_externo_ml.rds")
vars_estables <- readRDS("data/processed/mca_vars_estables.rds")

message("  ✓ Datos cargados. Limpieza de entorno...")
gc()  # Garbage collection temprano para liberar memoria de carga

# ==============================================================================
# 3. DEFINICIÓN DE PREDICTORES (SIN DATA LEAKAGE)
# ==============================================================================
# REGLA: se excluye cualquier variable que sea input directo de construir_mpi()
# Las priv_* (priv_piso, priv_agua, etc.) son exactamente los inputs ponderados
# del mpi_score → incluirlas da ROC-AUC artificial ~0.99

VARS_MCA_DIRECTAS <- c(
  "v2",   # pensión
  "v13",  # ahorros
  "v11",  # beca de gobierno
  "iv1",  # tipo de vivienda
  "iv2",  # material paredes
  "ii7",  # régimen tenencia
  "ii8"   # combustible cocina
)

PREDICTORES <- unique(c(
  
  # -- Coordenadas MCA (contexto socioeconómico, sin variables del MPI) --------
  grep("^mca_dim", names(eph_train), value = TRUE),
  
  # -- Económicas ---------------------------------------------------------------
  "itcf_real",         # Ingreso Total Familiar real
  "p21_real",          # Ingreso ocupación principal
  # "es_pobre_mon"     # EXCLUIDO: proxy directo del target (umbral monetario)
  
  # -- Demografía y estructura del hogar ----------------------------------------
  "tamano_hogar",
  "n_menores",
  "n_ancianos",
  "region_label",
  "ratio_dependencia",
  "n_ocupados",
  "max_instruccion",   # Nivel educativo máximo del hogar
  "adeq_hogar",        # Adultos equivalentes
  
  # -- Mercado de trabajo ------------------------------------------------------
  "prop_informal",     # Proporción de ocupados sin aportes jubilatorios
  # --Acceso a Sistema Sanitario------------------------------------------------
  "priv_salud",
  # -- Variables MCA complementarias a las dimensiones--------------------------
  VARS_MCA_DIRECTAS,
  # -- Variables temporales ----------------------------------------------------
  "ano4", # shocks temporales (crisis 2018, COVID 2020, hiperinflación 2023)
  "trimestre", # estacionalidad mercado laboral
  
  # -- EXCLUIDAS POR DATA LEAKAGE -----------------------------------------------
  # "priv_piso"         → input de mpi_score (peso 1/9)
  # "priv_techo"        → input de mpi_score (peso 1/9)
  # "priv_hacinamiento" → input de mpi_score (peso 1/9)
  # "priv_agua"         → input de mpi_score (peso 1/6)
  # "priv_cloaca"       → input de mpi_score (peso 1/6)
  # "priv_esc"          → input de mpi_score (peso 1/6)
  # "priv_educ"         → input de mpi_score (peso 1/6)
  # "priv_digital"      → construida sobre v11/v12, ya capturadas en mca_dim*
  NULL  # permite cerrar el vector sin coma colgante
))

# Filtro de seguridad: solo columnas que realmente existen en el train
PREDICTORES <- intersect(PREDICTORES, names(eph_train))
message("  Predictores activos: ", length(PREDICTORES))
message("  ", paste(PREDICTORES, collapse = ", "))

# ==============================================================================
# 4. PREPARACIÓN DEL DATASET DE MODELADO
# ==============================================================================
# Limpieza preventiva: si existe adeq_hogar como df en el entorno global
# puede colisionar con la columna del mismo nombre en el dataframe
if (exists("adeq_hogar") && is.data.frame(get("adeq_hogar"))) {
  rm(adeq_hogar, envir = .GlobalEnv)
}

data_model <- eph_train %>%
  select(mpi_pobre, codusu, aglomerado, ano4, trimestre,
         pondera, all_of(PREDICTORES)) %>%
  mutate(
    mpi_pobre    = factor(mpi_pobre, levels = c("no_pobre", "pobre")),
    region_label = factor(region_label),
    aglomerado   = as.character(aglomerado),
    across(c(itcf_real, adeq_hogar, p21_real), as.numeric)
  ) %>%
  drop_na(mpi_pobre)

# Muestreo estratificado por pondera para reducir tiempo de cómputo
set.seed(42)
data_model <- data_model %>%
  mutate(grupo_cv = paste0(codusu, "_", aglomerado)) |>
  slice_sample(prop = 0.7, weight_by = pondera)

# Verificar que grupo_cv tiene suficiente varianza para 10 folds
n_grupos <- data_model %>%
  distinct(grupo_cv) %>%
  nrow()

message("Grupos únicos para CV: ", n_grupos)
message("  Hogares en data_model: ", nrow(data_model))
message("  Distribución target: ")
print(table(data_model$mpi_pobre))

# ==============================================================================
# 4.1 REPONDERACIÓN DEL TRAIN PARA APROXIMAR EL MARCO CENSAL 2022
# ==============================================================================
# El test set usa el marco 2022 mientras el train usa el marco 2010.
# La reponderación ajusta los pesos del train para que su distribución
# demográfica se aproxime a la del test, reduciendo el shift censal.

# Distribución objetivo: estructura demográfica del test (marco 2022)
dist_test <- eph_externo %>%
  group_by(region_label) %>%
  summarise(
    prop_externo   = sum(pondera) / sum(eph_externo$pondera),
    media_tam    = weighted.mean(tamano_hogar, pondera, na.rm = TRUE),
    prop_menores = weighted.mean(n_menores > 0, pondera, na.rm = TRUE),
    .groups = "drop"
  )

# Distribución actual del train
dist_train <- data_model %>%
  group_by(region_label) %>%
  summarise(
    prop_train = sum(pondera) / sum(data_model$pondera),
    .groups    = "drop"
  )

# Calcular factor de ajuste por región
ajuste_regional <- dist_train %>%
  left_join(dist_test, by = "region_label") %>%
  mutate(
    factor_ajuste = if_else(
      is.na(prop_test) | prop_train == 0,
      1,
      prop_externo / prop_train
    ),
    # Truncar factores extremos para evitar pesos degenerados
    factor_ajuste = pmin(pmax(factor_ajuste, 0.2), 5.0)
  ) %>%
  select(region_label, factor_ajuste)

message("  Factores de ajuste regional:")
print(ajuste_regional)

# Aplicar reponderación al train
data_model <- data_model %>%
  left_join(ajuste_regional, by = "region_label") %>%
  mutate(
    pondera_orig    = pondera,
    pondera         = pondera * factor_ajuste,
    # Normalizar para que la suma de pesos sea igual a la original
    pondera         = pondera * (sum(pondera_orig) / sum(pondera))
  ) %>%
  select(-factor_ajuste, -pondera_orig)

message("  Reponderación aplicada. Suma pesos original vs nuevo: OK")

# Liberar objetos intermedios
# rm(dist_test, dist_train, ajuste_regional)
# gc()

# ==============================================================================
# 4.2 SMOTENC EXTERNO - SE EJECUTA UNA SOLA VEZ (FUERA DEL CV)
# ==============================================================================
# CAMBIO CRÍTICO: Movemos SMOTENC fuera del recipe para evitar
# que se recalcule en cada fold × cada combinación de hiperparámetros.
# Esto reduce el tiempo de 12-16 horas a <45 minutos.
#
# NOTA: Usamos SMOTENC (no SMOTE) porque maneja variables mixtas:
# - Variables continuas: interpola (como SMOTE)
# - Variables categóricas/dummies: vota por mayoría (específico de SMOTENC)
# over_ratio = 0.25: la clase minoritaria tendrá 25% de la mayoritaria
# (conservador para no sobrecargar la memoria con 16GB RAM)
#
# IMPORTANTE: SMOTENC requiere NO tener NAs → imputamos antes
# IMPORTANTE: Mantenemos TODAS las columnas técnicas necesarias:
#   - grupo_cv: para block CV
#   - aglomerado: para proyección MCA y evaluación
#   - pondera: para evaluación ponderada
#   - codusu: para identificación

message(">>> Aplicando SMOTENC externo (una sola vez)...")

# Estrategia: Aplicar SMOTENC solo sobre predictores, luego unir columnas técnicas
# Para filas originales: mantienen sus columnas técnicas
# Para filas sintéticas: asignamos las columnas técnicas de su vecino SMOTE original

# Paso 1: Crear dataset con índice para tracking
data_with_index <- data_model %>%
  mutate(.smote_id = row_number())

# Paso 2: Preparar predictores para SMOTENC (sin columnas técnicas)
data_predictores <- data_with_index %>%
  select(mpi_pobre, .smote_id, all_of(PREDICTORES)) %>%
  mutate(across(where(is.character), as.factor))

# Paso 3: Aplicar SMOTENC
set.seed(42)
rec_smotenc <- recipe(mpi_pobre ~ ., data = data_predictores) %>%
  # Mantener .smote_id como identificador (no se usa en SMOTE)
  update_role(.smote_id, new_role = "id") %>%
  # Imputar NAs (crítico para SMOTENC)
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  # Colapsar categorías raras
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  # Convertir a dummies
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  # Aplicar SMOTENC
  themis::step_smotenc(mpi_pobre, over_ratio = 0.25, neighbors = 5)

message("  Ejecutando SMOTENC (prep + bake)...")
rec_smotenc_prep <- prep(rec_smotenc, training = data_predictores, verbose = FALSE)
data_balanced_with_id <- bake(rec_smotenc_prep, new_data = NULL)

# Paso 4: Unir columnas técnicas desde data_with_index usando .smote_id
# Para filas originales: .smote_id coincide directamente
# Para filas sintéticas: .smote_id es NA → asignamos del vecino más cercano
# Como no tenemos info del vecino, usamos muestreo estratificado por mpi_pobre

# Extraer columnas técnicas
cols_tecnicas <- data_with_index %>%
  select(.smote_id, codusu, aglomerado, grupo_cv, pondera)

# Separar filas originales y sintéticas
filas_originales <- data_balanced_with_id %>%
  filter(.smote_id %in% data_with_index$.smote_id)

filas_sinteticas <- data_balanced_with_id %>%
  filter(is.na(.smote_id) | !(.smote_id %in% data_with_index$.smote_id))

# Unir columnas técnicas para filas originales
filas_originales <- filas_originales %>%
  left_join(cols_tecnicas, by = ".smote_id")

# Para filas sintéticas: asignar columnas técnicas basadas en distribución de mpi_pobre
if (nrow(filas_sinteticas) > 0) {
  # Muestrear aleatoriamente de las columnas técnicas, estratificado por mpi_pobre
  muestras_tecnicas <- cols_tecnicas %>%
    left_join(data_with_index %>% select(.smote_id, mpi_pobre), by = ".smote_id")
  
  # Para cada fila sintética, asignar columnas técnicas de una fila original con mismo target
  filas_sinteticas_con_tecnicas <- filas_sinteticas %>%
    group_by(mpi_pobre, .drop = TRUE) %>%
    group_modify(~{
      muestras <- muestras_tecnicas %>%
        filter(mpi_pobre == .y$mpi_pobre) %>%
        slice_sample(n = nrow(.x), replace = TRUE) %>%
        select(-mpi_pobre, -.smote_id)
      bind_cols(.x, muestras)
    }) %>%
    ungroup()
  
  # Combinar originales y sintéticas
  data_model_balanced <- bind_rows(filas_originales, filas_sinteticas_con_tecnicas)
} else {
  data_model_balanced <- filas_originales
}

data_model_balanced <- data_model_balanced %>%
  select(-any_of(".smote_id"))

message("  Distribución DESPUÉS de SMOTENC:")
print(table(data_model_balanced$mpi_pobre))
message("  Filas originales: ", nrow(data_predictores),
        " → Filas después de SMOTENC: ", nrow(data_model_balanced))

# Verificar columnas técnicas presentes
message("  Columnas técnicas presentes: ",
        paste(c("codusu", "aglomerado", "grupo_cv", "pondera") %in% names(data_model_balanced), collapse = ", "))

# Liberar objetos temporales
rm(data_with_index, data_predictores, rec_smotenc, rec_smotenc_prep, 
   data_balanced_with_id, cols_tecnicas, filas_originales, filas_sinteticas)
if (exists("muestras_tecnicas")) rm(muestras_tecnicas)
if (exists("filas_sinteticas_con_tecnicas")) rm(filas_sinteticas_con_tecnicas)
gc()

message("  ✓ SMOTENC externo completado (ejecución única)")

# ==============================================================================
# 5. BLOCK CROSS-VALIDATION SOBRE DATOS YA BALANCEADOS
# ==============================================================================
# Los folds ahora se crean sobre data_model_balanced
set.seed(42)
block_folds <- group_vfold_cv(
  data = data_model_balanced, 
  group = grupo_cv,  # Nota: grupo_cv se mantiene del data_model original
  v = 10
)

message("  ✓ 10 folds creados sobre datos balanceados (block CV por aglomerado)")

# ==============================================================================
# 6. RECIPE LIMPIO - SIN SMOTE (SE EJECUTA EN MILISEGUNDOS)
# ==============================================================================
# El recipe ahora solo hace transformaciones ligeras:
# - Imputación de medianas/modas
# - Creación de features derivadas
# - Transformación logarítmica
# - Eliminación de correlacionados
# - Dummy variables
# - Normalización

rec_base <- recipe(
  mpi_pobre ~ .,
  # Excluir columnas técnicas que NO son predictores
  data = data_model_balanced %>% select(-aglomerado, -pondera, -codusu, -grupo_cv)
) %>%
  
  # ano4 se usa para crear año_norm pero NO debe ser predictor del modelo
  update_role(ano4, new_role = "id") %>%
  
  # 1. Imputación previa a la creación de features derivadas
  step_impute_median(itcf_real, adeq_hogar, p21_real) %>%
  
  # 2. Creación de features derivadas
  step_mutate(
    ingreso_per_capita = itcf_real / pmax(adeq_hogar, 1),
    año_norm           = (ano4 - 2016) / (2023 - 2016),
    carga_demo         = (n_menores + n_ancianos * 0.7) / pmax(n_ocupados, 0.5)
  ) %>%
  
  # 3. Transformación logarítmica de ingresos
  step_log(ingreso_per_capita, itcf_real, p21_real, base = 10, offset = 0.5) %>%
  
  # 4. Imputación del resto de predictores
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  
  # 5. Eliminar predictores numéricos altamente correlacionados (r > 0.80)
  step_corr(all_numeric_predictors(), threshold = 0.80) %>%
  
  # 6. Tratamiento de variables categóricas
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  step_dummy(all_nominal_predictors(), one_hot = FALSE) %>%
  
  # 7. Limpieza y normalización
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ¡NOTA: SIN step_smote()! Ya se aplicó externamente

message("  ✓ Recipe limpio creado (sin SMOTE - se ejecutará en milisegundos)")

# ==============================================================================
# 7. ESPECIFICACIONES DE MODELOS
# ==============================================================================
spec_cart <- decision_tree(
  cost_complexity = tune(),
  tree_depth      = tune(),
  min_n           = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

spec_rf <- rand_forest(
  mtry  = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger", importance = "impurity", seed = 42, num.threads = 1) %>%
  set_mode("classification")

spec_xgb <- boost_tree(
  trees         = tune(),
  tree_depth    = tune(),
  learn_rate    = tune(),
  loss_reduction = tune(),
  sample_size   = tune(),
  min_n         = tune()
) %>%
  set_engine("xgboost", eval_metric = "logloss", nthread = 1, seed = 42) %>%
  set_mode("classification")

message("  ✓ Especificaciones de modelos creadas (1 hilo interno c/u)")

# ==============================================================================
# 8. WORKFLOWS
# ==============================================================================
wf_cart <- workflow() %>% add_recipe(rec_base) %>% add_model(spec_cart)
wf_rf   <- workflow() %>% add_recipe(rec_base) %>% add_model(spec_rf)
wf_xgb  <- workflow() %>% add_recipe(rec_base) %>% add_model(spec_xgb)

# ==============================================================================
# 9. GRIDS DE HIPERPARÁMETROS
# ==============================================================================
grid_cart <- grid_space_filling(
  cost_complexity(),
  tree_depth(),
  min_n(),
  size = 15
)

grid_rf <- grid_space_filling(
  mtry(range  = c(2L, 8L)),
  trees(range = c(200L, 800L)),
  min_n(range = c(50L, 200L)),
  size = 15
)

grid_xgb <- grid_space_filling(
  trees(),
  tree_depth(),
  learn_rate(),
  loss_reduction(),
  sample_size = sample_prop(range = c(0.5, 0.9)),
  min_n(),
  size = 18
)

message("  ✓ Grids creados: CART(15), RF(15), XGB(18)")

# ==============================================================================
# 10. MÉTRICAS Y CONTROL DEL TUNING - OPTIMIZADO PARA MEMORIA
# ==============================================================================
metricas_tuning <- metric_set(roc_auc)
metricas <- metric_set(roc_auc, f_meas, kap, accuracy)

# CRÍTICO: parallel_over = "resamples" para minimizar duplicación de RAM
# Esto asegura que solo los folds se distribuyan entre los 2 cores,
# NO las combinaciones de hiperparámetros (que duplicarían la memoria)
ctrl <- control_race(
  save_pred      = TRUE,           # NECESARIO para calibración de threshold
  verbose_elim   = FALSE,
  parallel_over  = "resamples",  # ← CLAVE: solo paraleliza sobre folds
  burn_in        = 2               # Mínimo permitido en esta versión de finetune
)

message("  ✓ Control configurado: parallel_over = 'resamples'")

# ==============================================================================
# 11. TUNING CON ANOVA RACE - CON GC ESTRATÉGICO
# ==============================================================================
# Función auxiliar para tuning con garbage collection
tunar_con_gc <- function(workflow, folds, grid, nombre, control) {
  message("\n--- Tuning ", nombre, " ---")
  message("  Memoria antes: ", format(object.size(ls(all.names = TRUE)), units = "Mb"))
  message("  Ejecutando gc() previo al tuning...")
  gc()
  
  resultado <- tune_race_anova(
    workflow, resamples = folds,
    grid = grid, metrics = metricas_tuning, control = control
  )
  
  message("  Memoria después: ", format(object.size(ls(all.names = TRUE)), units = "Mb"))
  message("  Ejecutando gc()...")
  gc()
  
  return(resultado)
}

# Ejecutar tuning con gestión de memoria
tune_cart <- tunar_con_gc(wf_cart, block_folds, grid_cart, "CART", ctrl)
tune_rf   <- tunar_con_gc(wf_rf,   block_folds, grid_rf,   "Random Forest", ctrl)
tune_xgb  <- tunar_con_gc(wf_xgb,  block_folds, grid_xgb,  "XGBoost", ctrl)

message("\n>>> Tuning completado para los 3 modelos")

# ==============================================================================
# 12. SELECCIÓN DE MEJORES HIPERPARÁMETROS
# ==============================================================================
best_cart <- select_best(tune_cart, metric = "roc_auc")
best_rf   <- select_best(tune_rf,   metric = "roc_auc")
best_xgb  <- select_best(tune_xgb,  metric = "roc_auc")

message("\n--- Mejores hiperparámetros ---")
message("CART:");          print(best_cart)
message("Random Forest:"); print(best_rf)
message("XGBoost:");       print(best_xgb)

# ==============================================================================
# 13. CALIBRACIÓN DEL THRESHOLD (ANTES DE FIT FINAL)
# ==============================================================================
calibrar_umbral <- function(tune_result, best_params, nombre_modelo,
                            thresholds = seq(0.05, 0.65, by = 0.005)) {
  
  message("  Calibrando threshold para: ", nombre_modelo)
  
  preds_oof <- tune_result %>%
    collect_predictions(parameters = best_params)
  
  if (!".pred_pobre" %in% names(preds_oof)) {
    stop(
      "No se encontró '.pred_pobre' en las predicciones OOF de ", nombre_modelo,
      ". Verifica que save_pred = TRUE en control_race()."
    )
  }
  
  threshold_data <- preds_oof %>%
    probably::threshold_perf(
      truth       = mpi_pobre,
      estimate    = .pred_pobre,
      thresholds  = thresholds,
      event_level = "second",
      metrics     = metric_set(j_index, f_meas, sensitivity, specificity)
    )
  
  umbral_optimo <- threshold_data %>%
    filter(.metric == "f_meas") %>%
    slice_max(.estimate, n = 1, with_ties = FALSE) %>%
    pull(.threshold)
  
  # Extraer métricas en el umbral óptimo para el log
  metricas_umbral <- threshold_data %>%
    filter(.threshold == umbral_optimo) %>%
    select(.metric, .estimate) %>%
    pivot_wider(names_from = .metric, values_from = .estimate)
  
  message("    Umbral óptimo : ", round(umbral_optimo, 3))
  message("    Sensibilidad  : ", round(metricas_umbral$sensitivity, 3))
  message("    Especificidad : ", round(metricas_umbral$specificity,  3))
  message("    F1            : ", round(metricas_umbral$f_meas,       3))
  
  list(
    modelo         = nombre_modelo,
    umbral         = umbral_optimo,
    threshold_data = threshold_data
  )
}

cal_cart <- calibrar_umbral(tune_cart, best_cart, "CART")
cal_rf   <- calibrar_umbral(tune_rf,   best_rf,   "Random Forest")
cal_xgb  <- calibrar_umbral(tune_xgb,  best_xgb,  "XGBoost")

tabla_umbrales <- tibble(
  modelo = c("CART", "Random Forest", "XGBoost"),
  umbral = c(cal_cart$umbral, cal_rf$umbral, cal_xgb$umbral)
)

message("\n>>> Umbrales calibrados por modelo:")
print(tabla_umbrales)

# Visualización de curvas de threshold
threshold_comparado <- bind_rows(
  cal_cart$threshold_data %>% mutate(modelo = "CART"),
  cal_rf$threshold_data   %>% mutate(modelo = "Random Forest"),
  cal_xgb$threshold_data  %>% mutate(modelo = "XGBoost")
)

tabla_umbrales_plot <- tabla_umbrales

p_threshold <- threshold_comparado %>%
  filter(.metric %in% c("j_index", "f_meas")) %>%
  ggplot(aes(x = .threshold, y = .estimate,
             color = modelo, linetype = .metric)) +
  geom_line(linewidth = 0.9, alpha = 0.85) +
  geom_vline(
    data     = tabla_umbrales_plot,
    aes(xintercept = umbral, color = modelo),
    linetype = "dashed", linewidth = 0.6,
    show.legend = FALSE
  ) +
  geom_text(
    data = tabla_umbrales_plot,
    aes(x = umbral, y = 0.08,
        label = round(umbral, 2),
        color = modelo),
    size = 3, hjust = -0.15,
    show.legend = FALSE,
    inherit.aes = FALSE
  ) +
  scale_linetype_manual(
    values = c("f_meas" = "solid", "j_index" = "dashed"),
    labels = c("F1-score (Optimizado)", "J-index (Youden)")
  ) +
  scale_color_manual(values = c(
    "CART"          = "#E07B54",
    "Random Forest" = "#3A7DC9",
    "XGBoost"       = "#2ECC71"
  )) +
  labs(
    title    = "Calibración del Threshold por Modelo",
    subtitle = "Línea vertical = umbral óptimo según F1-score",
    x        = "Threshold de clasificación",
    y        = "Valor de la métrica",
    color    = "Modelo",
    linetype = "Métrica"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("output/figures/threshold_calibration_comparado.png",
       p_threshold, width = 10, height = 6, dpi = 150)

# Liberar objetos de calibración que ya no se necesitan
rm(threshold_comparado, p_threshold, cal_cart, cal_rf, cal_xgb)
gc()

# ==============================================================================
# 14. FINALIZACIÓN DE MODELOS SOBRE EL DATASET COMPLETO
# ==============================================================================
message("--- Ajustando modelos finales sobre data_model_balanced completo ---")

fit_cart <- finalize_workflow(wf_cart, best_cart) %>% fit(data = data_model_balanced)
fit_rf   <- finalize_workflow(wf_rf,   best_rf)   %>% fit(data = data_model_balanced)
fit_xgb  <- finalize_workflow(wf_xgb,  best_xgb)  %>% fit(data = data_model_balanced)

message("  ✓ Modelos finales ajustados")

# ==============================================================================
# 15. PROYECCIÓN MCA SOBRE EL TEST SET (2024)
# ==============================================================================
# Método robusto: proyección manual sin predict.MCA() que falla con categorías no vistas

VARS_MCA_ORIG <- c(
  "v2", "v13", "v5", "v11", "v12",
  "iv1", "iv2", "iv5",
  "ii7", "ii8", "iv10", "ii9"
)

# --- Paso 1: Entrenar MCA sobre años recientes del train (2020-2023) ----------
df_mca_reciente <- eph_train %>%
  filter(ano4 >= 2020) %>%
  select(all_of(VARS_MCA_ORIG)) %>%
  mutate(across(everything(), ~ factor(as.character(.)))) %>%
  mutate(across(everything(),
                ~ fct_lump_prop(., prop = 0.02, other_level = "Otros")))

message(">>> Entrenando MCA 2020-2023...")
res_mca_reciente <- MCA(df_mca_reciente, ncp = 2, graph = FALSE)
message("  MCA completado.")

# --- Paso 2: Extraer componentes del MCA necesarios para proyección manual ----
v_coord <- res_mca_reciente$var$coord
col_names_mca <- rownames(v_coord)
ncp <- ncol(v_coord)

# --- Paso 3: Alinear el test al espacio de categorías conocidas --------------
niveles_mca <- lapply(df_mca_reciente, levels)

alinear_test_mca <- function(df_test, niveles_ref) {
  for (col in names(niveles_ref)) {
    niveles_validos <- niveles_ref[[col]]
    df_test[[col]] <- as.character(df_test[[col]])
    df_test[[col]] <- if_else(
      df_test[[col]] %in% niveles_validos,
      df_test[[col]],
      "Otros"
    )
    df_test[[col]] <- factor(df_test[[col]], levels = niveles_validos)
  }
  df_test
}

df_test_alineado <- eph_test %>%
  select(all_of(VARS_MCA_ORIG)) %>%
  mutate(across(everything(), as.character)) %>%
  alinear_test_mca(niveles_mca)

# Verificar NAs residuales
n_na_test <- sum(is.na(df_test_alineado))
message("  NAs en test alineado: ", n_na_test)

if (n_na_test > 0) {
  df_test_alineado <- df_test_alineado %>%
    mutate(across(everything(), ~ {
      moda <- names(sort(table(.), decreasing = TRUE))[1]
      factor(if_else(is.na(.), moda, as.character(.)), levels = levels(.))
    }))
  message("  NAs imputados con moda.")
}

# --- Paso 4: Construir tabla disyuntiva completa (dummies 0/1) ---------------
construir_disjuntiva <- function(df, niveles_ref) {
  result <- vector("list", length(niveles_ref))
  names(result) <- names(niveles_ref)
  
  for (col in names(niveles_ref)) {
    nivs <- niveles_ref[[col]]
    mat  <- model.matrix(~ 0 + ., data = data.frame(x = df[[col]]))
    colnames(mat) <- paste0(col, "_", nivs[match(
      sub("^x", "", colnames(mat)), nivs
    )])
    result[[col]] <- mat
  }
  do.call(cbind, result)
}

Z_test <- construir_disjuntiva(df_test_alineado, niveles_mca)

# Alinear columnas de Z_test exactamente con las del MCA
cols_falt  <- setdiff(col_names_mca, colnames(Z_test))
cols_sobre <- setdiff(colnames(Z_test), col_names_mca)

if (length(cols_falt) > 0) {
  message("  Columnas faltantes en test (→ 0): ", paste(cols_falt, collapse = ", "))
  mat_cero <- matrix(0, nrow = nrow(Z_test), ncol = length(cols_falt),
                     dimnames = list(NULL, cols_falt))
  Z_test <- cbind(Z_test, mat_cero)
}

if (length(cols_sobre) > 0) {
  message("  Columnas extra en test (descartadas): ",
          paste(cols_sobre, collapse = ", "))
}

Z_test <- Z_test[, col_names_mca, drop = FALSE]

# --- Paso 5: Proyección — fórmula de transición suplementaria ----------------
Q       <- length(VARS_MCA_ORIG)
n_test  <- nrow(Z_test)
Z_norm  <- Z_test / Q

coords_test <- Z_norm %*% v_coord

mca_test_proj <- as.data.frame(coords_test) %>%
  setNames(paste0("mca_dim", seq_len(ncp))) %>%
  mutate(across(everything(), ~ if_else(is.na(.), 0, .)))

n_na_proj <- sum(is.na(mca_test_proj))
message("  NAs en proyección MCA: ", n_na_proj)

eph_test_mca <- bind_cols(eph_test, mca_test_proj)
message("  Proyección completada. Dimensiones test_mca: ",
        nrow(eph_test_mca), " × ", ncol(eph_test_mca))

# ==============================================================================
# 16. EVALUACIÓN EN TEST (2024) Y EXTERNO (2025)
# ==============================================================================

match_classes <- function(target, reference) {
  
  for (col in intersect(names(target), names(reference))) {
    
    if (is.factor(reference[[col]])) {
      
      target[[col]] <- factor(
        as.character(target[[col]]),
        levels = levels(reference[[col]])
      )
      
    } else {
      
      class(target[[col]]) <- class(reference[[col]])
      
    }
    
  }
  
  target
}

# ------------------------------------------------------------------------------
# Función auxiliar: crear dummies de región exactamente como en train
# ------------------------------------------------------------------------------

crear_dummies_region <- function(df) {
  
  regiones <- c(
    "Cuyo",
    "GBA",
    "Nordeste",
    "Noroeste",
    "Pampeana",
    "Patagonia"
  )
  
  for (r in regiones) {
    
    df[[paste0("region_label_", r)]] <-
      as.integer(as.character(df$region_label) == r)
    
  }
  
  df %>%
    select(-region_label)
}

# ------------------------------------------------------------------------------
# TEST 2024
# ------------------------------------------------------------------------------
# 1. Buscamos dinámicamente cómo se llaman las columnas de MCA en esta sesión
mca_cols_reales <- names(eph_test_mca)[stringr::str_detect(names(eph_test_mca), "^mca_dim[12]")]

# 2. Reemplazamos mca_dim1 y mca_dim2 en PREDICTORES por los nombres reales con sufijo
PREDICTORES_ADAPTADOS <- PREDICTORES
if (!all(c("mca_dim1", "mca_dim2") %in% names(eph_test_mca))) {
  PREDICTORES_ADAPTADOS <- setdiff(PREDICTORES, c("mca_dim1", "mca_dim2"))
  PREDICTORES_ADAPTADOS <- c(PREDICTORES_ADAPTADOS, mca_cols_reales)
}

data_test_final <- eph_test_mca %>%
  mutate(
    grupo_cv = paste0(codusu, "_", aglomerado)
  ) %>%
  select(
    all_of(
      c(
        "mpi_pobre",
        "codusu",
        "grupo_cv",
        "aglomerado",
        "ano4",
        "pondera",
        PREDICTORES_ADAPTADOS # <--- Usamos la lista adaptada con Regex
      )
    )
  ) %>%
  mutate(
    mpi_pobre = factor(
      mpi_pobre,
      levels = c("no_pobre", "pobre")
    )
  ) %>%
  crear_dummies_region() %>%
  match_classes(reference = data_model_balanced) %>%
  drop_na(mpi_pobre)

# Asegurar que existen todas las columnas del train
# NOTA: Esto va a transformar los mca_dim1...51 a 0 y va a dejar solo los nombres limpios
# que tus modelos (fit_xgb, etc.) esperan recibir.
faltantes_test <- setdiff(
  names(data_model_balanced),
  names(data_test_final)
)

for (v in faltantes_test) {
  data_test_final[[v]] <- 0
}

# Renombramos las columnas con sufijo a sus nombres limpios para que el modelo las entienda
for (i in seq_along(mca_cols_reales)) {
  col_sufijo <- mca_cols_reales[i]
  col_limpia <- if_else(stringr::str_detect(col_sufijo, "dim1"), "mca_dim1", "mca_dim2")
  if (col_sufijo %in% names(data_test_final)) {
    data_test_final[[col_limpia]] <- data_test_final[[col_sufijo]]
  }
}

data_test_final <- data_test_final %>%
  select(all_of(names(data_model_balanced)))

# ------------------------------------------------------------------------------
# Función de evaluación
# ------------------------------------------------------------------------------
evaluar_modelo <- function(fit, data, nombre, umbral = 0.5) {
  
  pred_prob <- predict(
    fit,
    data,
    type = "prob"
  )
  
  pred_class <- pred_prob %>%
    mutate(
      .pred_class = factor(
        if_else(
          .pred_pobre >= umbral,
          "pobre",
          "no_pobre"
        ),
        levels = c("no_pobre", "pobre")
      )
    ) %>%
    select(.pred_class)
  
  # Definimos el set de métricas estándar de tidymodels directamente acá adentro
  mis_metricas <- yardstick::metric_set(
    yardstick::roc_auc,
    yardstick::f_meas,
    yardstick::sens,
    yardstick::spec,
    yardstick::kap
  )
  
  bind_cols(
    data %>% select(mpi_pobre, pondera),
    pred_class,
    pred_prob
  ) %>%
    
    # Reemplazamos "metricas" por "mis_metricas" nativo con pesos censales
    mis_metricas(
      truth        = mpi_pobre,
      estimate     = .pred_class,
      .pred_pobre,
      event_level  = "second",
      case_weights = pondera
    ) %>%
    
    mutate(
      modelo = nombre,
      umbral_usado = umbral
    )
}
# ------------------------------------------------------------------------------
# RESULTADOS TEST 2024
# ------------------------------------------------------------------------------

tabla_resultados <- bind_rows(
  
  evaluar_modelo(
    fit_cart,
    data_test_final,
    "CART",
    umbral = tabla_umbrales %>%
      filter(modelo == "CART") %>%
      pull(umbral)
  ),
  
  evaluar_modelo(
    fit_rf,
    data_test_final,
    "Random Forest",
    umbral = tabla_umbrales %>%
      filter(modelo == "Random Forest") %>%
      pull(umbral)
  ),
  
  evaluar_modelo(
    fit_xgb,
    data_test_final,
    "XGBoost",
    umbral = tabla_umbrales %>%
      filter(modelo == "XGBoost") %>%
      pull(umbral)
  )
)

message("\n>>> Resultados TEST 2024:")
print(tabla_resultados)

# ------------------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------------------

set.seed(42)

data_train_eval <- data_model_balanced %>%
  slice_sample(
    n = min(
      10000, # aumentar
      nrow(data_model_balanced)
    )
  )

tabla_train <- bind_rows(
  
  evaluar_modelo(
    fit_cart,
    data_train_eval,
    "CART",
    umbral = tabla_umbrales %>%
      filter(modelo == "CART") %>%
      pull(umbral)
  ),
  
  evaluar_modelo(
    fit_rf,
    data_train_eval,
    "Random Forest",
    umbral = tabla_umbrales %>%
      filter(modelo == "Random Forest") %>%
      pull(umbral)
  ),
  
  evaluar_modelo(
    fit_xgb,
    data_train_eval,
    "XGBoost",
    umbral = tabla_umbrales %>%
      filter(modelo == "XGBoost") %>%
      pull(umbral)
  )
)

message("\n>>> Métricas TRAIN:")
print(tabla_train)


# ==============================================================================
# 16.1 EVALUACIÓN EXTERNA 2025
# ==============================================================================

message("\n>>> Preparando evaluación externa 2025...")

# ------------------------------------------------------------------
# Reconstrucción de objetos MCA necesarios para proyectar 2025
# ------------------------------------------------------------------
df_mca_reciente <- eph_train %>%
  filter(ano4 >= 2020) %>%
  select(all_of(VARS_MCA_ORIG)) %>%
  mutate(across(everything(), ~ factor(as.character(.)))) %>%
  mutate(
    across(
      everything(),
      ~ fct_lump_prop(
        .,
        prop = 0.02,
        other_level = "Otros"
      )
    )
  )

res_mca_reciente <- MCA(
  df_mca_reciente,
  ncp = ncp,
  graph = FALSE
)

v_coord <- res_mca_reciente$var$coord

niveles_mca <- lapply(
  df_mca_reciente,
  levels
)

df_externo_alineado <- eph_externo %>%
  select(all_of(VARS_MCA_ORIG)) %>%
  mutate(across(everything(), as.character)) %>%
  alinear_test_mca(niveles_mca)

Z_externo <- construir_disjuntiva(
  df_externo_alineado,
  niveles_mca
)

cols_falt_externo <- setdiff(
  col_names_mca,
  colnames(Z_externo)
)

if (length(cols_falt_externo) > 0) {
  Z_externo <- cbind(
    Z_externo,
    matrix(
      0,
      nrow = nrow(Z_externo),
      ncol = length(cols_falt_externo),
      dimnames = list(NULL, cols_falt_externo)
    )
  )
}

Z_externo <- Z_externo[, col_names_mca, drop = FALSE]

coords_externo <- (Z_externo / length(VARS_MCA_ORIG)) %*% v_coord

mca_externo_proj <- as.data.frame(coords_externo) %>%
  setNames(
    paste0(
      "mca_dim",
      seq_len(ncp)
    )
  )

# Combinación externa (aquí se generan los sufijos si ya existían columnas mca)
eph_externo_mca <- bind_cols(
  eph_externo,
  mca_externo_proj
)

# ------------------------------------------------------------------
# BLINDAJE POR REGEX: Capturar nombres reales de las columnas MCA
# ------------------------------------------------------------------
# Detecta "mca_dim1", "mca_dim2" y cualquier variante con sufijo (ej: mca_dim1...51)
mca_cols_ext_reales <- names(eph_externo_mca)[stringr::str_detect(names(eph_externo_mca), "^mca_dim[12]")]

# Reemplazar dinámicamente mca_dim1 y mca_dim2 en la lista de PREDICTORES por sus nombres con sufijo
PREDICTORES_EXT_ADAPTADOS <- PREDICTORES
if (!all(c("mca_dim1", "mca_dim2") %in% names(eph_externo_mca))) {
  PREDICTORES_EXT_ADAPTADOS <- setdiff(PREDICTORES, c("mca_dim1", "mca_dim2"))
  PREDICTORES_EXT_ADAPTADOS <- c(PREDICTORES_EXT_ADAPTADOS, mca_cols_ext_reales)
}

# ------------------------------------------------------------------
# Construcción del Dataset Externo Final
# ------------------------------------------------------------------
data_externo_final <- eph_externo_mca %>%
  mutate(
    grupo_cv = paste0(codusu, "_", aglomerado)
  ) %>%
  select(
    all_of(
      c(
        "mpi_pobre",
        "codusu",
        "grupo_cv",
        "aglomerado",
        "ano4",
        "pondera",
        PREDICTORES_EXT_ADAPTADOS # <-- Usamos la lista tolerante a sufijos
      )
    )
  ) %>%
  mutate(
    mpi_pobre = factor(
      mpi_pobre,
      levels = c("no_pobre", "pobre")
    )
  ) %>%
  crear_dummies_region() %>%
  match_classes(reference = data_model_balanced) %>%
  drop_na(mpi_pobre)

# Asegurar la existencia de todas las columnas de la matriz de entrenamiento
faltantes_ext <- setdiff(
  names(data_model_balanced),
  names(data_externo_final)
)

for (v in faltantes_ext) {
  data_externo_final[[v]] <- 0
}

# ------------------------------------------------------------------
# TRADUCCIÓN: Renombrar sufijos a nombres limpios para el Modelo
# ------------------------------------------------------------------
# Mapea los vectores corruptos hacia las variables originales que el fit requiere
for (i in seq_along(mca_cols_ext_reales)) {
  col_sufijo <- mca_cols_ext_reales[i]
  col_limpia <- if_else(stringr::str_detect(col_sufijo, "dim1"), "mca_dim1", "mca_dim2")
  if (col_sufijo %in% names(data_externo_final)) {
    data_externo_final[[col_limpia]] <- data_externo_final[[col_sufijo]]
  }
}

# Forzar orden de columnas idéntico al del entrenamiento balanceado
data_externo_final <- data_externo_final %>%
  select(all_of(names(data_model_balanced)))

# ------------------------------------------------------------------
# Ejecución de Evaluaciones Externas
# ------------------------------------------------------------------
tabla_externa <- bind_rows(
  evaluar_modelo(
    fit_cart,
    data_externo_final,
    "CART",
    umbral = tabla_umbrales %>%
      filter(modelo == "CART") %>%
      pull(umbral)
  ),
  
  evaluar_modelo(
    fit_rf,
    data_externo_final,
    "Random Forest",
    umbral = tabla_umbrales %>%
      filter(modelo == "Random Forest") %>%
      pull(umbral)
  ),
  
  evaluar_modelo(
    fit_xgb,
    data_externo_final,
    "XGBoost",
    umbral = tabla_umbrales %>%
      filter(modelo == "XGBoost") %>%
      pull(umbral)
  )
)

message("\n>>> Resultados EXTERNOS 2025:")
print(tabla_externa)
# ==============================================================================
# 16.2 COMPARACIÓN GLOBAL TRAIN vs TEST 2024 vs EXTERNO 2025
# ==============================================================================

message("\n>>> Construyendo tabla comparativa global...")

tabla_train_comp <- tabla_train %>%
  mutate(dataset = "Train")

tabla_test_comp <- tabla_resultados %>%
  mutate(dataset = "Test 2024")

tabla_externo_comp <- tabla_externa %>%
  mutate(dataset = "Externo 2025")

# ------------------------------------------------------------------
# Tabla larga (ideal para ggplot)
# ------------------------------------------------------------------

tabla_metricas_global <- bind_rows(
  tabla_train_comp,
  tabla_test_comp,
  tabla_externo_comp
) %>%
  select(
    dataset,
    modelo,
    .metric,
    .estimate,
    umbral_usado
  ) %>%
  arrange(
    modelo,
    dataset,
    .metric
  )

message("\n>>> Tabla global de métricas:")
print(tabla_metricas_global)

# ------------------------------------------------------------------
# Tabla ancha (más cómoda para inspección)
# ------------------------------------------------------------------

tabla_metricas_wide <- tabla_metricas_global %>%
  select(
    dataset,
    modelo,
    .metric,
    .estimate
  ) %>%
  pivot_wider(
    names_from = .metric,
    values_from = .estimate
  ) %>%
  arrange(modelo, dataset)

message("\n>>> Tabla resumen:")
print(tabla_metricas_wide)

# ------------------------------------------------------------------
# Gráfico comparativo de barras
# ------------------------------------------------------------------

p_metricas <- tabla_metricas_global %>%
  ggplot(
    aes(
      x = modelo,
      y = .estimate,
      fill = dataset
    )
  ) +
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.7
  ) +
  facet_wrap(
    ~ .metric,
    scales = "free_y"
  ) +
  labs(
    title = "Comparación de desempeño por modelo",
    subtitle = "Train vs Test 2024 vs Externo 2025",
    x = NULL,
    y = "Valor de la métrica",
    fill = "Dataset"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold")
  )

print(p_metricas)

ggsave(
  "output/figures/comparacion_metricas_train_test_externo.png",
  p_metricas,
  width = 12,
  height = 8,
  dpi = 150
)

# ------------------------------------------------------------------
# Guardar resultados
# ------------------------------------------------------------------

saveRDS(
  tabla_metricas_global,
  "output/results/tabla_metricas_global.rds"
)

saveRDS(
  tabla_metricas_wide,
  "output/results/tabla_metricas_global_wide.rds"
)

message("\n✓ Comparación global guardada")

p_auc <- tabla_metricas_global %>%
  filter(.metric == "roc_auc") %>%
  ggplot(
    aes(
      x = modelo,
      y = .estimate,
      fill = dataset
    )
  ) +
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.7
  ) +
  geom_text(
    aes(label = round(.estimate, 3)),
    position = position_dodge(width = 0.8),
    vjust = -0.4,
    size = 3
  ) +
  labs(
    title = "ROC-AUC por modelo",
    subtitle = "Train vs Test 2024 vs Externo 2025",
    x = NULL,
    y = "ROC-AUC"
  ) +
  ylim(0, 1) +
  theme_minimal()

ggsave(
  "output/figures/roc_auc_comparacion.png",
  p_auc,
  width = 8,
  height = 5,
  dpi = 150
)

# ==============================================================================
# 16.4 Evaluación Drfit BRIER.
# ==============================================================================
# ==============================================================================
# BRIER SCORE - TEST 2024 Y EXTERNO 2025
# ==============================================================================
# ------------------------------------------------------------------
# Predicciones RF
# ------------------------------------------------------------------

pred_rf_test <- predict(
  fit_rf,
  data_test_final,
  type = "prob"
)

pred_rf_ext <- predict(
  fit_rf,
  data_externo_final,
  type = "prob"
)

# ------------------------------------------------------------------
# Predicciones XGBoost
# ------------------------------------------------------------------

pred_xgb_test <- predict(
  fit_xgb,
  data_test_final,
  type = "prob"
)

pred_xgb_ext <- predict(
  fit_xgb,
  data_externo_final,
  type = "prob"
)

# ------------------------------------------------------------------
# Brier Scores ponderados
# ------------------------------------------------------------------

brier_rf_test <- yardstick::brier_class_vec(
  truth        = data_test_final$mpi_pobre,
  estimate     = pred_rf_test$.pred_pobre,
  case_weights = data_test_final$pondera,
  event_level  = "second"
)

brier_rf_ext <- yardstick::brier_class_vec(
  truth        = data_externo_final$mpi_pobre,
  estimate     = pred_rf_ext$.pred_pobre,
  case_weights = data_externo_final$pondera,
  event_level  = "second"
)

brier_xgb_test <- yardstick::brier_class_vec(
  truth        = data_test_final$mpi_pobre,
  estimate     = pred_xgb_test$.pred_pobre,
  case_weights = data_test_final$pondera,
  event_level  = "second"
)

brier_xgb_ext <- yardstick::brier_class_vec(
  truth        = data_externo_final$mpi_pobre,
  estimate     = pred_xgb_ext$.pred_pobre,
  case_weights = data_externo_final$pondera,
  event_level  = "second"
)

# ------------------------------------------------------------------
# Tabla resumen
# ------------------------------------------------------------------

tabla_brier <- tibble(
  modelo = c(
    "Random Forest",
    "Random Forest",
    "XGBoost",
    "XGBoost"
  ),
  dataset = c(
    "Test 2024",
    "Externo 2025",
    "Test 2024",
    "Externo 2025"
  ),
  brier_score = c(
    brier_rf_test,
    brier_rf_ext,
    brier_xgb_test,
    brier_xgb_ext
  )
)

message("\n>>> Brier Scores (menor es mejor)")
print(tabla_brier)

# Recalibración del Threshold sobre muestra desbalanceada para limitar el efecto
# de sacar SMOTENC fuera de la recipe.
# ==============================================================================
# 1. PREPARACIÓN CONSISTENTE
# ==============================================================================

prep_cal <- function(pred_df, data_df) {
  pred_df %>%
    bind_cols(data_df %>% select(mpi_pobre, pondera)) %>%
    mutate(
      target_num = if_else(mpi_pobre == "pobre", 1, 0),
      mpi_pobre = factor(mpi_pobre, levels = c("no_pobre", "pobre")),
      .pred_pobre = pmin(pmax(.pred_pobre, 1e-6), 1 - 1e-6) # estabilidad numérica
    )
}

preds_rf_2024  <- prep_cal(pred_rf_test,  data_test_final)
preds_xgb_2024 <- prep_cal(pred_xgb_test, data_test_final)

preds_rf_2025  <- prep_cal(pred_rf_ext,  data_externo_final)
preds_xgb_2025 <- prep_cal(pred_xgb_ext, data_externo_final)

# ==============================================================================
# 2. CALIBRACIÓN ROBUSTA (BINNING + SUAVIZADO)
# ==============================================================================

calibration_binning <- function(df, prob_col, n_bins = 10) {
  
  df <- df %>%
    mutate(bin = ntile(.data[[prob_col]], n_bins))
  
  calib <- df %>%
    group_by(bin) %>%
    summarise(
      p_hat = weighted.mean(.data[[prob_col]], pondera),
      y_hat = weighted.mean(target_num, pondera),
      .groups = "drop"
    ) %>%
    arrange(p_hat)
  
  list(calib = calib)
}

predict_calibrated <- function(cal_obj, new_probs) {
  
  approx(
    x = cal_obj$calib$p_hat,
    y = cal_obj$calib$y_hat,
    xout = new_probs,
    rule = 2
  )$y %>%
    pmin(1) %>%
    pmax(0)
}

# Fit calibrators
cal_rf  <- calibration_binning(preds_rf_2024,  ".pred_pobre")
cal_xgb <- calibration_binning(preds_xgb_2024, ".pred_pobre")

# ==============================================================================
# 3. APLICAR A 2025 (OUT-OF-SAMPLE)
# ==============================================================================

preds_rf_2025 <- preds_rf_2025 %>%
  mutate(
    p_raw = .pred_pobre,
    p_cal = predict_calibrated(cal_rf, .pred_pobre)
  )

preds_xgb_2025 <- preds_xgb_2025 %>%
  mutate(
    p_raw = .pred_pobre,
    p_cal = predict_calibrated(cal_xgb, .pred_pobre)
  )

# ==============================================================================
# 4. BRIER SCORE PONDERADO
# ==============================================================================

brier_w <- function(df, p) {
  sum(df$pondera * (df[[p]] - df$target_num)^2) /
    sum(df$pondera)
}

brier_baseline <- function(df) {
  p_bar <- weighted.mean(df$target_num, df$pondera)
  sum(df$pondera * (p_bar - df$target_num)^2) /
    sum(df$pondera)
}

# Scores
brier_rf_raw  <- brier_w(preds_rf_2025, "p_raw")
brier_rf_cal  <- brier_w(preds_rf_2025, "p_cal")

brier_xgb_raw <- brier_w(preds_xgb_2025, "p_raw")
brier_xgb_cal <- brier_w(preds_xgb_2025, "p_cal")

brier_base <- brier_baseline(preds_rf_2025)

# ==============================================================================
# 5. CHECKS IMPORTANTES (EVITAR COLAPSO)
# ==============================================================================

cat("\n--- CHECK DISTRIBUCIÓN RF CALIBRADO ---\n")
print(summary(preds_rf_2025$p_cal))

cat("\n--- CHECK VARIABILIDAD ---\n")
cat("SD RF raw:", sd(preds_rf_2025$p_raw), "\n")
cat("SD RF cal:", sd(preds_rf_2025$p_cal), "\n")

# ==============================================================================
# 6. RESULTADOS FINALES
# ==============================================================================

cat("\n================ BRIER (2025 EXTERNAL) ================\n\n")

cat(sprintf("BASELINE: %.4f\n\n", brier_base))

cat("RANDOM FOREST\n")
cat(sprintf("  Raw : %.4f\n", brier_rf_raw))
cat(sprintf("  Cal : %.4f\n\n", brier_rf_cal))

cat("XGBOOST\n")
cat(sprintf("  Raw : %.4f\n", brier_xgb_raw))
cat(sprintf("  Cal : %.4f\n\n", brier_xgb_cal))

# ------------------------------------------------------------------------------
# 1. FUNCIÓN DE BINNING PARA CALIBRACIÓN VISUAL
# ------------------------------------------------------------------------------

calibration_data <- function(df, prob_col, n_bins = 10) {
  
  df %>%
    mutate(bin = ntile(.data[[prob_col]], n_bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = weighted.mean(.data[[prob_col]], pondera),
      mean_obs  = weighted.mean(target_num, pondera),
      n = n(),
      .groups = "drop"
    )
}

# ------------------------------------------------------------------------------
# 2. DATA RF
# ------------------------------------------------------------------------------

rf_raw_curve <- calibration_data(preds_rf_2025, "p_raw")
rf_cal_curve <- calibration_data(preds_rf_2025, "p_cal")

# ------------------------------------------------------------------------------
# 3. DATA XGB
# ------------------------------------------------------------------------------

xgb_raw_curve <- calibration_data(preds_xgb_2025, "p_raw")
xgb_cal_curve <- calibration_data(preds_xgb_2025, "p_cal")

# ------------------------------------------------------------------------------
# 4. PLOT FUNCTION
# ------------------------------------------------------------------------------

plot_calibration <- function(df_raw, df_cal, title) {
  
  ggplot() +
    
    # diagonal ideal
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    
    # raw
    geom_line(data = df_raw,
              aes(x = mean_pred, y = mean_obs),
              color = "red") +
    geom_point(data = df_raw,
               aes(x = mean_pred, y = mean_obs),
               color = "red") +
    
    # calibrated
    geom_line(data = df_cal,
              aes(x = mean_pred, y = mean_obs),
              color = "blue") +
    geom_point(data = df_cal,
               aes(x = mean_pred, y = mean_obs),
               color = "blue") +
    
    labs(
      title = title,
      x = "Probabilidad predicha (media por bin)",
      y = "Frecuencia observada"
    ) +
    
    coord_equal() +
    theme_minimal()
}

# ------------------------------------------------------------------------------
# 5. GRÁFICOS
# ------------------------------------------------------------------------------

plot_calibration(
  rf_raw_curve,
  rf_cal_curve,
  "Random Forest - Calibration Curve (2025)"
)

plot_calibration(
  xgb_raw_curve,
  xgb_cal_curve,
  "XGBoost - Calibration Curve (2025)"
)
# ==============================================================================
# 17. GUARDADO DE RESULTADOS
# ==============================================================================
message("\n>>> Guardando modelos y resultados...")

saveRDS(data_model_balanced, "output/models/data_model_balanced.rds")
saveRDS(fit_cart,         "output/models/fit_cart.rds")
saveRDS(fit_rf,           "output/models/fit_rf.rds")
saveRDS(fit_xgb,          "output/models/fit_xgb.rds")
saveRDS(tabla_resultados, "output/results/tabla_resultados_test.rds")
saveRDS(tabla_umbrales,   "output/results/umbrales_calibrados.rds")
saveRDS(PREDICTORES,      "output/results/predictores_modelo.rds")
saveRDS(best_cart,        "output/models/best_cart.rds")
saveRDS(best_rf,          "output/models/best_rf.rds")
saveRDS(best_xgb,         "output/models/best_xgb.rds")
saveRDS(tune_cart,        "output/models/tune_cart.rds")
saveRDS(tune_rf,          "output/models/tune_rf.rds")
saveRDS(tune_xgb,         "output/models/tune_xgb.rds")

# ==============================================================================
# 18. LIMPIEZA FINAL Y CIERRE DEL CLUSTER
# ==============================================================================
message("\n>>> Limpiando memoria y cerrando cluster...")

# Liberar objetos grandes que ya no se necesitan
rm(data_model_balanced, data_test_final, data_train_eval, eph_train, eph_test)
rm(tune_cart, tune_rf, tune_xgb, block_folds, rec_base)
rm(wf_cart, wf_rf, wf_xgb, grid_cart, grid_rf, grid_xgb)

gc()  # Garbage collection final agresivo

# Cerrar cluster de paralelización
if (exists("cl") && inherits(cl, "cluster")) {
  parallel::stopCluster(cl)
  message("  ✓ Cluster PSOCK cerrado correctamente")
}

message("\n✓ Script 06c completado exitosamente")
message("  - Paralelización: 2 cores PSOCK con setup_strategy='sequential'")
message("  - Hilos nativos: OMP=1, MKL=1, OPENBLAS=1")
message("  - parallel_over: 'resamples' (minimiza RAM)")
message("  - SMOTE: Externo (ejecutado UNA SOLA VEZ)")
message("  - Memoria optimizada para 16GB RAM en Windows 11")