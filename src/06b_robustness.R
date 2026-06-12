# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 06_robustness_analysis_final.R
# Propósito: Análisis de Robustez - Evaluación SIN max_instruccion
#            Usando los mejores parámetros ya calculados (SIN TUNING)
#
# METODOLOGÍA:
#   1. Cargar modelos y parámetros óptimos guardados del script 06_modelling.R
#   2. Excluir explícitamente max_instruccion de los predictores
#   3. Re-entrenar modelos con la misma configuración pero sin la variable
#   4. Evaluar en Test (2024) y Externo (2025) usando los umbrales calibrados
#   5. Guardar resultados de robustez consolidados
#
# NOTA: Este script NO hace tuning. Usa best_cart, best_rf, best_xgb guardados.
# ==============================================================================

message(">>> Iniciando Análisis de Robustez Estructural (Sin max_instruccion)...")

# ==============================================================================
# 0. CONFIGURACIÓN DE RENDIMIENTO
# ==============================================================================
# Control de hilos nativos para evitar sub-paralelización
Sys.setenv(
  OMP_NUM_THREADS = "1",
  OMP_THREAD_LIMIT = "1",
  MKL_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

# ==============================================================================
# 1. CARGA DE LIBRERÍAS
# ==============================================================================
library(tidyverse)
library(tidymodels)
library(themis)
library(FactoMineR)
library(janitor)

tidymodels_prefer()

message("  ✓ Librerías cargadas")

# ==============================================================================
# 2. CARGA DE DATOS Y OBJETOS GUARDADOS
# ==============================================================================
message(">>> Cargando datos y objetos del modelo principal...")

# Cargar datasets base
eph_train   <- readRDS("data/processed/eph_train_mca.rds")
eph_test    <- readRDS("data/processed/eph_test_ml.rds")
eph_externo <- readRDS("data/processed/eph_externo_ml.rds")

# Cargar mejores parámetros del tuning (sin re-tunear)
best_cart <- readRDS("output/models/best_cart.rds")
best_rf   <- readRDS("output/models/best_rf.rds")
best_xgb  <- readRDS("output/models/best_xgb.rds")

# Cargar tabla de umbrales calibrados (o recrear si no existe)
if (file.exists("output/results/umbrales_calibrados.rds")) {
  tabla_umbrales <- readRDS("output/results/umbrales_calibrados.rds")
  message("  ✓ Umbrales calibrados cargados desde archivo")
} else {
  message("  ! Archivo de umbrales no encontrado. Se recrearán umbrales por defecto.")
  message("  ! Para obtener umbrales precisos, ejecute primero 06_modelling.R")
  tabla_umbrales <- tibble(
    modelo = c("CART", "Random Forest", "XGBoost"),
    umbral = c(0.405, 0.430, 0.440)
  )
}

# ==============================================================================
# 2.1 RECREAR ESPECIFICACIONES DE MODELOS
# ==============================================================================
# Las especificaciones no se guardan explícitamente, las recreamos idénticas
# al script 06_modelling.R para poder armar los workflows
message("  Recreando especificaciones de modelos...")

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

message("  ✓ Especificaciones de modelos recreadas")

# Cargar predictores originales (para referencia)
if (file.exists("output/results/predictores_modelo.rds")) {
  PREDICTORES_ORIG <- readRDS("output/results/predictores_modelo.rds")
  message("  ✓ Predictores originales cargados")
} else {
  message("  ! predictores_modelo.rds no encontrado (se usará definición interna)")
  PREDICTORES_ORIG <- NULL
}

message("  ✓ Datos y objetos cargados exitosamente")
# gc()

# ==============================================================================
# 3. DEFINICIÓN DE PREDICTORES SIN max_instruccion
# ==============================================================================
message(">>> Definiendo predictores para análisis de robustez...")

# Variables MCA directas (no son input directo del MPI)
VARS_MCA_DIRECTAS <- c("v2", "v13", "v11", "iv1", "iv2", "ii7", "ii8")

# Construir predictores excluyendo explícitamente max_instruccion
PREDICTORES_ROB <- unique(c(
  grep("^mca_dim", names(eph_train), value = TRUE),
  "itcf_real", "p21_real", "tamano_hogar", "n_menores", "n_ancianos",
  "region_label", "ratio_dependencia", "n_ocupados", "adeq_hogar",
  "prop_informal", "priv_salud", VARS_MCA_DIRECTAS, "ano4", "trimestre"
))

# Exclusión explícita de max_instruccion (y variantes)
PREDICTORES_ROB <- setdiff(
  intersect(PREDICTORES_ROB, names(eph_train)),
  c("max_instruccion", "max_instruccion_hogar")
)

message("  Predictores de robustez: ", length(PREDICTORES_ROB))
message("  Variables excluidas: max_instruccion")

# ==============================================================================
# 4. PREPARACIÓN DEL DATASET DE ENTRENAMIENTO (SIN max_instruccion)
# ==============================================================================
message(">>> Preparando dataset de entrenamiento para robustez...")

data_model_rob <- eph_train %>%
  select(mpi_pobre, codusu, aglomerado, ano4, trimestre,
         pondera, all_of(PREDICTORES_ROB)) %>%
  mutate(
    mpi_pobre    = factor(mpi_pobre, levels = c("no_pobre", "pobre")),
    region_label = factor(region_label),
    aglomerado   = as.character(aglomerado),
    across(c(itcf_real, adeq_hogar, p21_real), as.numeric)
  ) %>%
  drop_na(mpi_pobre)

# Muestreo estratificado (mismo seed y proporción que script original)
set.seed(42)
data_model_rob <- data_model_rob %>%
  mutate(grupo_cv = paste0(codusu, "_", aglomerado)) %>%
  slice_sample(prop = 0.7, weight_by = pondera)

# ==============================================================================
# 4.1 REPONDERACIÓN REGIONAL (IDÉNTICA AL SCRIPT ORIGINAL)
# ==============================================================================
message("  Aplicando reponderación regional...")

# Distribución objetivo: estructura demográfica del test (marco 2022)
dist_test <- eph_externo %>%
  group_by(region_label) %>%
  summarise(
    prop_externo = sum(pondera) / sum(eph_externo$pondera),
    .groups = "drop"
  )

# Distribución actual del train de robustez
dist_train <- data_model_rob %>%
  group_by(region_label) %>%
  summarise(
    prop_train = sum(pondera) / sum(data_model_rob$pondera),
    .groups = "drop"
  )

# Calcular factor de ajuste por región
ajuste_regional <- dist_train %>%
  left_join(dist_test, by = "region_label") %>%
  mutate(
    factor_ajuste = if_else(
      is.na(prop_externo) | prop_train == 0,
      1,
      prop_externo / prop_train
    ),
    factor_ajuste = pmin(pmax(factor_ajuste, 0.2), 5.0)
  ) %>%
  select(region_label, factor_ajuste)

# Aplicar reponderación
data_model_rob <- data_model_rob %>%
  left_join(ajuste_regional, by = "region_label") %>%
  mutate(
    pondera_orig = pondera,
    pondera      = pondera * factor_ajuste,
    pondera      = pondera * (sum(pondera_orig) / sum(pondera))
  ) %>%
  select(-factor_ajuste, -pondera_orig)

message("  ✓ Dataset de robustez preparado: ", nrow(data_model_rob), " filas")

# Liberar objetos intermedios
# rm(dist_test, dist_train, ajuste_regional)
# # gc()

# ==============================================================================
# 4.2 SMOTENC EXTERNO - SE EJECUTA UNA SOLA VEZ (FUERA DEL CV)
# ==============================================================================
# IMPORTANTE: Mantenemos consistencia con el script 06_modelling.R aplicando
# SMOTENC externamente (fuera del recipe y de los folds) para:
# 1. Evitar data leakage (SMOTE dentro de CV es correcto, pero fuera es más rápido)
# 2. Mantener consistencia metodológica con el modelo original
# 3. Reducir tiempo de cómputo (se ejecuta UNA SOLA VEZ)
#
# over_ratio = 0.25: la clase minoritaria tendrá 25% de la mayoritaria

message(">>> Aplicando SMOTENC externo (una sola vez)...")

# Paso 1: Crear dataset con índice para tracking
data_rob_with_index <- data_model_rob %>%
  mutate(.smote_id = row_number())

# Paso 2: Preparar predictores para SMOTENC (sin columnas técnicas)
data_rob_predictores <- data_rob_with_index %>%
  select(mpi_pobre, .smote_id, all_of(PREDICTORES_ROB)) %>%
  mutate(across(where(is.character), as.factor))

# Paso 3: Aplicar SMOTENC
set.seed(42)
rec_smotenc_rob <- recipe(mpi_pobre ~ ., data = data_rob_predictores) %>%
  update_role(.smote_id, new_role = "id") %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  themis::step_smotenc(mpi_pobre, over_ratio = 0.25, neighbors = 5)

message("  Ejecutando SMOTENC (prep + bake)...")
rec_smotenc_rob_prep <- prep(rec_smotenc_rob, training = data_rob_predictores, verbose = FALSE)
data_rob_balanced_with_id <- bake(rec_smotenc_rob_prep, new_data = NULL)

# Paso 4: Unir columnas técnicas desde data_rob_with_index usando .smote_id
cols_tecnicas_rob <- data_rob_with_index %>%
  select(.smote_id, codusu, aglomerado, grupo_cv, pondera)

# Separar filas originales y sintéticas
filas_orig_rob <- data_rob_balanced_with_id %>%
  filter(.smote_id %in% data_rob_with_index$.smote_id)

filas_sint_rob <- data_rob_balanced_with_id %>%
  filter(is.na(.smote_id) | !(.smote_id %in% data_rob_with_index$.smote_id))

# Unir columnas técnicas para filas originales
filas_orig_rob <- filas_orig_rob %>%
  left_join(cols_tecnicas_rob, by = ".smote_id")

# Para filas sintéticas: asignar columnas técnicas basadas en distribución de mpi_pobre
if (nrow(filas_sint_rob) > 0) {
  muestras_tecnicas_rob <- cols_tecnicas_rob %>%
    left_join(data_rob_with_index %>% select(.smote_id, mpi_pobre), by = ".smote_id")
  
  filas_sint_rob_con_tecnicas <- filas_sint_rob %>%
    group_by(mpi_pobre, .drop = TRUE) %>%
    group_modify(~{
      muestras <- muestras_tecnicas_rob %>%
        filter(mpi_pobre == .y$mpi_pobre) %>%
        slice_sample(n = nrow(.x), replace = TRUE) %>%
        select(-mpi_pobre, -.smote_id)
      bind_cols(.x, muestras)
    }) %>%
    ungroup()
  
  data_model_rob_balanced <- bind_rows(filas_orig_rob, filas_sint_rob_con_tecnicas)
} else {
  data_model_rob_balanced <- filas_orig_rob
}

data_model_rob_balanced <- data_model_rob_balanced %>%
  select(-any_of(".smote_id"))

message("  Distribución DESPUÉS de SMOTENC:")
print(table(data_model_rob_balanced$mpi_pobre))
message("  Filas originales: ", nrow(data_rob_predictores),
        " → Filas después de SMOTENC: ", nrow(data_model_rob_balanced))

# Liberar objetos temporales
# rm(data_rob_with_index, data_rob_predictores, rec_smotenc_rob, rec_smotenc_rob_prep,
#    data_rob_balanced_with_id, cols_tecnicas_rob, filas_orig_rob, filas_sint_rob)
# if (exists("muestras_tecnicas_rob")) rm(muestras_tecnicas_rob)
# if (exists("filas_sint_rob_con_tecnicas")) rm(filas_sint_rob_con_tecnicas)
# gc()

message("  ✓ SMOTENC externo completado (ejecución única)")

# Actualizar data_model_rob con el dataset balanceado
data_model_rob <- data_model_rob_balanced
# rm(data_model_rob_balanced)

# ==============================================================================
# 5. NUEVO RECIPE ESTRUCTURAL (SIN max_instruccion, SIN SMOTE)
# ==============================================================================
message(">>> Creando recipe para análisis de robustez...")

rec_base_rob <- recipe(
  mpi_pobre ~ .,
  data = data_model_rob %>% select(-aglomerado, -pondera, -codusu, -grupo_cv)
) %>%
  # ano4 se usa para crear año_norm pero NO debe ser predictor
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

message("  ✓ Recipe creado (sin SMOTE - solo transformaciones)")

# ==============================================================================
# 6. ENSAMBLE DE WORKFLOWS DE ROBUSTEZ
# ==============================================================================
message(">>> Ensamblando workflows de robustez...")

wf_cart_rob <- workflow() %>% add_recipe(rec_base_rob) %>% add_model(spec_cart)
wf_rf_rob   <- workflow() %>% add_recipe(rec_base_rob) %>% add_model(spec_rf)
wf_xgb_rob  <- workflow() %>% add_recipe(rec_base_rob) %>% add_model(spec_xgb)

message("  ✓ Workflows ensamblados (CART, RF, XGB)")

# ==============================================================================
# 7. FIT DIRECTO (SIN TUNING) CON MEJORES PARÁMETROS EXISTENTES
# ==============================================================================
message(">>> Ajustando modelos de robustez (usando mejores parámetros guardados)...")

fit_cart_rob <- finalize_workflow(wf_cart_rob, best_cart) %>%
  fit(data = data_model_rob)

fit_rf_rob <- finalize_workflow(wf_rf_rob, best_rf) %>%
  fit(data = data_model_rob)

fit_xgb_rob <- finalize_workflow(wf_xgb_rob, best_xgb) %>%
  fit(data = data_model_rob)

message("  ✓ Modelos de robustez ajustados exitosamente")

# # Liberar objetos que ya no se necesitan
# rm(wf_cart_rob, wf_rf_rob, wf_xgb_rob, rec_base_rob)
# gc()

# ==============================================================================
# 8. EVALUACIÓN EN TEST 2024 Y EXTERNO 2025
# ==============================================================================
message(">>> Evaluando modelos de robustez en Test y Externo...")

# ==============================================================================
# 8.1 RECALIBRACIÓN DINÁMICA DE UMBRALES (MAXIMIZANDO F_MEAS)
# ==============================================================================
message(">>> Recalibrando umbrales óptimos sobre el nuevo Train maximizando F_meas...")

obtener_umbral_f1 <- function(fit_model, data_train) {
  # 1. Predecir probabilidades en el dataset de entrenamiento de robustez
  preds_train <- predict(fit_model, data_train, type = "prob") %>%
    bind_cols(data_train %>% select(mpi_pobre, pondera)) %>%
    mutate(mpi_pobre = factor(mpi_pobre, levels = c("no_pobre", "pobre")))
  
  # 2. Evaluar un vector de umbrales candidatos desde 0.05 hasta 0.95
  umbrales_candidatos <- seq(0.05, 0.95, by = 0.005)
  
  resultados_f1 <- map_dfr(umbrales_candidatos, function(u) {
    preds_train %>%
      mutate(
        .pred_class = factor(
          if_else(.pred_pobre >= u, "pobre", "no_pobre"),
          levels = c("no_pobre", "pobre")
        )
      ) %>%
      yardstick::f_meas(
        truth = mpi_pobre, 
        estimate = .pred_class, 
        event_level = "second", 
        case_weights = pondera
      ) %>%
      mutate(threshold = u)
  })
  
  # 3. Extraer el umbral que maximiza el F1-score (f_meas)
  umbral_optimo <- resultados_f1 %>%
    filter(.estimate == max(.estimate, na.rm = TRUE)) %>%
    slice(1) %>%
    pull(threshold)
  
  return(umbral_optimo)
}

# Calcular los nuevos umbrales adaptados y optimizados por F1
u_cart <- obtener_umbral_f1(fit_cart_rob, data_model_rob)
u_rf   <- obtener_umbral_f1(fit_rf_rob,   data_model_rob)
u_xgb  <- obtener_umbral_f1(fit_xgb_rob,  data_model_rob)

message("  ✓ Nuevos umbrales optimizados por F1: CART =", round(u_cart, 3), 
        " | RF =", round(u_rf, 3), " | XGB =", round(u_xgb, 3))

# ------------------------------------------------------------------------------
# Función auxiliar: crear dummies de región exactamente como en train
# ------------------------------------------------------------------------------
crear_dummies_region <- function(df) {
  regiones <- c("Cuyo", "GBA", "Nordeste", "Noroeste", "Pampeana", "Patagonia")
  
  for (r in regiones) {
    df[[paste0("region_label_", r)]] <- as.integer(as.character(df$region_label) == r)
  }
  
  df %>% select(-region_label)
}

# ------------------------------------------------------------------------------
# Función auxiliar: match_classes (alinear tipos entre datasets)
# ------------------------------------------------------------------------------
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
# Función de evaluación (siguiendo patrón del script 06_modelling.R)
# ------------------------------------------------------------------------------
evaluar_modelo_rob <- function(fit, data, nombre, umbral = 0.5) {
  
  # 1. Generar predicciones probabilísticas usando el workflow ajustado
  pred_prob <- predict(fit, data, type = "prob")
  
  # 2. Armar el dataframe consolidado con la clase predicha según el umbral fijo
  preds_consolidadas <- bind_cols(
    data %>% select(mpi_pobre, pondera),
    pred_prob
  ) %>%
    mutate(
      mpi_pobre = factor(mpi_pobre, levels = c("no_pobre", "pobre")),
      .pred_class = factor(
        if_else(.pred_pobre >= umbral, "pobre", "no_pobre"),
        levels = c("no_pobre", "pobre")
      )
    ) %>%
    drop_na(mpi_pobre)
  
  # 3. Calcular métricas individuales usando yardstick (inyectando case_weights)
  met_roc  <- yardstick::roc_auc(preds_consolidadas, truth = mpi_pobre, .pred_pobre, 
                                 event_level = "second", case_weights = pondera)
  
  met_f1   <- yardstick::f_meas(preds_consolidadas, truth = mpi_pobre, estimate = .pred_class, 
                                event_level = "second", case_weights = pondera)
  
  met_acc  <- yardstick::accuracy(preds_consolidadas, truth = mpi_pobre, estimate = .pred_class, 
                                  case_weights = pondera)
  
  met_sens <- yardstick::sensitivity(preds_consolidadas, truth = mpi_pobre, estimate = .pred_class, 
                                     event_level = "second", case_weights = pondera)
  
  met_spec <- yardstick::specificity(preds_consolidadas, truth = mpi_pobre, estimate = .pred_class, 
                                     event_level = "second", case_weights = pondera)
  
  met_kap  <- yardstick::kap(preds_consolidadas, truth = mpi_pobre, estimate = .pred_class, 
                             case_weights = pondera)
  
  met_brier <- yardstick::brier_class(preds_consolidadas, truth = mpi_pobre, .pred_pobre, 
                                      event_level = "second", case_weights = pondera)
  
  # 4. Consolidar todas las métricas en la estructura esperada por el script
  bind_rows(met_roc, met_f1, met_acc, met_sens, met_spec, met_kap, met_brier) %>%
    mutate(
      modelo = nombre,
      umbral_usado = umbral
    )
}

# ==============================================================================
# 8.1 VERIFICAR/PROYECTAR MCA SOBRE TEST Y EXTERNO
# ==============================================================================
# Los datasets pueden ya tener mca_dim1 y mca_dim2 proyectados.
# Si no, hay que proyectarlos.

message("  Verificando dimensiones MCA en datasets...")

# Verificar si ya existen mca_dim en eph_test
if (all(c("mca_dim1", "mca_dim2") %in% names(eph_test))) {
  message("    eph_test ya tiene mca_dim1, mca_dim2")
  eph_test_mca <- eph_test
} else {
  message("    eph_test NO tiene mca_dim - proyectando MCA...")
  
  VARS_MCA_ORIG <- c(
    "v2", "v13", "v5", "v11", "v12",
    "iv1", "iv2", "iv5",
    "ii7", "ii8", "iv10", "ii9"
  )
  
  # Entrenar MCA sobre años recientes del train
  df_mca_reciente <- eph_train %>%
    filter(ano4 >= 2020) %>%
    select(all_of(VARS_MCA_ORIG)) %>%
    mutate(across(everything(), ~ factor(as.character(.)))) %>%
    mutate(across(everything(), ~ fct_lump_prop(., prop = 0.02, other_level = "Otros")))
  
  res_mca_reciente <- MCA(df_mca_reciente, ncp = 2, graph = FALSE)
  v_coord <- res_mca_reciente$var$coord
  col_names_mca <- rownames(v_coord)
  ncp <- ncol(v_coord)
  niveles_mca <- lapply(df_mca_reciente, levels)
  
  # Función para alinear categorías
  alinear_mca <- function(df_test, niveles_ref) {
    for (col in names(niveles_ref)) {
      niveles_validos <- niveles_ref[[col]]
      df_test[[col]] <- as.character(df_test[[col]])
      df_test[[col]] <- if_else(df_test[[col]] %in% niveles_validos, df_test[[col]], "Otros")
      df_test[[col]] <- factor(df_test[[col]], levels = niveles_validos)
    }
    df_test
  }
  
  # Función para construir tabla disyuntiva
  construir_disjuntiva <- function(df, niveles_ref) {
    result <- vector("list", length(niveles_ref))
    names(result) <- names(niveles_ref)
    
    for (col in names(niveles_ref)) {
      nivs <- niveles_ref[[col]]
      mat <- model.matrix(~ 0 + ., data = data.frame(x = df[[col]]))
      colnames(mat) <- paste0(col, "_", nivs[match(sub("^x", "", colnames(mat)), nivs)])
      result[[col]] <- mat
    }
    do.call(cbind, result)
  }
  
  # Proyectar MCA para TEST
  df_test_alineado <- eph_test %>%
    select(all_of(VARS_MCA_ORIG)) %>%
    mutate(across(everything(), as.character)) %>%
    alinear_mca(niveles_mca)
  
  Z_test <- construir_disjuntiva(df_test_alineado, niveles_mca)
  
  # Alinear columnas
  cols_falt <- setdiff(col_names_mca, colnames(Z_test))
  if (length(cols_falt) > 0) {
    Z_test <- cbind(Z_test, matrix(0, nrow = nrow(Z_test), ncol = length(cols_falt),
                                   dimnames = list(NULL, cols_falt)))
  }
  Z_test <- Z_test[, col_names_mca, drop = FALSE]
  
  coords_test <- (Z_test / length(VARS_MCA_ORIG)) %*% v_coord
  mca_test_proj <- as.data.frame(coords_test) %>%
    setNames(paste0("mca_dim", seq_len(ncp))) %>%
    mutate(across(everything(), ~ if_else(is.na(.), 0, .)))
  
  # Eliminar columnas mca_dim existentes si las hay
  eph_test_mca <- eph_test %>% select(-any_of(paste0("mca_dim", 1:ncp)))
  eph_test_mca <- bind_cols(eph_test_mca, mca_test_proj)
}

# Verificar si ya existen mca_dim en eph_externo
if (all(c("mca_dim1", "mca_dim2") %in% names(eph_externo))) {
  message("    eph_externo ya tiene mca_dim1, mca_dim2")
  eph_externo_mca <- eph_externo
} else {
  message("    eph_externo NO tiene mca_dim - proyectando MCA...")
  
  # Proyectar MCA para EXTERNO (usando mismo MCA que para test)
  df_externo_alineado <- eph_externo %>%
    select(all_of(VARS_MCA_ORIG)) %>%
    mutate(across(everything(), as.character)) %>%
    alinear_mca(niveles_mca)
  
  Z_externo <- construir_disjuntiva(df_externo_alineado, niveles_mca)
  
  cols_falt_ext <- setdiff(col_names_mca, colnames(Z_externo))
  if (length(cols_falt_ext) > 0) {
    Z_externo <- cbind(Z_externo, matrix(0, nrow = nrow(Z_externo), ncol = length(cols_falt_ext),
                                         dimnames = list(NULL, cols_falt_ext)))
  }
  Z_externo <- Z_externo[, col_names_mca, drop = FALSE]
  
  coords_externo <- (Z_externo / length(VARS_MCA_ORIG)) %*% v_coord
  mca_externo_proj <- as.data.frame(coords_externo) %>%
    setNames(paste0("mca_dim", seq_len(ncp)))
  
  # Eliminar columnas mca_dim existentes si las hay
  eph_externo_mca <- eph_externo %>% select(-any_of(paste0("mca_dim", 1:ncp)))
  eph_externo_mca <- bind_cols(eph_externo_mca, mca_externo_proj)
}

message("  ✓ Dimensiones MCA verificadas/proyectadas")

# ==============================================================================
# 8.2 PREPARAR DATOS DE TEST (siguiendo patrón exacto del script 06_modelling.R)
# ==============================================================================
data_test_rob_final <- eph_test_mca %>%
  mutate(grupo_cv = paste0(codusu, "_", aglomerado)) %>%
  select(all_of(c("mpi_pobre", "codusu", "grupo_cv", "aglomerado", "ano4", "pondera", PREDICTORES_ROB))) %>%
  mutate(mpi_pobre = factor(mpi_pobre, levels = c("no_pobre", "pobre"))) %>%
  crear_dummies_region() %>%
  match_classes(reference = data_model_rob) %>%
  drop_na(mpi_pobre)

# Asegurar que existen todas las columnas del train
faltantes_test <- setdiff(names(data_model_rob), names(data_test_rob_final))
for (v in faltantes_test) {
  data_test_rob_final[[v]] <- 0
}
data_test_rob_final <- data_test_rob_final %>% select(all_of(names(data_model_rob)))

# ==============================================================================
# 8.3 PREPARAR DATOS DE EXTERNO
# ==============================================================================
data_externo_rob_final <- eph_externo_mca %>%
  mutate(grupo_cv = paste0(codusu, "_", aglomerado)) %>%
  select(all_of(c("mpi_pobre", "codusu", "grupo_cv", "aglomerado", "ano4", "pondera", PREDICTORES_ROB))) %>%
  mutate(mpi_pobre = factor(mpi_pobre, levels = c("no_pobre", "pobre"))) %>%
  crear_dummies_region() %>%
  match_classes(reference = data_model_rob) %>%
  drop_na(mpi_pobre)

faltantes_ext <- setdiff(names(data_model_rob), names(data_externo_rob_final))
for (v in faltantes_ext) {
  data_externo_rob_final[[v]] <- 0
}
data_externo_rob_final <- data_externo_rob_final %>% select(all_of(names(data_model_rob)))

# ==============================================================================
# 8.4 EVALUAR EN TEST 2024
# ==============================================================================
message("  Evaluando en Test 2024...")

tabla_test_rob <- bind_rows(
  evaluar_modelo_rob(fit_cart_rob, data_test_rob_final, "CART", umbral = u_cart),
  evaluar_modelo_rob(fit_rf_rob, data_test_rob_final, "Random Forest", umbral = u_rf),
  evaluar_modelo_rob(fit_xgb_rob, data_test_rob_final, "XGBoost", umbral = u_xgb)
) %>% mutate(conjunto = "Test 2024 Robustez")

# ==============================================================================
# 8.5 EVALUAR EN EXTERNO 2025
# ==============================================================================
message("  Evaluando en Externo 2025...")

tabla_ext_rob <- bind_rows(
  evaluar_modelo_rob(fit_cart_rob, data_externo_rob_final, "CART", umbral = u_cart),
  evaluar_modelo_rob(fit_rf_rob, data_externo_rob_final, "Random Forest", umbral = u_rf),
  evaluar_modelo_rob(fit_xgb_rob, data_externo_rob_final, "XGBoost", umbral = u_xgb)
) %>% mutate(conjunto = "Externo 2025 Robustez")

# ==============================================================================
# 9. CONSOLIDAR RESULTADOS
# ==============================================================================
message(">>> Consolidando resultados de robustez...")

tabla_robustez_consolidada <- bind_rows(tabla_test_rob, tabla_ext_rob)

# Vista para reporte (formato ancho)
tabla_tfm_view <- tabla_robustez_consolidada %>%
  select(modelo, conjunto, .metric, .estimate, umbral_usado) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

# ==============================================================================
# 10. REPORTE FINAL
# ==============================================================================
cat("\n")
cat("======================================================================\n")
cat("   REPORTE FINAL DE ROBUSTEZ: MODELO SIN max_instruccion\n")
cat("======================================================================\n\n")
print(tabla_tfm_view, n = Inf)
cat("\n======================================================================\n")

# ==============================================================================
# 11. GUARDADO DE RESULTADOS
# ==============================================================================
message(">>> Guardando resultados del análisis de robustez...")

# Asegurar que el directorio existe
if (!dir.exists("data/processed")) {
  dir.create("data/processed", recursive = TRUE)
}

# Guardar tabla consolidada
saveRDS(tabla_robustez_consolidada, "data/processed/tabla_robustez_consolidada.rds")

message("  ✓ Resultados guardados en: 'data/processed/tabla_robustez_consolidada.rds'")

# ==============================================================================
# 12. LIMPIEZA FINAL
# ==============================================================================
message(">>> Limpiando memoria...")

rm(eph_train, eph_test, eph_externo)
rm(best_cart, best_rf, best_xgb)
rm(spec_cart, spec_rf, spec_xgb)
rm(fit_cart_rob, fit_rf_rob, fit_xgb_rob)

gc()

message("\n✓ Script 06_robustness_analysis_final.R completado exitosamente")
message("  - Variable excluida: max_instruccion")
message("  - Sin tuning (usa mejores parámetros guardados)")
message("  - Evaluación en Test 2024 y Externo 2025 completada")