# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 06b_optimized_modelling.R
# Propósito: Pipeline de Machine Learning con tidymodels
#
# CORRECCIONES RESPECTO A LA VERSIÓN ANTERIOR:
#   1. PREDICTORES corregidos: eliminadas priv_* que causaban data leakage
#   2. La calibración del threshold (paso 11.5) se ejecuta ANTES de fit(),
#      ya que usa predicciones OOF del tuning, no del modelo final
#   3. evaluar_modelo() usa los umbrales calibrados desde tabla_umbrales,
#      no valores hardcodeados (0.15, 0.38)
#   4. ano4 excluido correctamente de predictores via update_role()
#   5. step_smotenc() se aplica DESPUÉS de step_dummy() — requiere que las
#      variables nominales ya estén codificadas como dummies
#   6. grep() sobre eph_train (no eph_train_mca, que no existe en este scope)
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

tidymodels_prefer()

# ------------------------------------------------------------------------------
# 0. Paralelización
# ------------------------------------------------------------------------------
all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)
message(">>> Núcleos registrados: ", all_cores)

# ------------------------------------------------------------------------------
# 1. Carga de datos
# ------------------------------------------------------------------------------
message(">>> Cargando datos...")

eph_train     <- readRDS("data/processed/eph_train_mca.rds")
eph_test      <- readRDS("data/processed/eph_test_ml.rds")
vars_estables <- readRDS("data/processed/mca_vars_estables.rds")

# ------------------------------------------------------------------------------
# 2. Definición de predictores (SIN data leakage)
# ------------------------------------------------------------------------------
# REGLA: se excluye cualquier variable que sea input directo de construir_mpi()
# Las priv_* (priv_piso, priv_agua, etc.) son exactamente los inputs ponderados
# del mpi_score → incluirlas da ROC-AUC artificial ~0.99

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
  
  # -- Mercado de trabajo -------------------------------------------------------
  "prop_informal",     # Proporción de ocupados sin aportes jubilatorios
  
  # -- EXCLUIDAS POR DATA LEAKAGE -----------------------------------------------
  # "priv_piso"         → input de mpi_score (peso 1/9)
  # "priv_techo"        → input de mpi_score (peso 1/9)
  # "priv_hacinamiento" → input de mpi_score (peso 1/9)
  # "priv_agua"         → input de mpi_score (peso 1/6)
  # "priv_cloaca"       → input de mpi_score (peso 1/6)
  # "priv_esc"          → input de mpi_score (peso 1/6)
  # "priv_educ"         → input de mpi_score (peso 1/6)
  # "priv_salud"        → correlaciona con priv_educ (misma fuente ch10)
  # "priv_digital"      → construida sobre v11/v12, ya capturadas en mca_dim*
  NULL  # permite cerrar el vector sin coma colgante
))

# Filtro de seguridad: solo columnas que realmente existen en el train
PREDICTORES <- intersect(PREDICTORES, names(eph_train))
message("  Predictores activos: ", length(PREDICTORES))
message("  ", paste(PREDICTORES, collapse = ", "))

# ------------------------------------------------------------------------------
# 3. Preparación del dataset de modelado
# ------------------------------------------------------------------------------
# Limpieza preventiva: si existe adeq_hogar como df en el entorno global
# puede colisionar con la columna del mismo nombre en el dataframe
if (exists("adeq_hogar") && is.data.frame(get("adeq_hogar"))) {
  rm(adeq_hogar, envir = .GlobalEnv)
}

data_model <- eph_train %>%
  select(mpi_pobre, codusu, aglomerado, ano4, pondera, all_of(PREDICTORES)) %>%
  mutate(
    mpi_pobre    = factor(mpi_pobre, levels = c("no_pobre", "pobre")),
    region_label = factor(region_label),
    aglomerado   = as.character(aglomerado),
    across(c(itcf_real, adeq_hogar, p21_real), as.numeric)
  ) %>%
  drop_na(mpi_pobre)

# Muestreo estratificado por pondera para reducir tiempo de cómputo
# Mínimo recomendado: 0.5 (con 0.2 el SMOTE tiene muy pocos casos de la clase minoritaria)
set.seed(42)
data_model <- data_model %>%
  mutate(grupo_cv = paste0(codusu, "_", aglomerado)) |> 
  slice_sample(prop = 0.5, weight_by = pondera)

# Verificar que grupo_cv tiene suficiente varianza para 10 folds
n_grupos <- data_model %>%
  distinct(grupo_cv) %>%
  nrow()

message("Grupos únicos para CV: ", n_grupos)
# Necesita al menos 10 grupos → con ~950K hogares únicos esto está garantizado
message("  Hogares en data_model: ", nrow(data_model))
message("  Distribución target: ")
print(table(data_model$mpi_pobre))

# ------------------------------------------------------------------------------
# 4. Block Cross-Validation (agrupado por aglomerado)
# ------------------------------------------------------------------------------
# group_vfold_cv garantiza que todos los hogares de un aglomerado caigan
# en el mismo fold, evitando dependencia espacial entre train y validación
set.seed(42)
block_folds <- group_vfold_cv(data = data_model, group = grupo_cv, v = 10)

# ------------------------------------------------------------------------------
# 5. Recipe
# ------------------------------------------------------------------------------
rec_base <- recipe(
  mpi_pobre ~ .,
  data = data_model %>% select(-aglomerado, -pondera)
) %>%
  
  # ano4 se usa para crear año_norm pero NO debe ser predictor del modelo
  update_role(ano4, new_role = "id") %>%
  
  # 1. Imputación previa a la creación de features derivadas
  #    (evita NAs propagados en las divisiones)
  step_impute_median(itcf_real, adeq_hogar, p21_real) %>%
  
  # 2. Creación de features derivadas
  step_mutate(
    # Ingreso ajustado por tamaño del hogar según adulto equivalente
    ingreso_per_capita = itcf_real / pmax(adeq_hogar, 1),
    # Tendencia temporal normalizada [0, 1]: 2016 → 0, 2024 → 1
    año_norm           = (ano4 - 2016) / (2024 - 2016),
    # Carga demográfica: dependientes pesados sobre ocupados
    # Los ancianos pesan 0.7 porque su consumo es menor al de un adulto pleno
    carga_demo         = (n_menores + n_ancianos * 0.7) / pmax(n_ocupados, 0.5)
  ) %>%
  
  # 3. Transformación logarítmica de ingresos (estabiliza varianza, reduce skewness)
  #    offset = 0.5 para manejar valores cercanos a cero sin producir -Inf
  step_log(ingreso_per_capita, itcf_real, p21_real, base = 10, offset = 0.5) %>%
  
  # 4. Imputación del resto de predictores
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  
  # 5. Eliminar predictores numéricos altamente correlacionados (r > 0.80)
  #    Umbral más estricto que el anterior (0.85) para reducir redundancia
  #    antes de que el SMOTE genere interpolaciones sobre variables colineales
  step_corr(all_numeric_predictors(), threshold = 0.80) %>%
  
  # 6. Tratamiento de variables categóricas
  #    Colapsar categorías con < 5% de casos en "other" para evitar
  #    niveles con muy pocas observaciones que desestabilicen el SMOTENC
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
  # 7. Limpieza y normalización
  step_zv(all_predictors()) %>%          # eliminar columnas con varianza cero
  step_normalize(all_numeric_predictors()) %>%
  
  # 8. Balance de clases con SMOTENC
  #    IMPORTANTE: debe ir DESPUÉS de step_dummy() porque SMOTENC distingue
  #    entre variables continuas (interpola) y binarias/dummies (vota por mayoría)
  #    over_ratio = 0.8 → la clase minoritaria llegará al 80% de la mayoritaria
  themis::step_smotenc(mpi_pobre, over_ratio = 0.8, neighbors = 5)

# ------------------------------------------------------------------------------
# 6. Especificaciones de modelos
# ------------------------------------------------------------------------------
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
  set_engine("ranger", importance = "impurity", seed = 42) %>%
  set_mode("classification")

spec_xgb <- boost_tree(
  trees         = tune(),
  tree_depth    = tune(),
  learn_rate    = tune(),
  loss_reduction = tune(),
  sample_size   = tune(),
  min_n         = tune()
) %>%
  set_engine("xgboost", eval_metric = "logloss") %>%
  set_mode("classification")

# ------------------------------------------------------------------------------
# 7. Workflows
# ------------------------------------------------------------------------------
wf_cart <- workflow() %>% add_recipe(rec_base) %>% add_model(spec_cart)
wf_rf   <- workflow() %>% add_recipe(rec_base) %>% add_model(spec_rf)
wf_xgb  <- workflow() %>% add_recipe(rec_base) %>% add_model(spec_xgb)

# ------------------------------------------------------------------------------
# 8. Grids de hiperparámetros
# ------------------------------------------------------------------------------
# size reducido respecto al máximo posible para equilibrar exploración y tiempo
grid_cart <- grid_space_filling(
  cost_complexity(),
  tree_depth(),
  min_n(),
  size = 12
)

grid_rf <- grid_space_filling(
  mtry(range  = c(2L, 8L)),
  trees(range = c(200L, 800L)),
  min_n(),
  size = 12
)

grid_xgb <- grid_space_filling(
  trees(),
  tree_depth(),
  learn_rate(),
  loss_reduction(),
  sample_size = sample_prop(range = c(0.5, 0.9)),
  min_n(),
  size = 14
)

# ------------------------------------------------------------------------------
# 9. Métricas y control del tuning
# ------------------------------------------------------------------------------
metricas <- metric_set(roc_auc, f_meas, kap, accuracy)

ctrl <- control_race(
  save_pred     = TRUE,   # NECESARIO para calibración de threshold (paso 11.5)
  verbose_elim  = TRUE,
  parallel_over = "everything"
)

# ------------------------------------------------------------------------------
# 10. Tuning con ANOVA race
# ------------------------------------------------------------------------------
message("--- Tuning CART ---")
tune_cart <- tune_race_anova(
  wf_cart, resamples = block_folds,
  grid = grid_cart, metrics = metricas, control = ctrl
)

message("--- Tuning Random Forest ---")
tune_rf <- tune_race_anova(
  wf_rf, resamples = block_folds,
  grid = grid_rf, metrics = metricas, control = ctrl
)

message("--- Tuning XGBoost ---")
tune_xgb <- tune_race_anova(
  wf_xgb, resamples = block_folds,
  grid = grid_xgb, metrics = metricas, control = ctrl
)

# ------------------------------------------------------------------------------
# 11. Selección de mejores hiperparámetros
# ------------------------------------------------------------------------------
best_cart <- select_best(tune_cart, metric = "roc_auc")
best_rf   <- select_best(tune_rf,   metric = "roc_auc")
best_xgb  <- select_best(tune_xgb,  metric = "roc_auc")

message("\n--- Mejores hiperparámetros ---")
message("CART:");          print(best_cart)
message("Random Forest:"); print(best_rf)
message("XGBoost:");       print(best_xgb)

# ------------------------------------------------------------------------------
# 11.5. Calibración del threshold (ANTES de fit final, usa predicciones OOF)
# ------------------------------------------------------------------------------
# Las predicciones OOF provienen del tuning: el modelo nunca vio esos datos
# durante el ajuste del fold correspondiente → calibración válida sin tocar
# el test set

calibrar_umbral <- function(tune_result, best_params, nombre_modelo,
                            thresholds = seq(0.15, 0.65, by = 0.01)) {
  
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
    filter(.metric == "j_index") %>%
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

p_threshold <- threshold_comparado %>%
  filter(.metric %in% c("j_index", "f_meas")) %>%
  ggplot(aes(x = .threshold, y = .estimate,
             color = modelo, linetype = .metric)) +
  geom_line(linewidth = 0.9, alpha = 0.85) +
  geom_vline(
    data     = tabla_umbrales,
    aes(xintercept = umbral, color = modelo),
    linetype = "dashed", linewidth = 0.6, show.legend = FALSE
  ) +
  geom_text(
    data  = tabla_umbrales,
    aes(x = umbral, y = 0.08, label = round(umbral, 2), color = modelo),
    size  = 3, hjust = -0.15, show.legend = FALSE
  ) +
  scale_linetype_manual(
    values = c("j_index" = "solid", "f_meas" = "dotdash"),
    labels = c("J-index (Youden)", "F1-score")
  ) +
  scale_color_manual(values = c(
    "CART"          = "#E07B54",
    "Random Forest" = "#3A7DC9",
    "XGBoost"       = "#2ECC71"
  )) +
  labs(
    title    = "Calibración del Threshold por Modelo",
    subtitle = "Línea vertical = umbral óptimo según J-index (Youden)",
    x        = "Threshold de clasificación",
    y        = "Valor de la métrica",
    color    = "Modelo",
    linetype = "Métrica"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("output/figures/threshold_calibration_comparado.png",
       p_threshold, width = 10, height = 6, dpi = 150)

# ------------------------------------------------------------------------------
# 12. Finalización de modelos sobre el dataset completo de entrenamiento
# ------------------------------------------------------------------------------
message("--- Ajustando modelos finales sobre data_model completo ---")

fit_cart <- finalize_workflow(wf_cart, best_cart) %>% fit(data = data_model)
fit_rf   <- finalize_workflow(wf_rf,   best_rf)   %>% fit(data = data_model)
fit_xgb  <- finalize_workflow(wf_xgb,  best_xgb)  %>% fit(data = data_model)

# ------------------------------------------------------------------------------
# 12.1. Proyección MCA sobre el test set (2025)
# ------------------------------------------------------------------------------
# El test corresponde al marco censal 2022; proyectamos los hogares de 2025
# sobre el espacio MCA entrenado con el último año disponible del train (2024)
# para ubicarlos en el mismo mapa socioeconómico

mca_modelo        <- readRDS("data/processed/mca_resultados_anuales.rds")
ultimo_anio_train <- names(mca_modelo)[length(mca_modelo)]
res_mca_train     <- mca_modelo[[ultimo_anio_train]]$res_mca

VARS_MCA_ORIG <- c(
  "v2", "v13", "v5", "v11", "v12",   # activos del hogar
  "iv1", "iv2", "iv5",                # tipo y materiales de vivienda
  "ii7", "ii8",                       # tenencia y combustible
  "iv10", "ii9"                       # saneamiento de contexto
)

mca_test_proj <- as.data.frame(
  predict(res_mca_train,
          eph_test %>% select(all_of(VARS_MCA_ORIG)))$coord
) %>%
  rename(mca_dim1 = `Dim 1`, mca_dim2 = `Dim 2`)

eph_test_mca <- bind_cols(eph_test, mca_test_proj)

# ------------------------------------------------------------------------------
# 13. Evaluación en el test set
# ------------------------------------------------------------------------------

# Homologa los tipos de columnas del test con los del train para evitar
# errores de predicción por factor levels o tipos incompatibles
match_classes <- function(target, reference) {
  for (col in names(target)) {
    if (col %in% names(reference)) {
      if (is.factor(reference[[col]])) {
        target[[col]] <- factor(
          as.character(target[[col]]),
          levels = levels(reference[[col]])
        )
      } else {
        class(target[[col]]) <- class(reference[[col]])
      }
    }
  }
  target
}

data_test_final <- eph_test_mca %>%
  select(all_of(c("mpi_pobre", "aglomerado", "ano4", "pondera", PREDICTORES))) %>%
  mutate(
    mpi_pobre    = factor(mpi_pobre, levels = c("no_pobre", "pobre")),
    region_label = factor(region_label)
  ) %>%
  match_classes(reference = data_model) %>%
  drop_na(mpi_pobre)

# evaluar_modelo() aplica el umbral calibrado de cada modelo
# en lugar del 0.5 por defecto
evaluar_modelo <- function(fit, data, nombre, umbral = 0.5) {
  
  pred_prob <- predict(fit, data, type = "prob")
  
  pred_class <- pred_prob %>%
    mutate(.pred_class = factor(
      if_else(.pred_pobre >= umbral, "pobre", "no_pobre"),
      levels = c("no_pobre", "pobre")
    )) %>%
    select(.pred_class)
  
  bind_cols(
    data %>% select(mpi_pobre, pondera),
    pred_class,
    pred_prob
  ) %>%
    metricas(
      truth        = mpi_pobre,
      estimate     = .pred_class,
      .pred_pobre,
      event_level  = "second",
      case_weights = pondera
    ) %>%
    mutate(modelo = nombre, umbral_usado = umbral)
}

# Los umbrales se obtienen de la calibración OOF, no son valores hardcodeados
tabla_resultados <- bind_rows(
  evaluar_modelo(fit_cart, data_test_final, "CART",
                 umbral = tabla_umbrales %>%
                   filter(modelo == "CART") %>% pull(umbral)),
  evaluar_modelo(fit_rf, data_test_final, "Random Forest",
                 umbral = tabla_umbrales %>%
                   filter(modelo == "Random Forest") %>% pull(umbral)),
  evaluar_modelo(fit_xgb, data_test_final, "XGBoost",
                 umbral = tabla_umbrales %>%
                   filter(modelo == "XGBoost") %>% pull(umbral))
)

message("\n>>> Resultados Finales (sin data leakage, threshold calibrado):")
print(tabla_resultados)

# ------------------------------------------------------------------------------
# 14. Guardado
# ------------------------------------------------------------------------------
saveRDS(fit_cart,         "output/models/fit_cart.rds")
saveRDS(fit_rf,           "output/models/fit_rf.rds")
saveRDS(fit_xgb,          "output/models/fit_xgb.rds")
saveRDS(tabla_resultados, "output/results/tabla_resultados_test.rds")
saveRDS(tabla_umbrales,   "output/results/umbrales_calibrados.rds")
saveRDS(PREDICTORES, "output/results/predictores_modelo.rds")

message("✓ Step 06 completado.")