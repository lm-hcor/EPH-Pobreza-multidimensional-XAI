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
#   7. Se aumentó a 0.7 la sample de entrenamiento.
#   8. Se incluyeron las vars MCA directas para complementar el análisis.
#   9. Se incluyó un placebo test sobre el 30% no entrenado del conjunto 2016-2024
#      para medir el impacto del shift censal 2010 vs 2022.
#   10. Se reponderó el train ajustándose a la distribución marginal de los pesos
#       del test para intentar limitar el impacto del shift censal. 
#   11. Se evaluaron los modelos en base a la métrica F1. Debatible sobre hacerlo
#       con Youden J-Index, tal vez más exacto para public-policy.
#   12. Se aumentó el range de arboles en el grid del RF para evitar overfitting.
#   13. Se añadió un bloque de trazabilidad de experimentos para guardar futuros 
#       cambios en métricas sin perder los modelos.
# ==============================================================================
# ------------------------------------------------------------------------------
# TRAZABILIDAD DE EXPERIMENTOS
# ------------------------------------------------------------------------------
# Cada ejecución crea una carpeta única con timestamp para no sobrescribir
# resultados anteriores. Permite comparar experimentos y revertir si es necesario.

experimento_id <- format(Sys.time(), "%Y%m%d_%H%M%S")
experimento_dir <- file.path("output", "experimentos", experimento_id)

dir.create(file.path(experimento_dir, "models"),  recursive = TRUE)
dir.create(file.path(experimento_dir, "results"), recursive = TRUE)
dir.create(file.path(experimento_dir, "figures"), recursive = TRUE)

# Guardar configuración del experimento para reproducibilidad
config_experimento <- list(
  timestamp     = experimento_id,
  prop_sample   = 0.7,
  grid_cart     = 15,
  grid_rf       = 15,
  grid_xgb      = 18,
  over_ratio    = 0.5,
  min_n_rf      = "50-200",
  predictores   = PREDICTORES,
  n_predictores = length(PREDICTORES),
  n_train       = nrow(data_model),
  seed          = 42
)

saveRDS(config_experimento,
        file.path(experimento_dir, "results", "config.rds"))

# Log de texto legible para revisión rápida
writeLines(
  c(
    paste("Experimento:", experimento_id),
    paste("Fecha:", Sys.time()),
    paste("Predictores:", length(PREDICTORES)),
    paste("N train:", nrow(data_model)),
    paste("Grid RF size:", 15),
    paste("Grid XGB size:", 18),
    paste("over_ratio SMOTENC:", 0.5),
    paste("min_n RF range:", "50-200"),
    paste("prop sample:", 0.7)
  ),
  file.path(experimento_dir, "results", "config.txt")
)

message(">>> Experimento ID: ", experimento_id)
message(">>> Resultados en: ", experimento_dir)

# ------------------------------------------------------------------------------
# LIBRERIAS
# ------------------------------------------------------------------------------
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
# Variables MCA originales como dummies directas
# Complementan mca_dim1/mca_dim2 con información categórica no capturada
# por las dos primeras dimensiones factoriales
VARS_MCA_DIRECTAS <- c(
  "v2",   # heladera
  "v13",  # auto/camioneta
  "v11",  # internet
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

# ------------------------------------------------------------------------------
# 3. Preparación del dataset de modelado
# ------------------------------------------------------------------------------
# Limpieza preventiva: si existe adeq_hogar como df en el entorno global
# puede colisionar con la columna del mismo nombre en el dataframe
if (exists("adeq_hogar") && is.data.frame(get("adeq_hogar"))) {
  rm(adeq_hogar, envir = .GlobalEnv)
}

data_model <- eph_train %>%
  select(mpi_pobre, codusu, aglomerado, ano4,trimestre,
         pondera, all_of(PREDICTORES)) %>%
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
  slice_sample(prop = 0.7, weight_by = pondera) # aumentar a 0.75?

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
# 3.1 Reponderación del train para aproximar el marco censal 2022
# ------------------------------------------------------------------------------
# El test set usa el marco 2022 mientras el train usa el marco 2010.
# La reponderación ajusta los pesos del train para que su distribución
# demográfica se aproxime a la del test, reduciendo el shift censal.
# Referencia: Little & Rubin (2002), Statistical Analysis with Missing Data.
# ------------------------------------------------------------------------------

# Distribución objetivo: estructura demográfica del test (marco 2022)
dist_test <- eph_test %>%
  group_by(region_label) %>%
  summarise(
    prop_test    = sum(pondera) / sum(eph_test$pondera),
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
      prop_test / prop_train
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
  
  # 1. Imputación previa a la creación de features derivadas
  #    (evita NAs propagados en las divisiones)
  step_impute_median(itcf_real, adeq_hogar, p21_real) %>%
  
  # 2. Creación de features derivadas
  step_mutate(
    # Ingreso ajustado por tamaño del hogar según adulto equivalente
    ingreso_per_capita = itcf_real / pmax(adeq_hogar, 1),
    # Tendencia temporal normalizada [0, 1]: 2016 → 0, 2024 → 1
    año_norm           = (ano4 - 2016) / (2024 - 2016), # feature adicional a las temporales. Si hay multicolinearidad la recipe elimina.
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
  themis::step_smotenc(mpi_pobre, over_ratio = 0.5, neighbors = 5) # bajar over_ratio a 0.5?

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
  size = 15
)

grid_rf <- grid_space_filling(
  mtry(range  = c(2L, 8L)),
  trees(range = c(200L, 800L)),
  min_n(range = c(50L, 200L)), #reduce overfitting.
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
# 11.1 Calibración del threshold (ANTES de fit final, usa predicciones OOF)
# ------------------------------------------------------------------------------
# Las predicciones OOF provienen del tuning: el modelo nunca vio esos datos
# durante el ajuste del fold correspondiente → calibración válida sin tocar
# el test set

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
    filter(.metric == "f_meas") %>% # cambio de j_index a f_meas.
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

# Tabla de umbrales óptimos separada (sin join problemático)
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
  # CORRECCIÓN: geom_text usa tabla_umbrales_plot directamente
  # sin depender de .metric que no existe en ese dataframe
  geom_text(
    data = tabla_umbrales_plot,
    aes(x = umbral, y = 0.08,
        label = round(umbral, 2),
        color = modelo),
    size = 3, hjust = -0.15,
    show.legend = FALSE,
    inherit.aes = FALSE   # ← clave: no hereda aes del ggplot principal
  ) +
  scale_linetype_manual(
    values = c("f_meas" = "solid", "j_index" = "dashed"), # F1 ahora es la principal
    labels = c("F1-score (Optimizado)", "J-index (Youden)")
  ) +
  scale_color_manual(values = c(
    "CART"          = "#E07B54",
    "Random Forest" = "#3A7DC9",
    "XGBoost"       = "#2ECC71"
  )) +
  labs(
    title    = "Calibración del Threshold por Modelo",
    subtitle = "Línea vertical = umbral óptimo según F1-score (Maximiza Kapppa)",
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
# predict.MCA() falla internamente con match.arg() cuando hay factor levels
# no vistos, incluso tras alinearlos. La alternativa robusta es proyectar
# manualmente: codificar el test como tabla disyuntiva completa (dummies 0/1)
# y multiplicar por la matriz de coordenadas de columnas del MCA.
# ------------------------------------------------------------------------------

VARS_MCA_ORIG <- c(
  "v2", "v13", "v5", "v11", "v12",
  "iv1", "iv2", "iv5",
  "ii7", "ii8", "iv10", "ii9"
)

# --- Paso 1: Entrenar MCA sobre años recientes del train (2021-2024) ----------
df_mca_reciente <- eph_train %>%
  filter(ano4 >= 2021) %>%
  select(all_of(VARS_MCA_ORIG)) %>%
  mutate(across(everything(), ~ factor(as.character(.)))) %>%
  mutate(across(everything(),
                ~ fct_lump_prop(., prop = 0.02, other_level = "Otros")))

message(">>> Entrenando MCA 2021-2024...")
res_mca_reciente <- MCA(df_mca_reciente, ncp = 2, graph = FALSE)
message("  MCA 2021-2024 completado.")

# --- Paso 2: Extraer componentes del MCA necesarios para proyección manual ----
# v  = coordenadas de columnas (categorías) en el espacio factorial
# Dr = masa de cada columna (frecuencia relativa)
# Ambas vienen de res_mca$var$coord y res_mca$call
v_coord <- res_mca_reciente$var$coord          # matriz (n_categorias x ncp)
col_names_mca <- rownames(v_coord)             # "v2_1", "v2_2", ..., etc.
ncp <- ncol(v_coord)                           # número de dimensiones (2)

# --- Paso 3: Alinear el test al espacio de categorías conocidas --------------
# Construir tabla disyuntiva del test con exactamente las mismas columnas
# que el MCA conoce. Categorías desconocidas → "Otros".
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

# Verificar NAs residuales (casos donde ni "Otros" existe en el nivel)
n_na_test <- sum(is.na(df_test_alineado))
message("  NAs en test alineado: ", n_na_test)

if (n_na_test > 0) {
  # Imputar con la moda de cada columna
  df_test_alineado <- df_test_alineado %>%
    mutate(across(everything(), ~ {
      moda <- names(sort(table(.), decreasing = TRUE))[1]
      factor(if_else(is.na(.), moda, as.character(.)), levels = levels(.))
    }))
  message("  NAs imputados con moda.")
}

# --- Paso 4: Construir tabla disyuntiva completa (dummies 0/1) ---------------
# Cada fila = un hogar; cada columna = una categoría del MCA (ej: "v2_1")
# El nombre de columna sigue el formato "variable_categoria" del MCA

construir_disjuntiva <- function(df, niveles_ref) {
  result <- vector("list", length(niveles_ref))
  names(result) <- names(niveles_ref)
  
  for (col in names(niveles_ref)) {
    nivs <- niveles_ref[[col]]
    mat  <- model.matrix(~ 0 + ., data = data.frame(x = df[[col]]))
    # model.matrix genera "xNIVEL", renombrar a "COL_NIVEL" para que coincida
    colnames(mat) <- paste0(col, "_", nivs[match(
      sub("^x", "", colnames(mat)), nivs
    )])
    result[[col]] <- mat
  }
  do.call(cbind, result)
}

Z_test <- construir_disjuntiva(df_test_alineado, niveles_mca)

# Alinear columnas de Z_test exactamente con las del MCA
# (mismo orden, mismas columnas; faltantes → 0, extras → descartadas)
cols_mca   <- col_names_mca
cols_test  <- colnames(Z_test)
cols_falt  <- setdiff(cols_mca, cols_test)
cols_sobre <- setdiff(cols_test, cols_mca)

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

Z_test <- Z_test[, cols_mca, drop = FALSE]   # orden canónico del MCA

# --- Paso 5: Proyección — fórmula de transición suplementaria ----------------
# coord_test = (1/Q) * Z_test_norm * v_coord
# donde Q = número de variables, Z_test_norm = tabla de frecuencias relativas
# por fila (cada fila suma 1).
# Equivalente a: para cada hogar, promediar las coordenadas de columna de sus
# categorías activas, ponderado por la frecuencia relativa de cada variable.

Q       <- length(VARS_MCA_ORIG)   # número de variables del MCA
n_test  <- nrow(Z_test)

# Frecuencia relativa por fila (para tabla disyuntiva, cada fila suma Q,
# por eso dividimos por Q para normalizar a [0,1] por hogar)
Z_norm  <- Z_test / Q

# Proyección: hogar × categorías · categorías × dimensiones
coords_test <- Z_norm %*% v_coord    # resultado: n_test × ncp

mca_test_proj <- as.data.frame(coords_test) %>%
  setNames(paste0("mca_dim", seq_len(ncp))) %>%
  mutate(across(everything(), ~ if_else(is.na(.), 0, .)))

n_na_proj <- sum(is.na(mca_test_proj))
message("  NAs en proyección MCA: ", n_na_proj,
        " (reemplazados por centroide 0 si existen)")

eph_test_mca <- bind_cols(eph_test, mca_test_proj)
message("  Proyección completada. Dimensiones test_mca: ",
        nrow(eph_test_mca), " × ", ncol(eph_test_mca))
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
  mutate(grupo_cv = paste0(codusu, "_", aglomerado)) %>%
  select(all_of(c("mpi_pobre", "codusu", "grupo_cv",
                  "aglomerado", "ano4", "pondera", PREDICTORES))) %>%
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
# Chequeo sobre el impacto del cambio censal. 
# Evaluación sobre muestra aleatoria del train
# Si aquí el ROC-AUC es mucho más alto, el problema es el shift censal
set.seed(42)
data_train_eval <- data_model %>%
  slice_sample(n = 10000)

tabla_train <- bind_rows(
  evaluar_modelo(fit_cart, data_train_eval, "CART",
                 umbral = tabla_umbrales %>%
                   filter(modelo == "CART") %>% pull(umbral)),
  evaluar_modelo(fit_rf, data_train_eval, "Random Forest",
                 umbral = tabla_umbrales %>%
                   filter(modelo == "Random Forest") %>% pull(umbral)),
  evaluar_modelo(fit_xgb, data_train_eval, "XGBoost",
                 umbral = tabla_umbrales %>%
                   filter(modelo == "XGBoost") %>% pull(umbral))
)

message("Métricas sobre train (referencia):")
print(tabla_train)

# Placebo Test para medir impacto de shift censal.
# El train usó prop = 0.7 con set.seed(42)
# Recrear exactamente qué hogares entraron y cuáles no
# El train usó slice_sample(prop = 0.7, weight_by = pondera, seed = 42)
# sobre el dataset completo. Recreamos exactamente qué hogares-trimestre
# entraron al train y usamos los excluidos de 2024 como placebo.
# Estos hogares son del marco censal 2010 (igual que el train) pero
# el modelo nunca los vio → evaluación limpia sin leakage temporal.
# ==============================================================================

# Paso 1: Recrear el dataset completo PRE-slice (mismo proceso que en el script)
# con la misma limpieza y variables para poder hacer anti_join por codusu+trimestre
data_full_premuestra <- eph_train %>%
  select(mpi_pobre, codusu, aglomerado, ano4, trimestre,
         pondera, all_of(PREDICTORES)) %>%
  mutate(
    mpi_pobre    = factor(mpi_pobre, levels = c("no_pobre", "pobre")),
    region_label = factor(region_label),
    aglomerado   = as.character(aglomerado),
    across(c(itcf_real, adeq_hogar, p21_real), as.numeric)
  ) %>%
  drop_na(mpi_pobre) %>%
  mutate(grupo_cv = paste0(codusu, "_", aglomerado))

message("Filas totales pre-muestra: ", nrow(data_full_premuestra))

# Paso 2: Recrear exactamente el mismo slice_sample con set.seed(42)
# Esto reproduce exactamente qué 70% entró al train
set.seed(42)
ids_en_train <- data_full_premuestra %>%
  slice_sample(prop = 0.7, weight_by = pondera) %>%
  # Llave única: hogar + trimestre + año
  mutate(id_obs = paste0(codusu, "_", ano4, "_", trimestre)) %>%
  pull(id_obs)

message("Observaciones en train: ", length(ids_en_train))

# Paso 3: Identificar el 30% excluido de 2024
# Son hogares del marco 2010, nunca vistos por el modelo
data_2024_holdout <- data_full_premuestra %>%
  filter(ano4 == 2024) %>%
  mutate(id_obs = paste0(codusu, "_", ano4, "_", trimestre)) %>%
  filter(!id_obs %in% ids_en_train) %>%
  select(-id_obs)

message("Hogares 2024 en holdout (30% excluido): ", nrow(data_2024_holdout))
message("Distribución target en holdout:")
print(table(data_2024_holdout$mpi_pobre))

# Verificación: no debe haber solapamiento con el train
n_solapamiento <- data_2024_holdout %>%
  mutate(id_obs = paste0(codusu, "_", ano4, "_", trimestre)) %>%
  filter(id_obs %in% ids_en_train) %>%
  nrow()

message("Solapamiento con train: ", n_solapamiento,
        " (debe ser 0)")

# Paso 4: Preparar para evaluación
data_2024_placebo_final <- data_2024_holdout %>%
  mutate(
    region_label = factor(region_label,
                          levels = levels(data_model$region_label))
  ) %>%
  match_classes(reference = data_model) %>%
  drop_na(mpi_pobre)

message("Filas en placebo final: ", nrow(data_2024_placebo_final))

# Paso 5: Evaluación
tabla_placebo_2024 <- bind_rows(
  evaluar_modelo(fit_rf,  data_2024_placebo_final,
                 "RF — Placebo 2024 (30% holdout)",
                 umbral = tabla_umbrales %>%
                   filter(modelo == "Random Forest") %>% pull(umbral)),
  evaluar_modelo(fit_xgb, data_2024_placebo_final,
                 "XGBoost — Placebo 2024 (30% holdout)",
                 umbral = tabla_umbrales %>%
                   filter(modelo == "XGBoost") %>% pull(umbral))
)

message("\n>>> Placebo test — 30% holdout 2024 (marco censal 2010):")
print(tabla_placebo_2024)

# Paso 6: Tabla comparativa completa para el TFM
message("\n>>> Comparativa: Placebo 2024 vs Test 2025")
bind_rows(
  tabla_placebo_2024 %>%
    mutate(
      # Estandarizar nombres para que coincidan con tabla_resultados
      modelo = case_when(
        str_detect(modelo, "Random Forest|RF") ~ "Random Forest",
        str_detect(modelo, "XGBoost")          ~ "XGBoost",
        TRUE                                   ~ modelo
      ),
      conjunto = "Placebo 2024 (marco 2010)"
    ),
  tabla_resultados %>%
    filter(modelo %in% c("Random Forest", "XGBoost")) %>%
    mutate(conjunto = "Test 2025 (marco 2022)")
) %>%
  filter(.metric == "roc_auc") %>%
  select(modelo, conjunto, .estimate) %>%
  pivot_wider(names_from = conjunto, values_from = .estimate) %>%
  mutate(
    diferencia_censal = round(
      `Placebo 2024 (marco 2010)` - `Test 2025 (marco 2022)`, 3
    )
  ) %>%
  arrange(modelo) %>%
  print()
# ------------------------------------------------------------------------------
# 14. Guardado
# ------------------------------------------------------------------------------

# En carpeta de experimento. 
saveRDS(fit_cart,         file.path(experimento_dir, "models",  "fit_cart.rds"))
saveRDS(fit_rf,           file.path(experimento_dir, "models",  "fit_rf.rds"))
saveRDS(fit_xgb,          file.path(experimento_dir, "models",  "fit_xgb.rds"))
saveRDS(tabla_resultados, file.path(experimento_dir, "results", "tabla_resultados_test.rds"))
saveRDS(tabla_umbrales,   file.path(experimento_dir, "results", "umbrales_calibrados.rds"))
saveRDS(PREDICTORES,      file.path(experimento_dir, "results", "predictores_modelo.rds"))
saveRDS(config_experimento, file.path(experimento_dir, "results", "config.rds"))

# En carpeta de proyecto.
saveRDS(fit_cart,         "output/models/fit_cart.rds")
saveRDS(fit_rf,           "output/models/fit_rf.rds")
saveRDS(fit_xgb,          "output/models/fit_xgb.rds")
saveRDS(tabla_resultados, "output/results/tabla_resultados_test.rds")
saveRDS(tabla_umbrales,   "output/results/umbrales_calibrados.rds")
saveRDS(PREDICTORES, "output/results/predictores_modelo.rds")
saveRDS(tabla_placebo_2024,
        "output/results/placebo_test_2024_holdout.rds")

message("✓ Step 06 completado.")
message("  Experimento guardado en: ", experimento_dir)