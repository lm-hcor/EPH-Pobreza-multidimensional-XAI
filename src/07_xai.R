# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 07_xai.R
# Propósito: Capa de interpretabilidad XAI con DALEX
#            SHAP global, SHAP regional, PDP y Break-down para 3 hogares tipo
#
# ACTUALIZACIÓN 2024-v2:
#   - Train: 2016-2023, Test: 2024, Externo: 2025
#   - PREDICTORES cargados desde archivo guardado en 06_modelling.R
#   - Diccionario de etiquetas: variables técnicas sustituidas por nombres
#     descriptivos en todos los gráficos
#   - region_label corregida con etiquetas canónicas antes de crear explainers
#   - Break-down sobre el test set 2024 con MCA proyectado
#   - Tres hogares tipo seleccionados con criterio claro:
#     * Mínima probabilidad predicha (no_pobre típico)
#     * Máxima probabilidad predicha (pobre típico)
#     * Frontera de decisión (|prob - umbral| mínimo)
#   - SHAP regional con etiquetas de región correctas
#   - Guardado de todos los objetos para uso en 08_graphs.R
#
# NOTA METODOLÓGICA:
#   SHAP (Shapley Additive Explanations) atribuye la predicción de cada hogar
#   a sus variables individuales usando teoría de juegos cooperativos.
#   Los valores SHAP suman exactamente la diferencia entre la predicción del
#   hogar y la predicción media del modelo (baseline).
#   Break-down descompone secuencialmente la predicción de un hogar específico,
#   mostrando cómo cada variable modifica la probabilidad respecto al baseline.
#   Referencia: Biecek & Burzykowski (2021); Lundberg & Lee (2017).
# ==============================================================================

library(tidyverse)
library(DALEX)
library(DALEXtra)
library(ingredients)
library(iBreakDown)
library(patchwork)
library(janitor)

message(">>> Iniciando Step 07: Análisis XAI con DALEX...")

# ==============================================================================
# 1. CARGA DE DATOS Y MODELOS
# ==============================================================================
message(">>> Cargando modelos y datos...")

fit_cart <- readRDS("output/models/fit_cart.rds")
fit_rf <- readRDS("output/models/fit_rf.rds")
fit_xgb <- readRDS("output/models/fit_xgb.rds")

# PREDICTORES cargados desde 06_modelling.R para garantizar coherencia exacta
PREDICTORES <- readRDS("output/results/predictores_modelo.rds")
tabla_umbrales <- readRDS("output/results/umbrales_calibrados.rds")

# Cargar datasets procesados
# eph_train: 2016-2023 (con proyección MCA ya aplicada)
# eph_test: 2024 (con proyección MCA ya aplicada)
# eph_externo: 2025 (con proyección MCA ya aplicada)
eph_train <- readRDS("data/processed/eph_train_mca.rds")
eph_test <- readRDS("data/processed/eph_test_ml.rds")
eph_externo <- readRDS("data/processed/eph_externo_ml.rds")

# Umbral calibrado del XGBoost (modelo principal para XAI)
umbral_xgb <- tabla_umbrales %>%
  filter(modelo == "XGBoost") %>%
  pull(umbral)

umbral_rf <- tabla_umbrales %>%
  filter(modelo == "Random Forest") %>%
  pull(umbral)

message("  PREDICTORES activos: ", length(PREDICTORES))
message("  Umbral XGBoost: ", round(umbral_xgb, 3))
message("  Umbral RF: ", round(umbral_rf, 3))

# ==============================================================================
# 2. DICCIONARIO DE ETIQUETAS
# ==============================================================================
# Sustituye nombres técnicos de la EPH por etiquetas descriptivas en los
# gráficos. Se aplica a todos los objetos DALEX mediante el parámetro label.
# ------------------------------------------------------------------------------
ETIQUETAS_VARS <- c(
  # Económicas
  "itcf_real" = "Ingreso familiar real",
  "p21_real" = "Ingreso ocup. principal",
  "ingreso_per_capita" = "Ingreso per cápita",
  
  # Educación
  "max_instruccion" = "Máx. instrucción hogar",
  
  # Demografía
  "tamano_hogar" = "Tamaño del hogar",
  "n_menores" = "N.º menores (<18)",
  "n_ancianos" = "N.º ancianos (≥65)",
  "adeq_hogar" = "Adulto equivalente",
  "ratio_dependencia" = "Ratio de dependencia",
  "carga_demo" = "Carga demográfica",
  
  # Mercado de trabajo
  "n_ocupados" = "N.º ocupados",
  "prop_informal" = "Prop. informalidad lab.",
  
  # Salud
  "priv_salud" = "Sin cobertura de salud",
  
  # Temporal
  "ano4" = "Año",
  "año_norm" = "Tendencia temporal",
  "trimestre" = "Trimestre",
  
  # Geografía
  "region_label" = "Región",
  
  # MCA — contexto socioeconómico
  "mca_dim1" = "MCA Dim.1 (Bienestar)",
  "mca_dim2" = "MCA Dim.2 (Infraestr.)",
  
  # Variables MCA originales (activos y vivienda)
  "v2" = "Vivió de jubilación",
  "v11" = "Vivió de Beca",
  "v13" = "Gastó ahorros",
  "iv1" = "Tipo de vivienda",
  "iv2" = "Material paredes",
  "ii7" = "Régimen de tenencia",
  "ii8" = "Combustible cocinar",
  "iv5" = "Revestimiento techo",
  "iv7" = "Ubicación agua",
  "iv8" = "Tiene baño",
  "iv10" = "Descarga del baño",
  "ii9" = "Ubicación del baño"
)

# Función: renombrar columnas de un dataframe según el diccionario
renombrar_con_etiquetas <- function(df, diccionario = ETIQUETAS_VARS) {
  rename_vec <- diccionario[names(diccionario) %in% names(df)]
  if (length(rename_vec) > 0) {
    df <- df %>% rename(!!!setNames(names(rename_vec), rename_vec))
  }
  df
}

# ==============================================================================
# 3. PREPARACIÓN DEL DATASET XAI
# ==============================================================================
message(">>> Preparando dataset XAI...")

# Corrección de region_label: garantizar etiquetas canónicas
ETIQUETAS_REGION <- c(
  "1" = "GBA",
  "2" = "Pampeana",
  "3" = "Noroeste",
  "4" = "Nordeste",
  "5" = "Cuyo",
  "6" = "Patagonia",
  # Por si ya vienen con etiqueta
  "GBA" = "GBA",
  "Pampeana" = "Pampeana",
  "Noroeste" = "Noroeste",
  "Nordeste" = "Nordeste",
  "Cuyo" = "Cuyo",
  "Patagonia" = "Patagonia"
)

# Función para crear dummies de región (igual que en 06_modelling.R)
crear_dummies_region <- function(df) {
  regiones <- c("Cuyo", "GBA", "Nordeste", "Noroeste", "Pampeana", "Patagonia")
  for (r in regiones) {
    df[[paste0("region_label_", r)]] <-
      as.integer(as.character(df$region_label) == r)
  }
  df %>% select(-region_label)
}

# Primero preparamos con region_label, luego creamos dummies
set.seed(42)
data_xai_prep <- eph_train %>%
  select(all_of(c("mpi_pobre", "region_label", "pondera", PREDICTORES))) %>%
  mutate(
    mpi_pobre = factor(mpi_pobre, levels = c("no_pobre", "pobre")),
    region_label = ETIQUETAS_REGION[as.character(region_label)],
    region_label = factor(region_label,
                          levels = c("GBA", "Pampeana", "Noroeste", "Nordeste", "Cuyo", "Patagonia")
    )
  ) %>%
  drop_na(mpi_pobre, region_label)

# Guardar info de regiones
regiones_info <- unique(data_xai_prep$region_label)

# Crear versión con dummies para SHAP global
data_xai <- data_xai_prep %>%
  crear_dummies_region() %>%
  slice_sample(n = min(5000, nrow(.)), weight_by = pondera)

# Crear versión con dummies y region_label para SHAP regional (sin muestrear)
data_xai_regional <- data_xai_prep %>%
  crear_dummies_region()

message("  Hogares en data_xai: ", nrow(data_xai))
message(
  "  Regiones representadas: ",
  paste(as.character(regiones_info), collapse = ", ")
)
message("  Distribución target:")
print(table(data_xai$mpi_pobre))

# Dataset con etiquetas descriptivas para los gráficos
# Nota: PREDICTORES incluye region_label, pero data_xai ya tiene dummies en su lugar
# Usamos los nombres de columnas que realmente existen en data_xai
predictores_sin_region <- setdiff(PREDICTORES, "region_label")
predictores_con_dummies <- c(
  predictores_sin_region,
  paste0("region_label_", c("Cuyo", "GBA", "Nordeste", "Noroeste", "Pampeana", "Patagonia"))
)
predictores_para_labels <- intersect(predictores_con_dummies, names(data_xai))

data_xai_labels <- data_xai %>%
  select(all_of(predictores_para_labels)) %>%
  renombrar_con_etiquetas()

# ==============================================================================
# 4. PREPARACIÓN DEL TEST SET 2024 PARA BREAK-DOWN
# ==============================================================================
# El test set ya tiene proyección MCA aplicada desde 06_modelling.R
# Solo necesitamos preparar el dataset con las columnas correctas
# ------------------------------------------------------------------------------
message(">>> Preparando test set 2024 para Break-down...")

# Función para homologar clases entre datasets
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

# Preparar test final para predicción
# El test ya tiene region_label como factor, pero PREDICTORES incluye region_label
# que debe convertirse en dummies para ser consistente con el train
hogar_test_xai_prep <- eph_test %>%
  mutate(
    grupo_cv = paste0(codusu, "_", aglomerado),
    region_label = factor(ETIQUETAS_REGION[as.character(region_label)],
                          levels = c("GBA", "Pampeana", "Noroeste", "Nordeste", "Cuyo", "Patagonia")
    )
  ) %>%
  select(all_of(c("mpi_pobre", "codusu", "grupo_cv", "aglomerado", "ano4", "pondera", PREDICTORES))) %>%
  mutate(mpi_pobre = factor(mpi_pobre, levels = c("no_pobre", "pobre"))) %>%
  drop_na(mpi_pobre)

# Convertir region_label en dummies (igual que en train)
hogar_test_xai <- hogar_test_xai_prep %>%
  crear_dummies_region()

message("  Test set preparado: ", nrow(hogar_test_xai), " hogares")

# ==============================================================================
# 5. EXPLAINERS DALEX
# ==============================================================================
message(">>> Creando explainers DALEX...")

# Función de predicción universal para tidymodels workflows
predict_fn <- function(model, newdata) {
  n <- nrow(newdata)
  
  # Añadir columnas auxiliares si faltan (el workflow las espera)
  if (!"codusu" %in% names(newdata)) {
    newdata$codusu <- as.character(seq_len(n))
  } else {
    newdata$codusu <- as.character(newdata$codusu)
  }
  
  if (!"grupo_cv" %in% names(newdata)) {
    newdata$grupo_cv <- paste0("g_", seq_len(n))
  } else {
    newdata$grupo_cv <- as.character(newdata$grupo_cv)
  }
  
  # Orden estable
  newdata <- newdata %>%
    relocate(codusu, grupo_cv)
  
  predict(
    model,
    new_data = newdata,
    type = "prob"
  )$.pred_pobre
}

# Dataset con etiquetas para los explainers
# PREDICTORES incluye region_label, pero data_xai ya tiene dummies
# Usamos las columnas que realmente existen en data_xai
predictores_sin_region <- setdiff(PREDICTORES, "region_label")
predictores_con_dummies <- c(
  predictores_sin_region,
  paste0("region_label_", c("Cuyo", "GBA", "Nordeste", "Noroeste", "Pampeana", "Patagonia"))
)
predictores_para_pred <- intersect(predictores_con_dummies, names(data_xai))

data_xai_pred <- data_xai %>%
  mutate(
    codusu = seq_len(n()),
    grupo_cv = paste0("g_", seq_len(n()))
  ) %>%
  select(codusu, grupo_cv, all_of(predictores_para_pred))

explainer_cart <- explain_tidymodels(
  model            = fit_cart,
  data             = data_xai_pred,
  y                = as.numeric(data_xai$mpi_pobre == "pobre"),
  label            = "CART",
  predict_function = predict_fn,
  verbose          = FALSE
)

explainer_rf <- explain_tidymodels(
  model            = fit_rf,
  data             = data_xai_pred,
  y                = as.numeric(data_xai$mpi_pobre == "pobre"),
  label            = "Random Forest",
  predict_function = predict_fn,
  verbose          = FALSE
)

explainer_xgb <- explain_tidymodels(
  model            = fit_xgb,
  data             = data_xai_pred,
  y                = as.numeric(data_xai$mpi_pobre == "pobre"),
  label            = "XGBoost",
  predict_function = predict_fn,
  verbose          = FALSE
)

message("  Explainers creados: CART, Random Forest, XGBoost")

# ==============================================================================
# 6. SHAP GLOBAL: IMPORTANCIA DE VARIABLES (LOS TRES MODELOS)
# ==============================================================================
message(">>> Calculando SHAP global (los tres modelos)...")

calcular_shap_global <- function(explainer, n_sample = 500, B = 10) {
  model_parts(
    explainer,
    type = "variable_importance",
    B    = B,
    N    = n_sample
  )
}

shap_cart <- calcular_shap_global(explainer_cart)
shap_rf <- calcular_shap_global(explainer_rf)
shap_xgb <- calcular_shap_global(explainer_xgb)

# Función para aplicar etiquetas a objetos SHAP antes de graficar
aplicar_etiquetas_shap <- function(shap_obj) {
  shap_obj$variable <- ifelse(
    shap_obj$variable %in% names(ETIQUETAS_VARS),
    ETIQUETAS_VARS[shap_obj$variable],
    shap_obj$variable
  )
  shap_obj
}

shap_cart_lab <- aplicar_etiquetas_shap(shap_cart)
shap_rf_lab <- aplicar_etiquetas_shap(shap_rf)
shap_xgb_lab <- aplicar_etiquetas_shap(shap_xgb)

# Gráficos SHAP global con etiquetas
p_shap_cart <- plot(shap_cart_lab) +
  ggtitle("SHAP – CART",
          subtitle = "Importancia media |SHAP| por variable"
  ) +
  theme_minimal(base_size = 11)

p_shap_rf <- plot(shap_rf_lab) +
  ggtitle("SHAP – Random Forest",
          subtitle = "Importancia media |SHAP| por variable"
  ) +
  theme_minimal(base_size = 11)

p_shap_xgb <- plot(shap_xgb_lab) +
  ggtitle("SHAP – XGBoost",
          subtitle = "Importancia media |SHAP| por variable"
  ) +
  theme_minimal(base_size = 11)

p_shap_comparado <- p_shap_cart / p_shap_rf / p_shap_xgb +
  plot_annotation(
    title = "Importancia de Variables — Comparativa de Modelos",
    subtitle = "SHAP global (Shapley Additive Explanations)",
    caption = "Muestra: 500 hogares | B = 10 permutaciones"
  )

ggsave("output/figures/shap_global_comparado.png",
       p_shap_comparado,
       width = 12, height = 18, dpi = 150
)
message("  SHAP global guardado.")

# ==============================================================================
# 7. SHAP REGIONAL: XGBOOST POR REGIÓN (CONTRIBUCIÓN ORIGINAL)
# ==============================================================================
message(">>> Calculando SHAP regional (XGBoost)...")

regiones_canon <- c("GBA", "Pampeana", "Noroeste", "Nordeste", "Cuyo", "Patagonia")

shap_por_region <- map(regiones_canon, function(reg) {
  # Filtrar por región en data_xai_prep (que aún tiene region_label), luego crear dummies
  datos_reg <- data_xai_prep %>%
    filter(as.character(region_label) == reg) %>%
    crear_dummies_region()
  
  if (nrow(datos_reg) < 50) {
    message("  ⚠️  Región ", reg, " con < 50 obs — omitida")
    return(NULL)
  }
  
  message("  → Región: ", reg, " (n = ", nrow(datos_reg), ")")
  
  # PREDICTORES incluye region_label, pero datos_reg ya tiene dummies
  # Usamos predictores_para_pred que ya fue calculado antes
  explainer_reg <- explain_tidymodels(
    model            = fit_xgb,
    data             = datos_reg %>% select(all_of(predictores_para_pred)),
    y                = as.numeric(datos_reg$mpi_pobre == "pobre"),
    label            = paste0("XGBoost — ", reg),
    predict_function = predict_fn,
    verbose          = FALSE
  )
  
  shap <- model_parts(
    explainer_reg,
    type = "variable_importance",
    B    = 10,
    N    = min(300, nrow(datos_reg))
  )
  
  list(region = reg, shap = shap, n = nrow(datos_reg))
}) %>%
  compact() %>%
  set_names(map_chr(., ~ .x$region))

# Tabla de importancia regional (top 5 por región)
tabla_shap_regional <- map_df(shap_por_region, function(x) {
  x$shap %>%
    as_tibble() %>%
    filter(!variable %in% c("_full_model_", "_baseline_")) %>%
    mutate(
      variable_label = ifelse(
        variable %in% names(ETIQUETAS_VARS),
        ETIQUETAS_VARS[variable],
        variable
      )
    ) %>%
    slice_max(dropout_loss, n = 5, with_ties = FALSE) %>%
    transmute(
      region         = x$region,
      n_hogares      = x$n,
      variable       = variable,
      variable_label = variable_label,
      importancia    = dropout_loss
    )
})

message("\n  Top 3 variables por región (XGBoost):")

tabla_shap_regional %>%
  group_by(region) %>%
  slice_max(importancia, n = 3, with_ties = FALSE) %>%
  select(region, variable_label, importancia) %>%
  arrange(region, desc(importancia)) %>%
  print(n = 30)

# ==============================================================================
# 7b. SHAP REGIONAL: RANDOM FOREST POR REGIÓN
# ==============================================================================
message(">>> Calculando SHAP regional (Random Forest)...")

shap_por_region_rf <- map(regiones_canon, function(reg) {
  datos_reg <- data_xai_prep %>%
    filter(as.character(region_label) == reg) %>%
    crear_dummies_region()
  
  if (nrow(datos_reg) < 50) {
    message("  ⚠️  Región ", reg, " con < 50 obs — omitida")
    return(NULL)
  }
  
  message("  → Región: ", reg, " (n = ", nrow(datos_reg), ")")
  
  explainer_reg <- explain_tidymodels(
    model            = fit_rf,
    data             = datos_reg %>% select(all_of(predictores_para_pred)),
    y                = as.numeric(datos_reg$mpi_pobre == "pobre"),
    label            = paste0("RF — ", reg),
    predict_function = predict_fn,
    verbose          = FALSE
  )
  
  shap <- model_parts(
    explainer_reg,
    type = "variable_importance",
    B    = 10,
    N    = min(300, nrow(datos_reg))
  )
  
  list(region = reg, shap = shap, n = nrow(datos_reg))
}) %>%
  compact() %>%
  set_names(map_chr(., ~ .x$region))

# Tabla de importancia regional RF (top 5 por región)
tabla_shap_regional_rf <- map_df(shap_por_region_rf, function(x) {
  x$shap %>%
    as_tibble() %>%
    filter(!variable %in% c("_full_model_", "_baseline_")) %>%
    mutate(
      variable_label = ifelse(
        variable %in% names(ETIQUETAS_VARS),
        ETIQUETAS_VARS[variable],
        variable
      )
    ) %>%
    slice_max(dropout_loss, n = 5, with_ties = FALSE) %>%
    transmute(
      region         = x$region,
      n_hogares      = x$n,
      variable       = variable,
      variable_label = variable_label,
      importancia    = dropout_loss
    )
})

message("\n  Top 3 variables por región (Random Forest):")

tabla_shap_regional_rf %>%
  group_by(region) %>%
  slice_max(importancia, n = 3, with_ties = FALSE) %>%
  select(region, variable_label, importancia) %>%
  arrange(region, desc(importancia)) %>%
  print(n = 30)

# ==============================================================================
# 7c. HEATMAP: VARIABLES IMPORTANTES POR REGIÓN (XGBOOST vs RF)
# ==============================================================================
message(">>> Generando heatmap comparativo de variables por región...")

# Combinar tablas de ambos modelos
tabla_comparativa <- bind_rows(
  tabla_shap_regional %>% mutate(modelo = "XGBoost"),
  tabla_shap_regional_rf %>% mutate(modelo = "Random Forest")
)

# Top 10 variables más frecuentes en el top 5 regional
top_vars <- tabla_comparativa %>%
  group_by(variable_label) %>%
  summarise(frecuencia = n(), .groups = "drop") %>%
  arrange(desc(frecuencia)) %>%
  slice(1:10) %>%
  pull(variable_label)

# Crear matriz para heatmap: variable x región, coloreado por importancia
heatmap_data <- tabla_comparativa %>%
  filter(variable_label %in% top_vars) %>%
  group_by(modelo, region, variable_label) %>%
  summarise(importancia = mean(importancia, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    variable_label = factor(variable_label, levels = rev(top_vars)),
    region = factor(region, levels = regiones_canon)
  )

# Heatmap facetado por modelo
p_heatmap <- ggplot(heatmap_data, aes(x = region, y = variable_label, fill = importancia)) +
  geom_tile(color = "white", linewidth = 0.3) +
  geom_text(aes(label = round(importancia, 3)), size = 3, color = "black") +
  scale_fill_gradient2(
    low = "#3A7DC9", mid = "#FFFFFF", high = "#E07B54",
    midpoint = median(heatmap_data$importancia, na.rm = TRUE)
  ) +
  facet_wrap(~modelo, ncol = 1) +
  labs(
    title = "Importancia de Variables por Región — Comparativa de Modelos",
    subtitle = "Top 10 variables más frecuentes en rankings regionales",
    x = "Región EPH",
    y = "Variable",
    fill = "Importancia"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
    axis.text.y = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 11)
  )

ggsave("output/figures/heatmap_variables_region_modelos.png",
       p_heatmap,
       width = 14, height = 10, dpi = 150
)
message("  Heatmap comparativo guardado.")

# ==============================================================================
# 8. PDP: EFECTOS MARGINALES DE VARIABLES CRÍTICAS
# ==============================================================================
message(">>> Calculando PDPs...")

vars_pdp_tecnicas <- c(
  "itcf_real", "ratio_dependencia",
  "max_instruccion", "n_menores", "mca_dim2"
)
vars_pdp <- intersect(vars_pdp_tecnicas, PREDICTORES)

pdp_xgb <- model_profile(
  explainer_xgb,
  variables = vars_pdp,
  N         = 500,
  type      = "partial"
)

# Renombrar variables en el objeto PDP para el gráfico
pdp_xgb$agr_profiles$`_vname_` <- ifelse(
  pdp_xgb$agr_profiles$`_vname_` %in% names(ETIQUETAS_VARS),
  ETIQUETAS_VARS[pdp_xgb$agr_profiles$`_vname_`],
  pdp_xgb$agr_profiles$`_vname_`
)

p_pdp <- plot(pdp_xgb) +
  ggtitle("Partial Dependence Plots — XGBoost",
          subtitle = "Efecto marginal promedio de cada variable sobre P(MPI-pobre)"
  ) +
  theme_minimal(base_size = 11)

ggsave("output/figures/pdp_xgboost.png",
       p_pdp,
       width = 12, height = 8, dpi = 150
)
message("  PDPs guardados.")

# ==============================================================================
# 9. BREAK-DOWN: CUATRO HOGARES TIPO DEL TEST SET
# ==============================================================================
# Criterios de selección para los 4 perfiles solicitados:
#   - Pobre típico:      hogar con P(pobre) máxima (el modelo está más seguro)
#   - Caso frontera:     hogar cuya P(pobre) más se aproxima al umbral calibrado por arriba
#   - Límite no pobre:   hogar cuya P(pobre) más se aproxima al umbral por abajo
#   - No pobre típico:   hogar con P(pobre) mínima (el modelo está más seguro)
# ==============================================================================

message(">>> Calculando Break-down para 4 hogares tipo (XGBoost y Random Forest)...")

# 1. Alineación estricta con el molde de entrenamiento (Nombres técnicos originales)
hogar_test_xai_listo <- hogar_test_xai %>%
  match_classes(reference = data_model_balanced)

columnas_modelo <- names(data_model_balanced) %>% setdiff("mpi_pobre")

# 2. Obtener predicciones de probabilidad para mapear los índices
preds_prob_xgb <- predict(fit_xgb, hogar_test_xai_listo %>% select(any_of(c(columnas_modelo, "codusu", "grupo_cv"))), type = "prob")$.pred_pobre
preds_prob_rf  <- predict(fit_rf,  hogar_test_xai_listo %>% select(any_of(c(columnas_modelo, "codusu", "grupo_cv"))), type = "prob")$.pred_pobre

# Índices de referencia basados en el modelo ancla (XGBoost)
idx_pobre_tipico <- which.max(preds_prob_xgb)
idx_no_pobre     <- which.min(preds_prob_xgb)
idx_frontera     <- which.min(abs(preds_prob_xgb - umbral_xgb))

# Límite no pobre: el más cercano por debajo del umbral
preds_no_pobre_subset <- preds_prob_xgb[preds_prob_xgb < umbral_xgb]
if (length(preds_no_pobre_subset) > 0) {
  val_limite_no_pobre <- preds_no_pobre_subset[which.min(umbral_xgb - preds_no_pobre_subset)]
  idx_limite_no_pobre <- which(preds_prob_xgb == val_limite_no_pobre)[1]
} else {
  idx_limite_no_pobre <- idx_no_pobre
}

indices_bd <- list(
  "Pobre tipico"     = idx_pobre_tipico,
  "Caso frontera"    = idx_frontera,
  "Limite no pobre"  = idx_limite_no_pobre,
  "No pobre tipico"  = idx_no_pobre
)

# 3. Creación de Explainers puros forzando explícitamente DALEX::explain
explainer_xgb_test <- DALEX::explain(
  model            = fit_xgb,
  data             = hogar_test_xai_listo %>% select(all_of(columnas_modelo)), 
  y                = as.numeric(hogar_test_xai_listo$mpi_pobre == "pobre"),
  predict_function = predict_fn,
  label            = "XGBoost",
  verbose          = FALSE
)

explainer_rf_test <- DALEX::explain(
  model            = fit_rf,
  data             = hogar_test_xai_listo %>% select(all_of(columnas_modelo)),
  y                = as.numeric(hogar_test_xai_listo$mpi_pobre == "pobre"),
  predict_function = predict_fn,
  label            = "Random Forest",
  verbose          = FALSE
)

# 4. Función auxiliar segura (Alineación por GGPLOT, objeto nativo e intacto)
procesar_bd_seguro <- function(explainer, idx, nombre_caso, predict_probs, umbral_modelo) {
  
  # Cálculo nativo sin mutaciones riesgosas
  bd <- predict_parts(
    explainer       = explainer,
    new_observation = hogar_test_xai_listo[idx, columnas_modelo],
    type            = "break_down"
  )
  
  prob_hogar <- round(predict_probs[idx], 3)
  clasif <- if_else(prob_hogar >= umbral_modelo, "POBRE", "NO POBRE")
  
  # Gráfico base nativo generado por DALEX
  p_base <- plot(bd)
  
  # scale_y_discrete intercepta el renderizado visual y traduce los nombres técnicos al vuelo
  p <- p_base +
    scale_y_discrete(labels = function(labels_originales) {
      sapply(labels_originales, function(label_indiv) {
        partes <- strsplit(label_indiv, " = ")[[1]]
        var_tecnica <- partes[1]
        
        if (var_tecnica %in% names(ETIQUETAS_VARS)) {
          var_descriptiva <- ETIQUETAS_VARS[var_tecnica]
          if (length(partes) > 1) {
            val_num <- suppressWarnings(as.numeric(partes[2]))
            val_final <- if (!is.na(val_num)) round(val_num, 2) else partes[2]
            return(paste0(var_descriptiva, " = ", val_final))
          } else {
            return(var_descriptiva)
          }
        }
        return(label_indiv)
      })
    }) +
    ggtitle(
      paste0("Contribucion Individual (Break-down) — ", nombre_caso),
      subtitle = paste0(
        explainer$label, " | P(MPI-pobre) = ", prob_hogar,
        " | Umbral = ", round(umbral_modelo, 3),
        " | Clasificacion: ", clasif
      )
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title    = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 9, color = "grey40")
    )
  
  return(list(bd = bd, plot = p))
}

# 5. Generación de corridas para XGBoost (4 casos)
message("  Procesando Break-downs para XGBoost...")
bd_resultados_xgb <- imap(indices_bd, function(idx, nombre) {
  res <- procesar_bd_seguro(explainer_xgb_test, idx, nombre, preds_prob_xgb, umbral_xgb)
  
  nombre_arch <- tolower(stringr::str_replace_all(nombre, " ", "_") %>% stringi::stri_trans_general("Latin-ASCII"))
  ggsave(filename = paste0("output/figures/breakdown_xgb_", nombre_arch, ".png"), 
         plot = res$plot, width = 11, height = 7, dpi = 150)
  return(res)
})

# 6. Generación de corridas para Random Forest (4 casos)
message("  Procesando Break-downs para Random Forest...")
bd_resultados_rf <- imap(indices_bd, function(idx, nombre) {
  res <- procesar_bd_seguro(explainer_rf_test, idx, nombre, preds_prob_rf, umbral_rf)
  
  nombre_arch <- tolower(stringr::str_replace_all(nombre, " ", "_") %>% stringi::stri_trans_general("Latin-ASCII"))
  ggsave(filename = paste0("output/figures/breakdown_rf_", nombre_arch, ".png"), 
         plot = res$plot, width = 11, height = 7, dpi = 150)
  return(res)
})

# 7. Consolidación de paneles comparativos (Patchwork)
library(patchwork)

p_panel_xgb <- (bd_resultados_xgb[["Pobre tipico"]]$plot / 
                  bd_resultados_xgb[["Caso frontera"]]$plot / 
                  bd_resultados_xgb[["Limite no pobre"]]$plot / 
                  bd_resultados_xgb[["No pobre tipico"]]$plot) +
  plot_annotation(
    title = "Atribuciones Globales Break-down: Perfiles Tipo — XGBoost",
    caption = paste0("Umbral optimizado XGBoost: ", round(umbral_xgb, 3))
  )

p_panel_rf <- (bd_resultados_rf[["Pobre tipico"]]$plot / 
                 bd_resultados_rf[["Caso frontera"]]$plot / 
                 bd_resultados_rf[["Limite no pobre"]]$plot / 
                 bd_resultados_rf[["No pobre tipico"]]$plot) +
  plot_annotation(
    title = "Atribuciones Globales Break-down: Perfiles Tipo — Random Forest",
    caption = paste0("Umbral optimizado Random Forest: ", round(umbral_rf, 3))
  )

ggsave("output/figures/breakdown_panel_4_hogares_xgb.png", p_panel_xgb, width = 13, height = 24, dpi = 150)
ggsave("output/figures/breakdown_panel_4_hogares_rf.png",  p_panel_rf,  width = 13, height = 24, dpi = 150)

message(">>> Todos los graficos Break-down individuales y paneles consolidados han sido exportados con exito.")
# ==============================================================================
# 10. GUARDADO FINAL
# ==============================================================================
message(">>> Guardando resultados XAI...")

saveRDS(shap_cart, "output/results/shap_cart.rds")
saveRDS(shap_rf, "output/results/shap_rf.rds")
saveRDS(shap_xgb, "output/results/shap_xgb.rds")
saveRDS(shap_por_region, "output/results/shap_por_region.rds")
saveRDS(shap_por_region_rf, "output/results/shap_por_region_rf.rds")
saveRDS(tabla_shap_regional, "output/results/shap_regional_top5.rds")
saveRDS(tabla_shap_regional_rf, "output/results/shap_regional_top5_rf.rds")
saveRDS(pdp_xgb, "output/results/pdp_xgboost.rds")
saveRDS(bd_resultados, "output/results/breakdown_hogares_tipo.rds")
saveRDS(preds_test_prob, "output/results/preds_test_xgb.rds")
saveRDS(preds_test_prob_rf, "output/results/preds_test_rf.rds")
saveRDS(ETIQUETAS_VARS, "output/results/etiquetas_vars.rds")

message("✓ Step 07 (XAI) completado.")
message("  Figuras en: output/figures/")
message("  Resultados en: output/results/")
