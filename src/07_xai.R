# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 07_xai.R
# Propósito: Capa de interpretabilidad XAI con DALEX
#            SHAP global, SHAP regional, PDP y Break-down para 3 hogares tipo
#
# MEJORAS RESPECTO A LA VERSIÓN ANTERIOR:
#   1. PREDICTORES cargados desde el archivo guardado en 06b para coherencia
#   2. Diccionario de etiquetas: variables técnicas (ii8, ii7...) sustituidas
#      por nombres descriptivos en todos los gráficos
#   3. region_label corregida con etiquetas canónicas antes de crear explainers
#   4. Break-down sobre el test set con MCA proyectado (no sobre eph_test crudo)
#   5. Tres hogares tipo seleccionados con criterio claro:
#      - Mínima probabilidad predicha (no_pobre típico)
#      - Máxima probabilidad predicha (pobre típico)
#      - Frontera de decisión (|prob - umbral| mínimo)
#   6. SHAP regional con etiquetas de región correctas
#   7. Guardado de todos los objetos para uso en 08_graphs.R
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
fit_rf   <- readRDS("output/models/fit_rf.rds")
fit_xgb  <- readRDS("output/models/fit_xgb.rds")

# PREDICTORES cargados desde 06b para garantizar coherencia exacta
PREDICTORES    <- readRDS("output/results/predictores_modelo.rds")
tabla_umbrales <- readRDS("output/results/umbrales_calibrados.rds")

eph_train <- readRDS("data/processed/eph_train_mca.rds")
eph_test  <- readRDS("data/processed/eph_test_ml.rds")

# Umbral calibrado del XGBoost (modelo principal para XAI)
umbral_xgb <- tabla_umbrales %>%
  filter(modelo == "XGBoost") %>%
  pull(umbral)

umbral_rf <- tabla_umbrales %>%
  filter(modelo == "Random Forest") %>%
  pull(umbral)

message("  PREDICTORES activos: ", length(PREDICTORES))
message("  Umbral XGBoost: ", umbral_xgb)

# ==============================================================================
# 2. DICCIONARIO DE ETIQUETAS
# ==============================================================================
# Sustituye nombres técnicos de la EPH por etiquetas descriptivas en los
# gráficos. Se aplica a todos los objetos DALEX mediante el parámetro label.
# ------------------------------------------------------------------------------
ETIQUETAS_VARS <- c(
  # Económicas
  "itcf_real"         = "Ingreso familiar real",
  "p21_real"          = "Ingreso ocup. principal",
  "ingreso_per_capita"= "Ingreso per cápita",
  
  # Educación
  "max_instruccion"   = "Máx. instrucción hogar",
  
  # Demografía
  "tamano_hogar"      = "Tamaño del hogar",
  "n_menores"         = "N.º menores (<18)",
  "n_ancianos"        = "N.º ancianos (≥65)",
  "adeq_hogar"        = "Adulto equivalente",
  "ratio_dependencia" = "Ratio de dependencia",
  "carga_demo"        = "Carga demográfica",
  
  # Mercado de trabajo
  "n_ocupados"        = "N.º ocupados",
  "prop_informal"     = "Prop. informalidad lab.",
  
  # Salud
  "priv_salud"        = "Sin cobertura de salud",
  
  # Temporal
  "ano4"              = "Año",
  "año_norm"          = "Tendencia temporal",
  "trimestre"         = "Trimestre",
  
  # Geografía
  "region_label"      = "Región",
  
  # MCA — contexto socioeconómico
  "mca_dim1"          = "MCA Dim.1 (Bienestar)",
  "mca_dim2"          = "MCA Dim.2 (Infraestr.)",
  
  # Variables MCA originales (activos y vivienda)
  "v2"                = "Posee heladera",
  "v11"               = "Tiene internet",
  "v13"               = "Tiene auto/camioneta",
  "iv1"               = "Tipo de vivienda",
  "iv2"               = "Material paredes",
  "ii7"               = "Régimen de tenencia",
  "ii8"               = "Combustible cocinar",
  "iv5"               = "Revestimiento techo",
  "iv7"               = "Ubicación agua",
  "iv8"               = "Tiene baño",
  "iv10"              = "Descarga del baño",
  "ii9"               = "Ubicación del baño"
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
  "GBA"       = "GBA",
  "Pampeana"  = "Pampeana",
  "Noroeste"  = "Noroeste",
  "Nordeste"  = "Nordeste",
  "Cuyo"      = "Cuyo",
  "Patagonia" = "Patagonia"
)

set.seed(42)
data_xai <- eph_train %>%
  select(all_of(c("mpi_pobre", "region_label", "pondera", PREDICTORES))) %>%
  mutate(
    mpi_pobre    = factor(mpi_pobre, levels = c("no_pobre", "pobre")),
    region_label = ETIQUETAS_REGION[as.character(region_label)],
    region_label = factor(region_label,
                          levels = c("GBA", "Pampeana", "Noroeste",
                                     "Nordeste", "Cuyo", "Patagonia"))
  ) %>%
  drop_na(mpi_pobre, region_label) %>%
  # Muestra representativa ponderada — 5000 observaciones para cómputo eficiente
  slice_sample(n = min(5000, nrow(.)), weight_by = pondera)

message("  Hogares en data_xai: ", nrow(data_xai))
message("  Regiones representadas: ",
        paste(levels(data_xai$region_label), collapse = ", "))
message("  Distribución target:")
print(table(data_xai$mpi_pobre))

# Dataset con etiquetas descriptivas para los gráficos
data_xai_labels <- data_xai %>%
  select(all_of(PREDICTORES)) %>%
  renombrar_con_etiquetas()

# ==============================================================================
# 4. PREPARACIÓN DEL TEST SET CON PROYECCIÓN MCA
# ==============================================================================
# Necesario para los Break-down plots sobre hogares del test set 2025
# Reproduce la proyección MCA del script 06b
# ------------------------------------------------------------------------------
message(">>> Proyectando test set en espacio MCA...")

VARS_MCA_ORIG <- c(
  "v2", "v13", "v5", "v11", "v12",
  "iv1", "iv2", "iv5",
  "ii7", "ii8", "iv10", "ii9"
)

# Cargar MCA entrenado sobre 2021-2024
df_mca_reciente <- eph_train %>%
  filter(ano4 >= 2021) %>%
  select(all_of(VARS_MCA_ORIG)) %>%
  mutate(across(everything(), ~ factor(as.character(.)))) %>%
  mutate(across(everything(),
                ~ fct_lump_prop(., prop = 0.02, other_level = "Otros")))

library(FactoMineR)
res_mca_reciente <- MCA(df_mca_reciente, ncp = 2, graph = FALSE)

# Alinear test con niveles del MCA
niveles_mca <- lapply(df_mca_reciente, levels)

df_test_alineado <- eph_test %>%
  select(all_of(VARS_MCA_ORIG)) %>%
  mutate(across(everything(), as.character))

for (col in names(niveles_mca)) {
  lvls <- niveles_mca[[col]]
  df_test_alineado[[col]] <- factor(
    if_else(df_test_alineado[[col]] %in% lvls,
            df_test_alineado[[col]], "Otros"),
    levels = lvls
  )
}

# Proyección manual robusta
v_coord       <- res_mca_reciente$var$coord
col_names_mca <- rownames(v_coord)
Q             <- length(VARS_MCA_ORIG)

construir_disjuntiva <- function(df, niveles_ref) {
  result <- vector("list", length(niveles_ref))
  names(result) <- names(niveles_ref)
  for (col in names(niveles_ref)) {
    nivs <- niveles_ref[[col]]
    mat  <- model.matrix(~ 0 + ., data = data.frame(x = df[[col]]))
    colnames(mat) <- paste0(col, "_", nivs[match(
      sub("^x", "", colnames(mat)), nivs)])
    result[[col]] <- mat
  }
  do.call(cbind, result)
}

Z_test     <- construir_disjuntiva(df_test_alineado, niveles_mca)
cols_falt  <- setdiff(col_names_mca, colnames(Z_test))
if (length(cols_falt) > 0) {
  mat_cero <- matrix(0, nrow(Z_test), length(cols_falt),
                     dimnames = list(NULL, cols_falt))
  Z_test   <- cbind(Z_test, mat_cero)
}
Z_test        <- Z_test[, col_names_mca, drop = FALSE]
coords_test   <- (Z_test / Q) %*% v_coord
mca_test_proj <- as.data.frame(coords_test) %>%
  setNames(paste0("mca_dim", seq_len(ncol(coords_test)))) %>%
  mutate(across(everything(), ~ if_else(is.na(.), 0, .)))

eph_test_mca <- bind_cols(eph_test, mca_test_proj)

# Preparar test final para predicción
match_classes <- function(target, reference) {
  for (col in names(target)) {
    if (col %in% names(reference)) {
      if (is.factor(reference[[col]])) {
        target[[col]] <- factor(as.character(target[[col]]),
                                levels = levels(reference[[col]]))
      } else {
        class(target[[col]]) <- class(reference[[col]])
      }
    }
  }
  target
}

# Cargar data_model para homologar tipos
data_model_ref <- eph_train %>%
  select(mpi_pobre, codusu, aglomerado, ano4, trimestre,
         pondera, all_of(PREDICTORES)) %>%
  mutate(
    mpi_pobre    = factor(mpi_pobre, levels = c("no_pobre", "pobre")),
    region_label = factor(ETIQUETAS_REGION[as.character(region_label)],
                          levels = c("GBA", "Pampeana", "Noroeste",
                                     "Nordeste", "Cuyo", "Patagonia")),
    aglomerado   = as.character(aglomerado),
    across(c(itcf_real, adeq_hogar, p21_real), as.numeric)
  ) %>%
  drop_na(mpi_pobre) %>%
  mutate(grupo_cv = paste0(codusu, "_", aglomerado)) %>%
  slice_sample(prop = 0.01)  # muestra pequeña solo para referencia de tipos

hogar_test_xai <- eph_test_mca %>%
  mutate(
    grupo_cv     = paste0(codusu, "_", aglomerado),
    region_label = factor(ETIQUETAS_REGION[as.character(region_label)],
                          levels = c("GBA", "Pampeana", "Noroeste",
                                     "Nordeste", "Cuyo", "Patagonia"))
  ) %>%
  select(all_of(c("mpi_pobre", "codusu", "grupo_cv",
                  "aglomerado", "ano4", "pondera", PREDICTORES))) %>%
  mutate(mpi_pobre = factor(mpi_pobre, levels = c("no_pobre", "pobre"))) %>%
  match_classes(reference = data_model_ref) %>%
  drop_na(mpi_pobre)

message("  Test set preparado: ", nrow(hogar_test_xai), " hogares")

# ==============================================================================
# 5. EXPLAINERS DALEX
# ==============================================================================
message(">>> Creando explainers DALEX...")
predict_fn <- function(model, newdata) {
  
  n <- nrow(newdata)
  
  # --- columnas auxiliares EXACTAMENTE con los tipos esperados ---
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
  
  # orden estable
  newdata <- newdata %>%
    relocate(codusu, grupo_cv)
  
  predict(
    model,
    new_data = newdata,
    type = "prob"
  )$.pred_pobre
}

# Dataset con etiquetas para los explainers
data_xai_pred <- data_xai %>%
  mutate(
    codusu = seq_len(n()),
    grupo_cv = paste0("g_", seq_len(n()))
  ) %>%
  select(codusu, grupo_cv, all_of(PREDICTORES))

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
shap_rf   <- calcular_shap_global(explainer_rf)
shap_xgb  <- calcular_shap_global(explainer_xgb)

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
shap_rf_lab   <- aplicar_etiquetas_shap(shap_rf)
shap_xgb_lab  <- aplicar_etiquetas_shap(shap_xgb)

# Gráficos SHAP global con etiquetas
p_shap_cart <- plot(shap_cart_lab) +
  ggtitle("SHAP – CART",
          subtitle = "Importancia media |SHAP| por variable") +
  theme_minimal(base_size = 11)

p_shap_rf <- plot(shap_rf_lab) +
  ggtitle("SHAP – Random Forest",
          subtitle = "Importancia media |SHAP| por variable") +
  theme_minimal(base_size = 11)

p_shap_xgb <- plot(shap_xgb_lab) +
  ggtitle("SHAP – XGBoost",
          subtitle = "Importancia media |SHAP| por variable") +
  theme_minimal(base_size = 11)

p_shap_comparado <- p_shap_cart / p_shap_rf / p_shap_xgb +
  plot_annotation(
    title   = "Importancia de Variables — Comparativa de Modelos",
    subtitle = "SHAP global (Shapley Additive Explanations)",
    caption  = "Muestra: 500 hogares | B = 10 permutaciones"
  )

ggsave("output/figures/shap_global_comparado.png",
       p_shap_comparado, width = 12, height = 18, dpi = 150)
message("  SHAP global guardado.")

# ==============================================================================
# 7. SHAP REGIONAL: XGBOOST POR REGIÓN (CONTRIBUCIÓN ORIGINAL)
# ==============================================================================
message(">>> Calculando SHAP regional (XGBoost)...")

regiones_canon <- c("GBA", "Pampeana", "Noroeste", "Nordeste", "Cuyo", "Patagonia")

shap_por_region <- map(regiones_canon, function(reg) {
  
  datos_reg <- data_xai %>%
    filter(as.character(region_label) == reg)
  
  if (nrow(datos_reg) < 50) {
    message("  ⚠️  Región ", reg, " con < 50 obs — omitida")
    return(NULL)
  }
  
  message("  → Región: ", reg, " (n = ", nrow(datos_reg), ")")
  
  explainer_reg <- explain_tidymodels(
    model            = fit_xgb,
    data             = datos_reg %>% select(all_of(PREDICTORES)),
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

message("\n  Top 3 variables por región:")

tabla_shap_regional %>%
  group_by(region) %>%
  slice_max(importancia, n = 3, with_ties = FALSE) %>%
  select(region, variable_label, importancia) %>%
  arrange(region, desc(importancia)) %>%
  print(n = 30)
# ==============================================================================
# 8. PDP: EFECTOS MARGINALES DE VARIABLES CRÍTICAS
# ==============================================================================
message(">>> Calculando PDPs...")
