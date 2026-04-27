# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 07_xai.R
# Propósito: Capa de interpretabilidad XAI con DALEX
#            SHAP global, LIME local, PDP y análisis regional
# ==============================================================================
# NOTA METODOLÓGICA (Methods §3):
#   - SHAP (Shapley Additive Explanations): contribución global de cada variable.
#   - LIME / Break-down: explicaciones locales de hogares individuales.
#   - PDP (Partial Dependence Plots): efectos marginales de variables críticas.
#   El mejor modelo en ROC-AUC (habitualmente XGBoost) es el usado para XAI
#   regional. Los tres modelos se analizan con SHAP global para comparación.
#   Referencia: Biecek & Burzykowski (2021); Lundberg & Lee (2017).
# ==============================================================================

library(tidyverse)
library(DALEX)         # Explainer agnóstico al modelo
library(DALEXtra)      # Integración con tidymodels
library(ingredients)   # PDP y SHAP
library(iBreakDown)    # Break-down / LIME local
library(patchwork)     # Composición de gráficos
library(janitor)

message(">>> Iniciando Step 07: Análisis XAI con DALEX...")

# 1. Carga de modelos y datos
# ------------------------------------------------------------------------------
fit_cart <- readRDS("output/models/fit_cart.rds")
fit_rf   <- readRDS("output/models/fit_rf.rds")
fit_xgb  <- readRDS("output/models/fit_xgb.rds")

eph_train <- readRDS("data/processed/eph_train_mca.rds")
eph_test  <- readRDS("data/processed/eph_test_ml.rds")

vars_estables <- readRDS("data/processed/mca_vars_estables.rds")

PREDICTORES <- unique(c(
  vars_estables,
  "itcf_real", "p21_real",
  "tamano_hogar", "n_menores", "n_ancianos",
  "ratio_dependencia", "n_ocupados",
  #"es_pobre_mon",
  grep("^mca_dim", names(eph_train), value = TRUE)
))
PREDICTORES <- intersect(PREDICTORES, names(eph_train))

# Subconjunto representativo de entrenamiento (DALEX es intensivo en cómputo)
set.seed(42)
data_xai <- eph_train %>%
  select(all_of(c("mpi_pobre", "region_label", "pondera", PREDICTORES))) %>%
  mutate(
    mpi_pobre = factor(mpi_pobre, levels = c("no_pobre", "pobre")),
    across(any_of(c("iv3","iv4","iv6","iv11","iv12","ii7","ii1")),
           ~ factor(as.character(.)))
  ) %>%
  drop_na(mpi_pobre) %>%
  # Muestreo estratificado para reducir tiempo de cómputo manteniendo representatividad
  slice_sample(n = min(5000, nrow(.)), weight_by = pondera)

# 2. Explainers DALEX para cada modelo
# ------------------------------------------------------------------------------
# DALEX usa una función de predicción que devuelve la probabilidad de la clase positiva

predict_fn <- function(model, newdata) {
  predict(model, newdata, type = "prob")$.pred_pobre
}

explainer_cart <- explain_tidymodels(
  model      = fit_cart,
  data       = data_xai %>% select(all_of(PREDICTORES)),
  y          = as.numeric(data_xai$mpi_pobre == "pobre"),
  label      = "CART",
  predict_function = predict_fn,
  verbose    = FALSE
)

explainer_rf <- explain_tidymodels(
  model      = fit_rf,
  data       = data_xai %>% select(all_of(PREDICTORES)),
  y          = as.numeric(data_xai$mpi_pobre == "pobre"),
  label      = "Random Forest",
  predict_function = predict_fn,
  verbose    = FALSE
)

explainer_xgb <- explain_tidymodels(
  model      = fit_xgb,
  data       = data_xai %>% select(all_of(PREDICTORES)),
  y          = as.numeric(data_xai$mpi_pobre == "pobre"),
  label      = "XGBoost",
  predict_function = predict_fn,
  verbose    = FALSE
)

# 3. SHAP Global: Contribución de variables (los tres modelos)
# ------------------------------------------------------------------------------
message("  Calculando SHAP global (los tres modelos)...")

calcular_shap_global <- function(explainer, n_sample = 500) {
  model_parts(
    explainer,
    type        = "shap",
    B           = 10,           # Número de permutaciones para estabilizar SHAP
    N           = n_sample
  )
}

shap_cart <- calcular_shap_global(explainer_cart)
shap_rf   <- calcular_shap_global(explainer_rf)
shap_xgb  <- calcular_shap_global(explainer_xgb)

# Gráficos SHAP global
p_shap_cart <- plot(shap_cart) + ggtitle("SHAP – CART")
p_shap_rf   <- plot(shap_rf)   + ggtitle("SHAP – Random Forest")
p_shap_xgb  <- plot(shap_xgb)  + ggtitle("SHAP – XGBoost")

p_shap_comparado <- p_shap_cart / p_shap_rf / p_shap_xgb
ggsave("output/figures/shap_global_comparado.png",
       p_shap_comparado, width = 12, height = 16, dpi = 150)

# 4. SHAP Regional: XGBoost desagregado por región (innovación metodológica)
# ------------------------------------------------------------------------------
message("  Calculando SHAP por región (XGBoost)...")

regiones <- unique(data_xai$region_label)

shap_por_region <- map(regiones, function(reg) {
  
  datos_reg <- data_xai %>% filter(region_label == reg)
  if (nrow(datos_reg) < 50) return(NULL)
  
  explainer_reg <- explain_tidymodels(
    model      = fit_xgb,
    data       = datos_reg %>% select(all_of(PREDICTORES)),
    y          = as.numeric(datos_reg$mpi_pobre == "pobre"),
    label      = paste0("XGBoost – ", reg),
    predict_function = predict_fn,
    verbose    = FALSE
  )
  
  shap <- model_parts(explainer_reg, type = "shap", B = 10,
                      N = min(300, nrow(datos_reg)))
  list(region = reg, shap = shap)
}) %>%
  compact() %>%
  set_names(map_chr(., ~ .x$region))

# Tabla de importancia por región (top 5 variables)
tabla_shap_regional <- map_df(shap_por_region, function(x) {
  x$shap %>%
    as_tibble() %>%
    group_by(variable) %>%
    summarise(mean_abs_shap = mean(abs(contribution)), .groups = "drop") %>%
    arrange(desc(mean_abs_shap)) %>%
    slice(1:5) %>%
    mutate(region = x$region)
})

message("  Top 3 variables por región:")
tabla_shap_regional %>%
  group_by(region) %>%
  slice(1:3) %>%
  print()

saveRDS(tabla_shap_regional, "output/results/shap_regional_top5.rds")

# 5. PDP: Efectos marginales de variables críticas
# ------------------------------------------------------------------------------
message("  Calculando PDPs para variables clave...")

vars_pdp <- c("itcf_real", "ratio_dependencia", "max_instruccion", "n_menores")
vars_pdp <- intersect(vars_pdp, PREDICTORES)

pdp_xgb <- model_profile(
  explainer_xgb,
  variables = vars_pdp,
  N         = 500,
  type      = "partial"
)

p_pdp <- plot(pdp_xgb) +
  ggtitle("Partial Dependence Plots – XGBoost") +
  theme_minimal()

ggsave("output/figures/pdp_xgboost.png", p_pdp, width = 12, height = 8, dpi = 150)

# 6. Break-down Local: Explicación de hogares individuales
# ------------------------------------------------------------------------------
message("  Calculando explicaciones locales (Break-down)...")

# Seleccionamos 3 hogares representativos: pobre "típico", "borderline", no pobre
preds_test <- predict(fit_xgb, eph_test %>%
                        select(all_of(PREDICTORES)) %>%
                        mutate(across(any_of(c("iv3","iv4","iv6","iv11","iv12")),
                                      ~ factor(as.character(.)))),
                      type = "prob")$.pred_pobre

indices <- list(
  pobre_tipico   = which.max(preds_test),
  borderline     = which.min(abs(preds_test - 0.5)),
  no_pobre       = which.min(preds_test)
)

hogar_test_xai <- eph_test %>%
  select(all_of(PREDICTORES)) %>%
  mutate(across(any_of(c("iv3","iv4","iv6","iv11","iv12")),
                ~ factor(as.character(.))))

bd_plots <- imap(indices, function(idx, nombre) {
  bd <- predict_parts(
    explainer = explainer_xgb,
    new_observation = hogar_test_xai[idx, ],
    type = "break_down"
  )
  plot(bd) + ggtitle(paste0("Break-down: hogar ", nombre))
})

# Guardamos los plots locales
walk2(bd_plots, names(bd_plots), function(p, nombre) {
  ggsave(
    filename = paste0("output/figures/breakdown_", nombre, ".png"),
    plot     = p, width = 10, height = 6, dpi = 150
  )
})

# 7. Guardado final
# ------------------------------------------------------------------------------
saveRDS(shap_cart,          "output/results/shap_cart.rds")
saveRDS(shap_rf,            "output/results/shap_rf.rds")
saveRDS(shap_xgb,           "output/results/shap_xgb.rds")
saveRDS(shap_por_region,    "output/results/shap_por_region.rds")
saveRDS(pdp_xgb,            "output/results/pdp_xgboost.rds")

message("✓ Step 07 (XAI) completado.")
message("  Figuras guardadas en: output/figures/")
