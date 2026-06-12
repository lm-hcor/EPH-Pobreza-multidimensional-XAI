# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 08_graphs.R
# Propósito: Gráficos de calidad para la tesis
#            - Mapa de Argentina con resultados por región
#            - SHAP regional comparado
#            - Evolución temporal del MPI
#            - Comparativa de modelos
#            - Distribución de privaciones
#
# DEPENDENCIAS:
#   install.packages(c("sf", "rnaturalearth", "rnaturalearthdata",
#                      "ggspatial", "viridis", "scales", "patchwork"))
#
# NOTA: Los mapas usan el paquete {rnaturalearth} para las provincias
# argentinas y las asignan manualmente a las 6 regiones EPH.
# ==============================================================================

# Esto debería descargar los datos de mapas (aprox 12MB)
remotes::install_github("ropensci/rnaturalearthhires")
library(tidyverse)
library(patchwork)
library(scales)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(viridis)

# Verificar disponibilidad de paquetes espaciales
if (!requireNamespace("rnaturalearth", quietly = TRUE)) {
  stop("Instalar: install.packages(c('rnaturalearth', 'rnaturalearthdata', 'sf'))")
}

message(">>> Iniciando Step 08: Gráficos para la tesis...")
dir.create("output/figures/thesis", recursive = TRUE, showWarnings = FALSE)

# ==============================================================================
# 1. CARGA DE DATOS Y RESULTADOS
# ==============================================================================
message(">>> Cargando resultados del pipeline...")

# Función para crear dummies de región (usada en varias secciones)
crear_dummies_region <- function(df) {
  # Solo crear dummies si region_label existe
  if (!"region_label" %in% names(df)) {
    return(df)
  }
  regiones <- c("Cuyo", "GBA", "Nordeste", "Noroeste", "Pampeana", "Patagonia")
  for (r in regiones) {
    df[[paste0("region_label_", r)]] <-
      as.integer(as.character(df$region_label) == r)
  }
  df %>% select(-region_label)
}

eph_train <- readRDS("data/processed/eph_train_mca.rds")
eph_test <- readRDS("data/processed/eph_test_ml.rds")
eph_externo <- readRDS("data/processed/eph_externo_ml.rds")
tabla_externo <- readRDS("output/results/robustez/tabla_externo_rob.rds")

# ==============================================================================
# 1.1 SELECCIÓN AUTOMÁTICA DEL MEJOR MODELO
# ==============================================================================
# Criterio: Mayor ROC-AUC en datos no vistos (Test 2024 y Externo 2025)
# Compara: Baseline (con max_instruccion) vs Robustez (sin max_instruccion)

message(">>> Seleccionando automáticamente el mejor modelo...")

# Cargar resultados baseline
tabla_test_baseline <- readRDS("output/results/tabla_resultados_test.rds")

# Intentar cargar resultados de robustez si existen
tabla_robustez <- NULL
if (file.exists("output/results/robustez/tabla_test_rob.rds")) {
  tabla_robustez <- readRDS("output/results/robustez/tabla_test_rob.rds")
}

# Intentar cargar resultados del externo 2025
tabla_externo <- NULL
if (file.exists("output/results/robustez/tabla_externo_rob.rds")) {
  tabla_externo <- readRDS("output/results/robustez/tabla_externo_rob.rds")
}

# Función para extraer ROC-AUC por modelo
extraer_auc <- function(tabla, conjunto = "Test") {
  if (is.null(tabla)) {
    return(NULL)
  }
  tabla %>%
    filter(.metric == "roc_auc") %>%
    select(modelo, .estimate) %>%
    mutate(conjunto = conjunto)
}

auc_baseline <- extraer_auc(tabla_test_baseline, "Baseline")
auc_robustez <- extraer_auc(tabla_robustez, "Robustez")

# Combinar y seleccionar mejor modelo
todos_auc <- bind_rows(auc_baseline, auc_robustez) %>%
  arrange(desc(.estimate))

if (nrow(todos_auc) > 0) {
  mejor_modelo_info <- todos_auc %>% slice(1)
  MEJOR_MODELO <- mejor_modelo_info$modelo[1]
  MEJOR_AUC <- mejor_modelo_info$.estimate[1]
  MEJOR_CONJUNTO <- mejor_modelo_info$conjunto[1]
  
  message(
    "  🏆 Mejor modelo: ", MEJOR_MODELO,
    " (ROC-AUC: ", round(MEJOR_AUC, 4),
    " - Conjunto: ", MEJOR_CONJUNTO, ")"
  )
} else {
  # Fallback a baseline
  MEJOR_MODELO <- "Random Forest"
  MEJOR_AUC <- NA
  MEJOR_CONJUNTO <- "Baseline"
  message("  ⚠️ No se pudieron cargar resultados completos. Usando ", MEJOR_MODELO, " por defecto.")
}

# ==============================================================================
# 1.2 CARGA DE RESULTADOS (continuación)
# ==============================================================================
tabla_resultados <- readRDS("output/results/tabla_resultados_test.rds")
tabla_shap_regional <- readRDS("output/results/shap_regional_top5.rds")

# Cargar SHAP RF regional si existe (nuevo en 07_xai.R actualizado)
tabla_shap_regional_rf <- NULL
if (file.exists("output/results/shap_regional_top5_rf.rds")) {
  tabla_shap_regional_rf <- readRDS("output/results/shap_regional_top5_rf.rds")
}
shap_xgb <- readRDS("output/results/shap_xgb.rds")
shap_rf <- readRDS("output/results/shap_rf.rds")
shap_cart <- readRDS("output/results/shap_cart.rds")
pdp_xgb <- readRDS("output/results/pdp_xgboost.rds")
ETIQUETAS_VARS <- readRDS("output/results/etiquetas_vars.rds")
tabla_umbrales <- readRDS("output/results/umbrales_calibrados.rds")
PREDICTORES <- readRDS("output/results/predictores_modelo.rds")
bd_resultados <- readRDS("output/results/breakdown_hogares_tipo.rds")

# Nota: El placebo test ya no forma parte del pipeline actual
# (Train: 2016-2023, Test: 2024, Externo: 2025)
tabla_placebo <- NULL

# Paleta de colores consistente en todos los gráficos
COLORES_REGION <- c(
  "GBA"       = "#1A535C",
  "Pampeana"  = "#4ECDC4",
  "Noroeste"  = "#FF6B6B",
  "Nordeste"  = "#FFE66D",
  "Cuyo"      = "#A8DADC",
  "Patagonia" = "#457B9D"
)

COLORES_MODELO <- c(
  "CART"          = "#E07B54",
  "Random Forest" = "#3A7DC9",
  "XGBoost"       = "#2ECC71"
)

# Etiquetas de región canónicas
ETIQUETAS_REGION <- c(
  "1" = "GBA", "2" = "Pampeana",
  "3" = "Noroeste", "4" = "Nordeste",
  "5" = "Cuyo", "6" = "Patagonia",
  "GBA" = "GBA", "Pampeana" = "Pampeana",
  "Noroeste" = "Noroeste", "Nordeste" = "Nordeste",
  "Cuyo" = "Cuyo", "Patagonia" = "Patagonia"
)

# Tema base para todos los gráficos de tesis
tema_tesis <- theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0),
    plot.subtitle = element_text(size = 11, color = "grey40", hjust = 0),
    plot.caption = element_text(size = 9, color = "grey50", hjust = 1),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold")
  )

# ==============================================================================
# 2. MAPA DE ARGENTINA — TASA MPI POR REGIÓN
# ==============================================================================
message(">>> Generando mapa de Argentina...")

# A. Cargar provincias (Sin el argumento scale que rompe todo)
arg_provincias <- ne_states(country = "Argentina", returnclass = "sf") %>%
  mutate(name = str_to_title(name))

# B. Normalizar nombres de provincias para que el JOIN sea perfecto
# Esto corrige el descalce entre RNaturalEarth y la tabla EPH
arg_provincias <- arg_provincias %>%
  mutate(name = case_when(
    name == "Ciudad Autónoma De Buenos Aires" ~ "Ciudad De Buenos Aires",
    name == "Tierra Del Fuego Antártida E Islas Del Atlántico Sur" ~ "Tierra Del Fuego",
    TRUE ~ name
  ))

# C. Tu tabla de mapeo (Normalizada a Títulos)
mapa_prov_region <- tribble(
  ~name,                       ~region,
  "Ciudad De Buenos Aires",    "GBA",
  "Buenos Aires",              "GBA",
  "Santa Fe",                  "Pampeana",
  "Córdoba",                   "Pampeana",
  "Entre Ríos",                "Pampeana",
  "La Pampa",                  "Pampeana",
  "Jujuy",                     "Noroeste",
  "Salta",                     "Noroeste",
  "Tucumán",                   "Noroeste",
  "Santiago Del Estero",       "Noroeste",
  "Catamarca",                 "Noroeste",
  "La Rioja",                  "Noroeste",
  "Formosa",                   "Nordeste",
  "Chaco",                     "Nordeste",
  "Misiones",                  "Nordeste",
  "Corrientes",                "Nordeste",
  "Mendoza",                   "Cuyo",
  "San Juan",                  "Cuyo",
  "San Luis",                  "Cuyo",
  "Neuquén",                   "Patagonia",
  "Río Negro",                 "Patagonia",
  "Chubut",                    "Patagonia",
  "Santa Cruz",                "Patagonia",
  "Tierra Del Fuego",          "Patagonia"
) %>%
  mutate(name = str_to_title(name))

# Calcular tasa MPI ponderada por región
tasa_mpi_region <- eph_train %>%
  mutate(region_label = ETIQUETAS_REGION[as.character(region_label)]) %>%
  filter(!is.na(region_label)) %>%
  group_by(region_label) %>%
  summarise(
    tasa_mpi    = weighted.mean(mpi_pobre == "pobre", pondera, na.rm = TRUE),
    n_hogares   = n(),
    .groups     = "drop"
  )

# SHAP regional: variable más importante por región
top1_shap_region <- tabla_shap_regional %>%
  group_by(region) %>%
  slice_max(importancia, n = 1, with_ties = FALSE) %>%
  select(region,
         variable_top1 = variable_label,
         shap_top1 = importancia
  )

message(">>> Ensamblando mapa final...")

# Crear dataframe de tasas MPI por región para el join
df_tasa_mpi <- tasa_mpi_region %>%
  rename(tasa_mpi_mapa = tasa_mpi)

# Crear dataframe SHAP top por región para el join
df_shap_top <- top1_shap_region

# 1. Partir de las provincias con geometría
arg_map_data <- arg_provincias %>%
  select(-any_of("region")) # Solo borra 'region' si ya existe

# 2. Unir la región usando el mapeo provincia->región
arg_map_data <- arg_map_data %>%
  left_join(mapa_prov_region %>% select(name, region), by = "name")

# 3. Unir Tasa MPI por región (usando "region" como clave)
arg_map_data <- arg_map_data %>%
  left_join(df_tasa_mpi, by = c("region" = "region_label"))

# Verificar que la columna existe
if (!"tasa_mpi_mapa" %in% names(arg_map_data)) {
  warning("No se pudo unir tasa_mpi_mapa. Verificando columnas:")
  print(names(arg_map_data))
}

# 4. Unir SHAP top por región
arg_map_data <- arg_map_data %>%
  left_join(df_shap_top, by = "region")

message(">>> ¡Ensamblaje terminado! Listos para ggplot.")
# Calcular centroides de región para etiquetas
centroides_region <- arg_map_data %>%
  filter(!is.na(region)) %>%
  group_by(region) %>%
  summarise(geometry = st_union(geometry), .groups = "drop") %>%
  st_centroid() %>%
  mutate(
    lon = st_coordinates(.)[, 1],
    lat = st_coordinates(.)[, 2]
  ) %>%
  st_drop_geometry() %>%
  left_join(tasa_mpi_region, by = c("region" = "region_label"))

# 2A. Mapa de tasa MPI por región
p_mapa_mpi <- ggplot(arg_map_data) +
  geom_sf(aes(fill = tasa_mpi_mapa * 100), color = "white", linewidth = 0.3) +
  geom_label(
    data = centroides_region,
    aes(
      x = lon, y = lat,
      label = paste0(
        region, "\n",
        round(tasa_mpi * 100, 1), "%"
      )
    ),
    size = 3,
    fontface = "bold",
    fill = "white",
    alpha = 0.85,
    linewidth = 0.2,
    label.padding = unit(0.15, "lines")
  ) +
  scale_fill_viridis_c(
    option = "plasma",
    direction = -1,
    name = "Tasa MPI (%)",
    labels = label_number(suffix = "%", accuracy = 0.1)
  ) +
  coord_sf(xlim = c(-74, -52), ylim = c(-56, -21)) +
  labs(
    title    = "Tasa de Pobreza Multidimensional por Región EPH",
    subtitle = "Indicador Alkire-Foster (k = 1/3) — Series 2016-2023, ponderado EPH",
    caption  = "Fuente: EPH-INDEC. Cálculo propio mediante Alkire-Foster."
  ) +
  tema_tesis +
  theme(
    axis.text       = element_blank(),
    axis.ticks      = element_blank(),
    panel.grid      = element_blank(),
    legend.position = "right"
  )

ggsave("output/figures/thesis/mapa_tasa_mpi.png",
       p_mapa_mpi,
       width = 9, height = 11, dpi = 200
)
message("  Mapa MPI guardado.")

# 2B. Mapa de variable SHAP más importante por región
p_mapa_shap <- ggplot(arg_map_data %>% filter(!is.na(region))) +
  geom_sf(aes(fill = region), color = "white", linewidth = 0.3) +
  geom_label(
    data = centroides_region %>%
      left_join(top1_shap_region, by = "region"),
    aes(
      x = lon, y = lat,
      label = paste0(region, "\n", variable_top1)
    ),
    size = 2.8,
    fontface = "bold",
    fill = "white",
    alpha = 0.9,
    label.size = 0.2,
    label.padding = unit(0.15, "lines")
  ) +
  scale_fill_manual(values = COLORES_REGION, guide = "none") +
  coord_sf(xlim = c(-74, -52), ylim = c(-56, -21)) +
  labs(
    title    = "Principal Predictor de Pobreza MPI por Región",
    subtitle = "Variable con mayor |SHAP| promedio — XGBoost",
    caption  = "SHAP calculado sobre 300 hogares por región."
  ) +
  tema_tesis +
  theme(
    axis.text       = element_blank(),
    axis.ticks      = element_blank(),
    panel.grid      = element_blank()
  )

ggsave("output/figures/thesis/mapa_shap_top1_region.png",
       p_mapa_shap,
       width = 9, height = 11, dpi = 200
)
message("  Mapa SHAP regional guardado.")

# ==============================================================================
# 3. SHAP REGIONAL COMPARADO — RANKING DE VARIABLES
# ==============================================================================
message(">>> Generando gráfico SHAP regional comparado...")

p_shap_regional <- tabla_shap_regional %>%
  group_by(region) %>%
  mutate(rank = row_number()) %>%
  ungroup() %>%
  mutate(
    region = factor(region,
                    levels = c(
                      "GBA", "Pampeana", "Noroeste",
                      "Nordeste", "Cuyo", "Patagonia"
                    )
    ),
    variable_label = fct_reorder(variable_label, importancia)
  ) %>%
  ggplot(aes(
    x = importancia, y = variable_label,
    fill = region
  )) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = round(importancia, 3)),
            hjust = -0.1, size = 3
  ) +
  facet_wrap(~region, scales = "free_y", ncol = 3) +
  scale_fill_manual(values = COLORES_REGION) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(
    title    = "Top 5 Predictores por Región — SHAP (XGBoost)",
    subtitle = "Importancia media |SHAP| sobre muestra ponderada por región",
    x        = "Importancia media |SHAP|",
    y        = NULL,
    caption  = "Variables ordenadas por importancia dentro de cada región."
  ) +
  tema_tesis +
  theme(strip.background = element_rect(fill = "grey92", color = NA))

ggsave("output/figures/thesis/shap_regional_comparado.png",
       p_shap_regional,
       width = 14, height = 10, dpi = 200
)
message("  SHAP regional comparado guardado.")

# ==============================================================================
# 4. HEATMAP DE IMPORTANCIA REGIONAL
# ==============================================================================
message(">>> Generando heatmap SHAP regional...")

# Variables que aparecen en el top 5 de al menos una región
vars_en_top5 <- tabla_shap_regional %>%
  pull(variable_label) %>%
  unique()

# Pivot: variable × región con SHAP medio
shap_heatmap_data <- tabla_shap_regional %>%
  select(region, variable_label, importancia) %>%
  complete(region, variable_label, fill = list(importancia = 0)) %>%
  mutate(
    region = factor(region,
                    levels = c(
                      "GBA", "Pampeana", "Noroeste",
                      "Nordeste", "Cuyo", "Patagonia"
                    )
    ),
    variable_label = fct_reorder(variable_label,
                                 importancia,
                                 .fun = mean
    )
  )

p_heatmap <- shap_heatmap_data %>%
  ggplot(aes(x = region, y = variable_label, fill = importancia)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(
    aes(label = if_else(importancia > 0,
                        round(importancia, 3), NA_real_
    )),
    size = 3, color = "white", fontface = "bold"
  ) +
  scale_fill_viridis_c(
    option = "magma", direction = -1,
    name = "|SHAP| medio",
    labels = label_number(accuracy = 0.001)
  ) +
  labs(
    title    = "Heterogeneidad Regional en la Importancia de Variables",
    subtitle = "Valores SHAP medios por región — XGBoost",
    x        = "Región EPH",
    y        = NULL,
    caption  = "Celdas en blanco: variable fuera del top 5 de esa región."
  ) +
  tema_tesis +
  theme(legend.position = "right")

ggsave("output/figures/thesis/heatmap_shap_regional.png",
       p_heatmap,
       width = 12, height = 7, dpi = 200
)
message("  Heatmap SHAP regional guardado.")

# ==============================================================================
# 5. COMPARATIVA DE MODELOS — MÉTRICAS EN TEST
# ==============================================================================
message(">>> Generando gráfico comparativo de modelos...")

metricas_orden <- c("roc_auc", "f_meas", "kap", "accuracy", "sens", "spec")
metricas_labels <- c(
  "roc_auc"  = "ROC-AUC",
  "f_meas"   = "F1-score",
  "kap"      = "Kappa",
  "accuracy" = "Accuracy",
  "sens"     = "Sensibilidad",
  "spec"     = "Especificidad"
)

# Preparar datos baseline (agregando conjunto)
datos_baseline <- tabla_resultados %>%
  filter(.metric %in% metricas_orden) %>%
  mutate(conjunto = "Baseline")

# Preparar datos de robustez (test 2024) si existen
datos_robustez <- NULL
if (!is.null(tabla_robustez) && nrow(tabla_robustez) > 0) {
  datos_robustez <- tabla_robustez %>%
    filter(.metric %in% metricas_orden) %>%
    mutate(conjunto = "Robustez (2024)")
}

# Preparar datos de externo (2025) si existen
datos_externo <- NULL
if (!is.null(tabla_externo) && nrow(tabla_externo) > 0) {
  datos_externo <- tabla_externo %>%
    filter(.metric %in% metricas_orden) %>%
    mutate(conjunto = "Externo (2025)")
}

# Combinar todos los conjuntos
datos_combinados <- bind_rows(datos_baseline, datos_robustez, datos_externo)

p_metricas <- datos_combinados %>%
  mutate(
    .metric = factor(.metric,
                     levels = metricas_orden,
                     labels = metricas_labels
    ),
    modelo = factor(modelo,
                    levels = c("CART", "Random Forest", "XGBoost")
    ),
    conjunto = factor(conjunto,
                      levels = c("Baseline", "Robustez (2024)", "Externo (2025)")
    )
  ) %>%
  ggplot(aes(x = .estimate, y = modelo, fill = conjunto)) +
  geom_col(position = "dodge", width = 0.6, alpha = 0.8) +
  geom_text(aes(label = round(.estimate, 3)),
            position = position_dodge(width = 0.6),
            hjust = -0.1, size = 3, fontface = "bold"
  ) +
  facet_wrap(~.metric, scales = "free_x", ncol = 3) +
  scale_fill_manual(
    values = c(
      "Baseline" = "#3A7DC9",
      "Robustez (2024)" = "#B0C4DE",
      "Externo (2025)" = "#FFB347"
    ),
    name = "Conjunto"
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.20))) +
  labs(
    title    = "Rendimiento de Modelos — Comparativa de Conjuntos",
    subtitle = "Baseline (test 2024) vs Robustez (test 2024) vs Externo (2025) — Métricas ponderadas",
    x        = "Valor de la métrica",
    y        = NULL,
    caption  = "Threshold calibrado mediante J-index/F1 sobre predicciones OOF."
  ) +
  tema_tesis

ggsave("output/figures/thesis/comparativa_modelos_test.png",
       p_metricas,
       width = 11, height = 7, dpi = 200
)
message("  Comparativa modelos guardada.")

# ==============================================================================
# 6. EVOLUCIÓN TEMPORAL DE LA TASA MPI (2016-2025)
# ==============================================================================
# Nota: El placebo test ya no forma parte del pipeline actual
message(">>> Generando evolución temporal del MPI...")

# Preparar datos de train (2016-2023)
evolucion_train <- eph_train %>%
  mutate(region_label = ETIQUETAS_REGION[as.character(region_label)]) %>%
  filter(!is.na(region_label)) %>%
  group_by(ano4, region_label) %>%
  summarise(
    tasa_mpi  = weighted.mean(mpi_pobre == "pobre", pondera, na.rm = TRUE),
    n_hogares = n(),
    .groups   = "drop"
  )

# Preparar datos de test (2024)
evolucion_test <- eph_test %>%
  mutate(region_label = ETIQUETAS_REGION[as.character(region_label)]) %>%
  filter(!is.na(region_label)) %>%
  group_by(ano4, region_label) %>%
  summarise(
    tasa_mpi  = weighted.mean(mpi_pobre == "pobre", pondera, na.rm = TRUE),
    n_hogares = n(),
    .groups   = "drop"
  )

# Preparar datos de externo (2025)
evolucion_externo <- eph_externo %>%
  mutate(region_label = ETIQUETAS_REGION[as.character(region_label)]) %>%
  filter(!is.na(region_label)) %>%
  group_by(ano4, region_label) %>%
  summarise(
    tasa_mpi  = weighted.mean(mpi_pobre == "pobre", pondera, na.rm = TRUE),
    n_hogares = n(),
    .groups   = "drop"
  )

# Combinar todas las series temporales
evolucion_mpi <- bind_rows(evolucion_train, evolucion_test, evolucion_externo)

# Tasa nacional ponderada por año
evolucion_nacional <- evolucion_mpi %>%
  group_by(ano4) %>%
  summarise(
    tasa_mpi     = weighted.mean(tasa_mpi),
    region_label = "Nacional",
    n_hogares    = sum(n_hogares),
    .groups      = "drop"
  )

# Eventos económicos clave para anotar en el gráfico
eventos <- tribble(
  ~ano4, ~evento,
  2018,  "Crisis\ncambiaria",
  2020,  "COVID-19",
  2023,  "Hiperinflación"
)

p_evolucion <- evolucion_mpi %>%
  mutate(region_label = factor(region_label,
                               levels = c(
                                 "GBA", "Pampeana", "Noroeste",
                                 "Nordeste", "Cuyo", "Patagonia"
                               )
  )) %>%
  ggplot(aes(
    x = ano4, y = tasa_mpi * 100,
    color = region_label, group = region_label
  )) +
  geom_line(linewidth = 0.9, alpha = 0.8) +
  geom_point(size = 2) +
  # Línea nacional
  geom_line(
    data = evolucion_nacional,
    aes(x = ano4, y = tasa_mpi * 100),
    color = "black", linewidth = 1.3,
    linetype = "dashed", inherit.aes = FALSE
  ) +
  # Eventos económicos
  geom_vline(
    data = eventos,
    aes(xintercept = ano4),
    linetype = "dotted", color = "grey50", linewidth = 0.8
  ) +
  geom_text(
    data = eventos,
    aes(x = ano4, y = Inf, label = evento),
    vjust = 1.2, size = 2.8, color = "grey40",
    inherit.aes = FALSE
  ) +
  scale_color_manual(values = COLORES_REGION, name = "Región") +
  scale_x_continuous(breaks = 2016:2025) +
  scale_y_continuous(labels = label_number(suffix = "%", accuracy = 0.1)) +
  labs(
    title    = "Evolución de la Tasa MPI por Región (2016–2025)",
    subtitle = "Indicador Alkire-Foster ponderado | Línea negra = promedio nacional",
    x        = "Año",
    y        = "Tasa MPI (%)",
    caption  = "Fuente: EPH-INDEC. Elaboración propia."
  ) +
  tema_tesis

ggsave("output/figures/thesis/evolucion_mpi_regional.png",
       p_evolucion,
       width = 12, height = 7, dpi = 200
)
message("  Evolución temporal MPI guardada.")

# ==============================================================================
# 8. DISTRIBUCIÓN DE PRIVACIONES POR REGIÓN
# ==============================================================================
message(">>> Generando distribución de privaciones...")

vars_priv <- c(
  "priv_piso", "priv_techo", "priv_hacinamiento",
  "priv_agua", "priv_cloaca", "priv_esc", "priv_educ"
)

etiquetas_priv <- c(
  "priv_piso"         = "Piso inadecuado",
  "priv_techo"        = "Techo inadecuado",
  "priv_hacinamiento" = "Hacinamiento",
  "priv_agua"         = "Agua insuficiente",
  "priv_cloaca"       = "Sin cloaca",
  "priv_esc"          = "Niño sin escolarizar",
  "priv_educ"         = "Jefe sin primaria"
)

priv_por_region <- eph_train %>%
  mutate(region_label = ETIQUETAS_REGION[as.character(region_label)]) %>%
  filter(!is.na(region_label)) %>%
  select(region_label, pondera, all_of(vars_priv)) %>%
  pivot_longer(
    cols = all_of(vars_priv),
    names_to = "privacion",
    values_to = "valor"
  ) %>%
  mutate(
    privacion = etiquetas_priv[privacion],
    region_label = factor(region_label,
                          levels = c(
                            "GBA", "Pampeana", "Noroeste",
                            "Nordeste", "Cuyo", "Patagonia"
                          )
    )
  ) %>%
  group_by(region_label, privacion) %>%
  summarise(
    tasa = weighted.mean(valor == 1, pondera, na.rm = TRUE),
    .groups = "drop"
  )

p_privaciones <- priv_por_region %>%
  ggplot(aes(
    x = tasa * 100,
    y = reorder(privacion, tasa),
    fill = region_label
  )) +
  geom_col(position = "dodge", width = 0.7) +
  scale_fill_manual(values = COLORES_REGION, name = "Región") +
  scale_x_continuous(labels = label_number(suffix = "%", accuracy = 1)) +
  labs(
    title    = "Incidencia de Privaciones por Dimensión y Región",
    subtitle = "Porcentaje de hogares con privación — ponderado EPH",
    x        = "% hogares con privación",
    y        = NULL,
    caption  = "Dimensiones del MPI Alkire-Foster: Vivienda, Saneamiento y Educación."
  ) +
  tema_tesis +
  theme(legend.position = "right")

ggsave("output/figures/thesis/privaciones_por_region.png",
       p_privaciones,
       width = 13, height = 7, dpi = 200
)
message("  Distribución de privaciones guardada.")

# ==============================================================================
# 9. PANEL FINAL PARA LA TESIS (PORTADA DE RESULTADOS)
# ==============================================================================
message(">>> Generando panel resumen para la tesis...")

# Panel 3x2 con los gráficos más importantes
# (requiere que todos los anteriores se hayan ejecutado)
p_panel_resumen <- (p_metricas + labs(title = "A. Rendimiento en test")) +
  (p_shap_regional + labs(title = "B. Importancia SHAP por región")) +
  plot_layout(ncol = 1, heights = c(1, 1.5)) +
  plot_annotation(
    title = "Resultados del Pipeline ML-XAI sobre Pobreza Multidimensional",
    subtitle = "Argentina urbana, 2016-2025 | EPH-INDEC",
    caption = "MPI Alkire-Foster (k=1/3) | RF: 0.842 ROC-AUC | XGBoost: 0.838 ROC-AUC (test 2024)"
  )

ggsave("output/figures/thesis/panel_resumen_resultados.png",
       p_panel_resumen,
       width = 14, height = 16, dpi = 200
)
message("  Panel resumen guardado.")

# ==============================================================================
# 10. ROBUSTEZ DE LA DISCRIMINACIÓN Y DEGRADACIÓN DEL THRESHOLD
# ==============================================================================
# Objetivo:
# Demostrar si el cambio censal de la EPH en 2025 afecta principalmente a la
# calibración y al umbral de clasificación, mientras que la capacidad del modelo
# para ordenar correctamente los hogares según su riesgo relativo permanece más estable.
# ==============================================================================
message("\n>>> Sección 10: Robustez de la discriminación y degradación del threshold...")

library(tidyverse)
library(scales)

# 1. Validar y crear estructura de directorios requerida
dir.create("output/figures/thesis", recursive = TRUE, showWarnings = FALSE)
dir.create("output/tables", recursive = TRUE, showWarnings = FALSE)

# 2. Carga de objetos base guardados en los pasos 06 y 07
tabla_umbrales <- readRDS("output/results/umbrales_calibrados.rds")
preds_test_xgb <- readRDS("output/results/preds_test_xgb.rds")
preds_test_rf  <- readRDS("output/results/preds_test_rf.rds")
eph_test       <- readRDS("data/processed/eph_test_ml.rds")
PREDICTORES    <- readRDS("output/results/predictores_modelo.rds")

# Extraer umbrales con fallback de seguridad
umbral_xgb <- tabla_umbrales %>% filter(modelo == "XGBoost") %>% pull(umbral) %>% purrr::pluck(1, .default = 0.5)
umbral_rf  <- tabla_umbrales %>% filter(modelo == "Random Forest") %>% pull(umbral) %>% purrr::pluck(1, .default = 0.5)

# 3. RECONSTRUCCIÓN RIGUROSA DEL DATASET DE PRUEBA (Solución al objeto no encontrado)
ETIQUETAS_REGION <- c(
  "1" = "GBA", "2" = "Pampeana", "3" = "Noroeste", 
  "4" = "Nordeste", "5" = "Cuyo", "6" = "Patagonia",
  "GBA" = "GBA", "Pampeana" = "Pampeana", "Noroeste" = "Noroeste", 
  "Nordeste" = "Nordeste", "Cuyo" = "Cuyo", "Patagonia" = "Patagonia"
)

crear_dummies_region_local <- function(df) {
  regiones <- c("Cuyo", "GBA", "Nordeste", "Noroeste", "Pampeana", "Patagonia")
  for (r in regiones) {
    df[[paste0("region_label_", r)]] <- as.integer(as.character(df$region_label) == r)
  }
  df %>% select(-region_label)
}

# Aquí creamos el objeto contenedor uniendo datos y vectores de predicción
hogar_test_xai_preds <- eph_test %>%
  mutate(
    grupo_cv = paste0(codusu, "_", aglomerado),
    region_label = factor(ETIQUETAS_REGION[as.character(region_label)],
                          levels = c("GBA", "Pampeana", "Noroeste", "Nordeste", "Cuyo", "Patagonia"))
  ) %>%
  select(all_of(c("mpi_pobre", "codusu", "grupo_cv", "aglomerado", "ano4", "pondera", PREDICTORES))) %>%
  mutate(mpi_pobre = factor(mpi_pobre, levels = c("no_pobre", "pobre"))) %>%
  drop_na(mpi_pobre) %>%
  crear_dummies_region_local() %>%
  mutate(
    pred_xgb = as.numeric(preds_test_xgb),
    pred_rf  = as.numeric(preds_test_rf)
  )

message("  Dataset de validación de robustez acoplado con éxito: ", nrow(hogar_test_xai_preds), " hogares.")

# =============================================
# 10.1 DISTRIBUCIÓN DE PROBABILIDADES POR CLASE
# =============================================
message("  10.1 Generando distribución de probabilidades por clase...")

df_densidad <- hogar_test_xai_preds %>%
  select(mpi_pobre, pred_rf, pred_xgb) %>%
  pivot_longer(cols = c(pred_rf, pred_xgb), names_to = "modelo", values_to = "prob") %>%
  mutate(
    modelo = if_else(modelo == "pred_rf", "Random Forest", "XGBoost"),
    clase  = if_else(mpi_pobre == "pobre", "Pobre", "No pobre")
  )

p_densidad_rf <- df_densidad %>%
  filter(modelo == "Random Forest") %>%
  ggplot(aes(x = prob, fill = clase, alpha = clase)) +
  geom_density() +
  scale_fill_manual(values = c("Pobre" = "#E07B54", "No pobre" = "#3A7DC9")) +
  scale_alpha_manual(values = c("Pobre" = 0.5, "No pobre" = 0.5)) +
  labs(
    title = "Distribución de Probabilidades — Random Forest",
    subtitle = "Test 2024", x = "Probabilidad predicha de pobreza",
    y = "Densidad", fill = "Clase real", alpha = "Clase real"
  ) +
  tema_tesis

p_densidad_xgb <- df_densidad %>%
  filter(modelo == "XGBoost") %>%
  ggplot(aes(x = prob, fill = clase, alpha = clase)) +
  geom_density() +
  scale_fill_manual(values = c("Pobre" = "#2ECC71", "No pobre" = "#3A7DC9")) +
  scale_alpha_manual(values = c("Pobre" = 0.5, "No pobre" = 0.5)) +
  labs(
    title = "Distribución de Probabilidades — XGBoost",
    subtitle = "Test 2024", x = "Probabilidad predicha de pobreza",
    y = "Densidad", fill = "Clase real", alpha = "Clase real"
  ) +
  tema_tesis

ggsave("output/figures/thesis/densidad_prob_rf.png",  p_densidad_rf,  width = 9, height = 6, dpi = 200)
ggsave("output/figures/thesis/densidad_prob_xgb.png", p_densidad_xgb, width = 9, height = 6, dpi = 200)

# =============================================
# 10.2 DECILES DE RIESGO (LIFT)
# =============================================
message("  10.2 Generando deciles de riesgo (Lift)...")

df_deciles <- hogar_test_xai_preds %>%
  select(mpi_pobre, pred_rf, pred_xgb) %>%
  pivot_longer(cols = c(pred_rf, pred_xgb), names_to = "modelo", values_to = "prob") %>%
  mutate(modelo = if_else(modelo == "pred_rf", "Random Forest", "XGBoost")) %>%
  group_by(modelo) %>%
  mutate(decil = ntile(prob, 10)) %>%
  group_by(modelo, decil) %>%
  summarise(
    prevalencia = mean(if_else(mpi_pobre == "pobre", 1, 0), na.rm = TRUE), 
    n_hogares   = n(), 
    .groups     = "drop"
  )

p_deciles <- df_deciles %>%
  ggplot(aes(x = decil, y = prevalencia * 100, color = modelo, group = modelo)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_text(aes(label = paste0(round(prevalencia * 100, 1), "%")),
            vjust = -1.2, size = 3, fontface = "bold") +
  scale_color_manual(values = c("Random Forest" = "#3A7DC9", "XGBoost" = "#2ECC71")) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(labels = scales::label_number(suffix = "%", accuracy = 0.1)) +
  labs(
    title = "Deciles de Riesgo — Prevalencia de Pobreza por Decil",
    subtitle = "Test 2024 | Decil 1 = menor riesgo, Decil 10 = mayor riesgo",
    x = "Decil de riesgo", y = "Prevalencia observada de pobreza (%)",
    color = "Modelo"
  ) +
  tema_tesis

ggsave("output/figures/thesis/deciles_riesgo.png", p_deciles, width = 10, height = 7, dpi = 200)

# =============================================
# 10.3 CURVAS PRECISIÓN-RECALL
# =============================================
message("  10.3 Generando curvas Precisión-Recall...")

calcular_pr_curve_safe <- function(df_sub) {
  truth_vec <- if_else(df_sub$mpi_pobre == "pobre", 1L, 0L)
  prob_vec  <- df_sub$prob
  
  data.frame(truth = truth_vec, prob = prob_vec) %>%
    arrange(desc(prob)) %>%
    mutate(
      cum_tp = cumsum(truth),
      cum_fp = cumsum(1 - truth),
      precision = cum_tp / (cum_tp + cum_fp),
      recall = cum_tp / max(1, sum(truth))
    )
}

df_pr <- hogar_test_xai_preds %>%
  select(mpi_pobre, pred_rf, pred_xgb) %>%
  pivot_longer(cols = c(pred_rf, pred_xgb), names_to = "modelo", values_to = "prob") %>%
  mutate(modelo = if_else(modelo == "pred_rf", "Random Forest", "XGBoost")) %>%
  group_by(modelo) %>%
  group_modify(~ calcular_pr_curve_safe(.x)) %>%
  ungroup()

auprc_vals <- df_pr %>%
  group_by(modelo) %>%
  summarise(
    auprc = sum(diff(recall) * (head(precision, -1) + tail(precision, -1)) / 2, na.rm = TRUE),
    .groups = "drop"
  )

p_pr <- df_pr %>%
  left_join(auprc_vals, by = "modelo") %>%
  ggplot(aes(x = recall * 100, y = precision * 100, color = modelo)) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = c("Random Forest" = "#3A7DC9", "XGBoost" = "#2ECC71")) +
  scale_x_continuous(labels = scales::label_number(suffix = "%")) +
  scale_y_continuous(labels = scales::label_number(suffix = "%")) +
  geom_text(data = auprc_vals, aes(x = 85, y = 15, label = paste0("AUPRC: ", round(auprc, 3))),
            size = 4, fontface = "bold", inherit.aes = TRUE) +
  labs(
    title = "Curvas Precision-Recall",
    subtitle = "Test 2024 | AUPRC = Average Precision",
    x = "Recall (Sensibilidad)", y = "Precision", color = "Modelo"
  ) +
  tema_tesis

ggsave("output/figures/thesis/curva_precision_recall.png", p_pr, width = 9, height = 7, dpi = 200)

# =============================================
# 10.4 F1 EN FUNCIÓN DEL THRESHOLD
# =============================================
message("  10.4 Generando F1 vs Threshold...")

calcular_f1_grid <- function(df_sub) {
  truth_vec <- if_else(df_sub$mpi_pobre == "pobre", 1L, 0L)
  prob_vec  <- df_sub$prob
  
  thresholds <- seq(0.01, 0.50, by = 0.01)
  
  map_dfr(thresholds, function(t) {
    pred_class <- if_else(prob_vec >= t, 1L, 0L)
    tp <- sum(pred_class == 1 & truth_vec == 1)
    fp <- sum(pred_class == 1 & truth_vec == 0)
    fn <- sum(pred_class == 0 & truth_vec == 1)
    
    precision <- if_else(tp + fp > 0, tp / (tp + fp), 0)
    recall    <- if_else(tp + fn > 0, tp / (tp + fn), 0)
    f1        <- if_else(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    
    data.frame(threshold = t, precision = precision, recall = recall, f1 = f1)
  })
}

df_f1 <- hogar_test_xai_preds %>%
  select(mpi_pobre, pred_rf, pred_xgb) %>%
  pivot_longer(cols = c(pred_rf, pred_xgb), names_to = "modelo", values_to = "prob") %>%
  mutate(modelo = if_else(modelo == "pred_rf", "Random Forest", "XGBoost")) %>%
  group_by(modelo) %>%
  group_modify(~ calcular_f1_grid(.x)) %>%
  ungroup()

umbral_optimo <- df_f1 %>%
  group_by(modelo) %>%
  slice_max(f1, n = 1, with_ties = FALSE) %>%
  select(modelo, umbral_optimo = threshold, f1_max = f1)

f1_oof <- df_f1 %>%
  group_by(modelo) %>%
  mutate(
    target_umbral = if_else(modelo == "Random Forest", umbral_rf, umbral_xgb),
    distancia = abs(threshold - target_umbral)
  ) %>%
  slice_min(distancia, n = 1, with_ties = FALSE) %>%
  select(modelo, f1_oof = f1, threshold_oof = threshold) %>%
  ungroup()

p_f1_threshold <- df_f1 %>%
  ggplot(aes(x = threshold * 100, y = f1 * 100, color = modelo)) +
  geom_line(linewidth = 1.2) +
  geom_vline(data = umbral_optimo, aes(xintercept = umbral_optimo * 100, color = modelo), linetype = "dashed", linewidth = 0.8) +
  geom_vline(data = f1_oof, aes(xintercept = threshold_oof * 100, color = modelo), linetype = "dotted", linewidth = 1) +
  scale_color_manual(values = c("Random Forest" = "#3A7DC9", "XGBoost" = "#2ECC71")) +
  scale_x_continuous(labels = scales::label_number(suffix = "%")) +
  scale_y_continuous(labels = scales::label_number(suffix = "%")) +
  labs(
    title = "F1-Score en Funcion del Threshold",
    subtitle = "Linea punteada = threshold OOF calibrado | Linea discontinua = threshold optimo F1",
    x = "Threshold de clasificacion", y = "F1-Score", color = "Modelo"
  ) +
  tema_tesis

ggsave("output/figures/thesis/f1_vs_threshold.png", p_f1_threshold, width = 10, height = 7, dpi = 200)

# =============================================
# 10.5 CURVAS DE CALIBRACIÓN
# =============================================
message("  10.5 Generando curvas de calibración...")

df_cal <- hogar_test_xai_preds %>%
  select(mpi_pobre, pred_rf, pred_xgb) %>%
  pivot_longer(cols = c(pred_rf, pred_xgb), names_to = "modelo", values_to = "prob") %>%
  mutate(modelo = if_else(modelo == "pred_rf", "Random Forest", "XGBoost")) %>%
  group_by(modelo) %>%
  mutate(bin = ntile(prob, 10)) %>%
  group_by(modelo, bin) %>%
  summarise(
    prob_media  = mean(prob, na.rm = TRUE), 
    prevalencia = mean(if_else(mpi_pobre == "pobre", 1, 0), na.rm = TRUE), 
    n           = n(), 
    .groups     = "drop"
  )

p_calibracion <- df_cal %>%
  ggplot(aes(x = prob_media * 100, y = prevalencia * 100, color = modelo)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Random Forest" = "#3A7DC9", "XGBoost" = "#2ECC71")) +
  scale_x_continuous(labels = scales::label_number(suffix = "%"), limits = c(0, 100)) +
  scale_y_continuous(labels = scales::label_number(suffix = "%"), limits = c(0, 100)) +
  coord_fixed() +
  labs(
    title = "Curvas de Calibracion",
    subtitle = "Probabilidad predicha vs Frecuencia observada | Linea punteada = calibracion perfecta",
    x = "Probabilidad predicha media (%)", y = "Frecuencia observada de pobreza (%)",
    color = "Modelo"
  ) +
  tema_tesis

ggsave("output/figures/thesis/calibracion.png", p_calibracion, width = 9, height = 8, dpi = 200)

# =============================================
# 10.6 TABLA RESUMEN DE ROBUSTEZ
# =============================================
message("  10.6 Generando tabla resumen de robustez...")

auc_calculado <- hogar_test_xai_preds %>%
  select(mpi_pobre, pred_rf, pred_xgb) %>%
  pivot_longer(cols = c(pred_rf, pred_xgb), names_to = "modelo", values_to = "prob") %>%
  mutate(modelo = if_else(modelo == "pred_rf", "Random Forest", "XGBoost")) %>%
  group_by(modelo) %>%
  summarise(
    roc_auc = as.numeric(yardstick::roc_auc_vec(factor(mpi_pobre, levels = c("pobre", "no_pobre")), prob)),
    .groups = "drop"
  )

tabla_final <- umbral_optimo %>%
  left_join(f1_oof, by = "modelo") %>%
  mutate(
    diff_f1  = f1_max - f1_oof, 
    conjunto = "Test 2024"
  ) %>%
  left_join(auprc_vals, by = "modelo") %>%
  left_join(auc_calculado, by = "modelo") %>%
  select(
    modelo, conjunto, roc_auc, 
    pr_auc = auprc, umbral_optimo, 
    f1_max, f1_oof, diff_f1
  )

write.csv(tabla_final, "output/tables/robustez_threshold.csv", row.names = FALSE, na = "")
message(">>> Sección 10 ejecutada y guardada sin errores intermedios.")

# ==============================================================================
# 11. RESUMEN DE ARCHIVOS GENERADOS
# ==============================================================================
message("\n✓ Step 08 completado.")
message("  Figuras generadas en output/figures/thesis/:")
archivos_generados <- list.files("output/figures/thesis/", pattern = "\\.png$")
for (f in archivos_generados) {
  message("  - ", f)
}
