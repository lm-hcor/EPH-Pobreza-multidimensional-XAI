# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 08_graphs.R
# Propósito: Gráficos de calidad para la tesis
#            - Mapa de Argentina con resultados por región
#            - SHAP regional comparado
#            - Evolución temporal del MPI
#            - Comparativa de modelos
#            - Distribución de privaciones
# ==============================================================================

# Esto debería descargar los datos de mapas (aprox 12MB)
remotes::install_github("ropensci/rnaturalearthhires")

library(tidyverse) #nstall.packages si no hecho antes.
library(patchwork)
library(scales)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(viridis)
library(tidytext)
library(stringi)

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

eph_train           <- readRDS("data/processed/eph_train_mca.rds")
eph_test            <- readRDS("data/processed/eph_test_ml.rds")
tabla_resultados    <- readRDS("output/results/tabla_resultados_test.rds")
tabla_shap_regional <- readRDS("output/results/shap_regional_top5.rds")
shap_xgb            <- readRDS("output/results/shap_xgb.rds")
shap_rf             <- readRDS("output/results/shap_rf.rds")
shap_cart           <- readRDS("output/results/shap_cart.rds")
pdp_xgb             <- readRDS("output/results/pdp_xgboost.rds")
ETIQUETAS_VARS      <- readRDS("output/results/etiquetas_vars.rds")
tabla_umbrales      <- readRDS("output/results/umbrales_calibrados.rds")
PREDICTORES         <- readRDS("output/results/predictores_modelo.rds")

# Resultados del placebo si existen
placebo_path <- "output/results/placebo_test_2024_holdout.rds"

if (file.exists(placebo_path)) {
  tabla_placebo <- readRDS(placebo_path)
} else {
  tabla_placebo <- NULL
}

# Paleta de colores consistente
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
  "1" = "GBA",
  "2" = "Pampeana",
  "3" = "Noroeste",
  "4" = "Nordeste",
  "5" = "Cuyo",
  "6" = "Patagonia",
  "GBA" = "GBA",
  "Pampeana" = "Pampeana",
  "Noroeste" = "Noroeste",
  "Nordeste" = "Nordeste",
  "Cuyo" = "Cuyo",
  "Patagonia" = "Patagonia"
)

# Tema base
tema_tesis <- theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 14, hjust = 0),
    plot.subtitle    = element_text(size = 11, color = "grey40", hjust = 0),
    plot.caption     = element_text(size = 9, color = "grey50", hjust = 1),
    legend.position  = "bottom",
    panel.grid.minor = element_blank(),
    strip.text       = element_text(face = "bold")
  )

# ==============================================================================
# 2. MAPA DE ARGENTINA — TASA MPI POR REGIÓN
# ==============================================================================
message(">>> Generando mapa de Argentina...")

# Provincias
arg_provincias <- ne_states(
  country = "Argentina",
  returnclass = "sf"
) %>%
  mutate(name = str_to_title(name))

# Corrección de nombres
arg_provincias <- arg_provincias %>%
  mutate(
    name = case_when(
      name == "Ciudad Autónoma De Buenos Aires" ~ "Ciudad De Buenos Aires",
      name == "Tierra Del Fuego Antártida E Islas Del Atlántico Sur" ~ "Tierra Del Fuego",
      TRUE ~ name
    )
  )

# Normalización ASCII robusta
arg_provincias <- arg_provincias %>%
  mutate(
    name = stri_trans_general(name, "Latin-ASCII")
  )

# Mapeo provincia → región
mapa_prov_region <- tribble(
  ~name,                       ~region,
  "Ciudad De Buenos Aires",    "GBA",
  "Buenos Aires",              "GBA",
  "Santa Fe",                  "Pampeana",
  "Cordoba",                   "Pampeana",
  "Entre Rios",                "Pampeana",
  "La Pampa",                  "Pampeana",
  "Jujuy",                     "Noroeste",
  "Salta",                     "Noroeste",
  "Tucuman",                   "Noroeste",
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
  "Neuquen",                   "Patagonia",
  "Rio Negro",                 "Patagonia",
  "Chubut",                    "Patagonia",
  "Santa Cruz",                "Patagonia",
  "Tierra Del Fuego",          "Patagonia"
) %>%
  mutate(
    name = stri_trans_general(name, "Latin-ASCII")
  )

# Tasa MPI regional
tasa_mpi_region <- eph_train %>%
  mutate(
    region_label = ETIQUETAS_REGION[as.character(region_label)]
  ) %>%
  filter(!is.na(region_label)) %>%
  group_by(region_label) %>%
  summarise(
    tasa_mpi  = weighted.mean(
      mpi_pobre == "pobre",
      pondera,
      na.rm = TRUE
    ),
    n_hogares = n(),
    .groups   = "drop"
  )

# Top SHAP por región
top1_shap_region <- tabla_shap_regional %>%
  group_by(region) %>%
  slice_max(importancia, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(
    region,
    variable_top1 = variable_label,
    shap_top1 = importancia
  )

message(">>> Ensamblando mapa final...")

# ------------------------------------------------------------------------------
# Normalización robusta de nombres
# ------------------------------------------------------------------------------

arg_provincias <- arg_provincias %>%
  mutate(
    name = str_trim(name),
    name = str_to_lower(name),
    name = stringi::stri_trans_general(name, "Latin-ASCII")
  )

mapa_prov_region <- mapa_prov_region %>%
  mutate(
    name = str_trim(name),
    name = str_to_lower(name),
    name = stringi::stri_trans_general(name, "Latin-ASCII")
  )

# ------------------------------------------------------------------------------
# Construcción segura del mapa base
# ------------------------------------------------------------------------------

# 1. Extraer geometría
geom_arg <- st_geometry(arg_provincias)

# 2. Construir dataframe plano SIN sf
arg_df <- arg_provincias %>%
  st_drop_geometry() %>%
  select(name)

# 3. Join provincia -> región
arg_df <- arg_df %>%
  left_join(
    mapa_prov_region,
    by = "name"
  )

# Verificación defensiva
if(!"region" %in% names(arg_df)){
  stop("ERROR: No se pudo crear la columna region.")
}

# Ver provincias no matcheadas
provincias_sin_region <- arg_df %>%
  filter(is.na(region)) %>%
  pull(name)

if(length(provincias_sin_region) > 0){
  message(">>> Provincias sin región detectadas:")
  print(provincias_sin_region)
}

# 4. Join tasas MPI
arg_df <- arg_df %>%
  left_join(
    tasa_mpi_region,
    by = c("region" = "region_label")
  )

# 5. Join SHAP regional
arg_df <- arg_df %>%
  left_join(
    top1_shap_region,
    by = "region"
  )

# 6. Reconstruir objeto sf
arg_map_data <- st_sf(
  arg_df,
  geometry = geom_arg
)

message(">>> ¡Ensamblaje terminado! Listos para ggplot.")

# Centroides
centroides_region <- arg_map_data %>%
  filter(!is.na(region)) %>%
  group_by(region) %>%
  summarise(
    geometry = st_union(geometry),
    .groups = "drop"
  ) %>%
  st_point_on_surface() %>%
  mutate(
    lon = st_coordinates(.)[, 1],
    lat = st_coordinates(.)[, 2]
  ) %>%
  st_drop_geometry() %>%
  left_join(
    tasa_mpi_region,
    by = c("region" = "region_label")
  )

# ------------------------------------------------------------------------------
# 2A. MAPA MPI
# ------------------------------------------------------------------------------

p_mapa_mpi <- ggplot(arg_map_data) +
  geom_sf(
    aes(fill = tasa_mpi * 100),
    color = "white",
    linewidth = 0.3
  ) +
  geom_label(
    data = centroides_region,
    aes(
      x = lon,
      y = lat,
      label = paste0(
        region,
        "\n",
        round(tasa_mpi * 100, 1),
        "%"
      )
    ),
    size = 3,
    fontface = "bold",
    fill = "white",
    alpha = 0.85,
    label.size = 0.2,
    label.padding = unit(0.15, "lines")
  ) +
  scale_fill_viridis_c(
    option = "plasma",
    direction = -1,
    na.value = "grey85",
    name = "Tasa MPI (%)",
    labels = label_number(
      suffix = "%",
      accuracy = 0.1
    )
  ) +
  coord_sf(
    xlim = c(-74, -52),
    ylim = c(-56, -21)
  ) +
  labs(
    title = "Tasa de Pobreza Multidimensional por Región EPH",
    subtitle = "Indicador Alkire-Foster (k = 1/3) — Series 2016-2024, ponderado EPH",
    caption = "Fuente: EPH-INDEC. Cálculo propio mediante Alkire-Foster."
  ) +
  tema_tesis +
  theme(
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    legend.position = "right"
  )

ggsave(
  "output/figures/thesis/mapa_tasa_mpi.png",
  p_mapa_mpi,
  width = 9,
  height = 11,
  dpi = 200
)

message("  Mapa MPI guardado.")

# ------------------------------------------------------------------------------
# 2B. MAPA SHAP TOP 1
# ------------------------------------------------------------------------------

p_mapa_shap <- ggplot(
  arg_map_data %>% filter(!is.na(region))
) +
  geom_sf(
    aes(fill = region),
    color = "white",
    linewidth = 0.3
  ) +
  geom_label(
    data = centroides_region %>%
      left_join(top1_shap_region, by = "region"),
    aes(
      x = lon,
      y = lat,
      label = paste0(region, "\n", variable_top1)
    ),
    size = 2.8,
    fontface = "bold",
    fill = "white",
    alpha = 0.9,
    label.size = 0.2,
    label.padding = unit(0.15, "lines")
  ) +
  scale_fill_manual(
    values = COLORES_REGION,
    na.value = "grey85",
    guide = "none"
  ) +
  coord_sf(
    xlim = c(-74, -52),
    ylim = c(-56, -21)
  ) +
  labs(
    title = "Principal Predictor de Pobreza MPI por Región",
    subtitle = "Variable con mayor |SHAP| promedio — XGBoost",
    caption = "SHAP calculado sobre 300 hogares por región."
  ) +
  tema_tesis +
  theme(
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank()
  )

ggsave(
  "output/figures/thesis/mapa_shap_top1_region.png",
  p_mapa_shap,
  width = 9,
  height = 11,
  dpi = 200
)

message("  Mapa SHAP regional guardado.")

# ==============================================================================
# 3. SHAP REGIONAL COMPARADO
# ==============================================================================
message(">>> Generando gráfico SHAP regional comparado...")

p_shap_regional <- tabla_shap_regional %>%
  mutate(
    region = factor(
      region,
      levels = c(
        "GBA",
        "Pampeana",
        "Noroeste",
        "Nordeste",
        "Cuyo",
        "Patagonia"
      )
    )
  ) %>%
  group_by(region) %>%
  arrange(desc(importancia), .by_group = TRUE) %>%
  mutate(
    variable_label = reorder_within(
      variable_label,
      importancia,
      region
    )
  ) %>%
  ungroup() %>%
  ggplot(
    aes(
      x = importancia,
      y = variable_label,
      fill = region
    )
  ) +
  geom_col(show.legend = FALSE) +
  geom_text(
    aes(label = round(importancia, 3)),
    hjust = -0.15,
    size = 3,
    fontface = "bold"
  ) +
  facet_wrap(
    ~region,
    scales = "free_y",
    ncol = 3
  ) +
  scale_y_reordered() +
  scale_fill_manual(values = COLORES_REGION) +
  scale_x_continuous(
    expand = expansion(mult = c(0, 0.25))
  ) +
  labs(
    title = "Top 5 Predictores por Región — SHAP (XGBoost)",
    subtitle = "Importancia media |SHAP| sobre muestra ponderada por región",
    x = "Importancia media |SHAP|",
    y = NULL,
    caption = "Variables ordenadas por importancia dentro de cada región."
  ) +
  tema_tesis +
  theme(
    strip.background = element_rect(
      fill = "grey92",
      color = NA
    )
  )

ggsave(
  "output/figures/thesis/shap_regional_comparado.png",
  p_shap_regional,
  width = 14,
  height = 10,
  dpi = 200
)

message("  SHAP regional comparado guardado.")

# ==============================================================================
# 4. HEATMAP DE IMPORTANCIA REGIONAL
# ==============================================================================
message(">>> Generando heatmap SHAP regional...")

shap_heatmap_data <- tabla_shap_regional %>%
  select(region, variable_label, importancia) %>%
  mutate(
    region = factor(
      region,
      levels = c(
        "GBA",
        "Pampeana",
        "Noroeste",
        "Nordeste",
        "Cuyo",
        "Patagonia"
      )
    )
  ) %>%
  group_by(variable_label) %>%
  mutate(
    importancia_media = mean(importancia, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  mutate(
    variable_label = fct_reorder(
      variable_label,
      importancia_media
    )
  )

lim_sup_heatmap <- quantile(
  shap_heatmap_data$importancia,
  0.95,
  na.rm = TRUE
)

p_heatmap <- shap_heatmap_data %>%
  ggplot(
    aes(
      x = region,
      y = variable_label,
      fill = importancia
    )
  ) +
  geom_tile(
    color = "white",
    linewidth = 0.5
  ) +
  geom_text(
    aes(label = round(importancia, 3)),
    size = 3,
    color = "white",
    fontface = "bold"
  ) +
  scale_fill_viridis_c(
    option = "magma",
    direction = -1,
    limits = c(0, lim_sup_heatmap),
    oob = squish,
    name = "|SHAP| medio",
    labels = label_number(accuracy = 0.001)
  ) +
  labs(
    title = "Heterogeneidad Regional en la Importancia de Variables",
    subtitle = "Valores SHAP medios por región — XGBoost",
    x = "Región EPH",
    y = NULL,
    caption = "Variables del top 5 regional."
  ) +
  tema_tesis +
  theme(
    legend.position = "right"
  )

ggsave(
  "output/figures/thesis/heatmap_shap_regional.png",
  p_heatmap,
  width = 12,
  height = 7,
  dpi = 200
)

message("  Heatmap SHAP regional guardado.")

# ==============================================================================
# 5. COMPARATIVA DE MODELOS
# ==============================================================================
message(">>> Generando gráfico comparativo de modelos...")

metricas_orden <- c(
  "roc_auc",
  "f_meas",
  "kap",
  "accuracy"
)

metricas_labels <- c(
  "roc_auc" = "ROC-AUC",
  "f_meas" = "F1-score",
  "kap" = "Kappa",
  "accuracy" = "Accuracy"
)

p_metricas <- tabla_resultados %>%
  filter(.metric %in% metricas_orden) %>%
  mutate(
    .metric = factor(
      .metric,
      levels = metricas_orden,
      labels = metricas_labels
    ),
    modelo = factor(
      modelo,
      levels = c(
        "CART",
        "Random Forest",
        "XGBoost"
      )
    )
  ) %>%
  ggplot(
    aes(
      x = .estimate,
      y = modelo,
      fill = modelo
    )
  ) +
  geom_col(
    width = 0.6,
    show.legend = FALSE
  ) +
  geom_text(
    aes(label = round(.estimate, 3)),
    hjust = -0.1,
    size = 3.5,
    fontface = "bold"
  ) +
  facet_wrap(
    ~.metric,
    scales = "free_x",
    ncol = 2
  ) +
  scale_fill_manual(values = COLORES_MODELO) +
  scale_x_continuous(
    expand = expansion(mult = c(0, 0.20))
  ) +
  labs(
    title = "Rendimiento de Modelos en el Test Set 2025",
    subtitle = "Métricas ponderadas por PONDERA (marco censal 2022)",
    x = "Valor de la métrica",
    y = NULL,
    caption = "Threshold calibrado mediante J-index/F1 sobre predicciones OOF."
  ) +
  tema_tesis

ggsave(
  "output/figures/thesis/comparativa_modelos_test.png",
  p_metricas,
  width = 11,
  height = 7,
  dpi = 200
)

message("  Comparativa modelos guardada.")