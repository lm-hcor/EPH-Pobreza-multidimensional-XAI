# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 05_mca_feature_selection.R
# Propósito: Análisis de Correspondencias Múltiples (MCA) anual para
#            selección dinámica de variables y reducción de dimensionalidad
# ==============================================================================
# NOTA METODOLÓGICA (Methods §3):
#   El MCA se aplica año a año sobre las variables categóricas de la EPH para:
#     1. Identificar los indicadores con mayor contribución a la varianza
#        socioeconómica en cada periodo.
#     2. Generar coordenadas factoriales como features adicionales para el ML.
#     3. Adaptar la selección de variables a shocks temporales (ej. COVID-19).
#   Solo se envían al pipeline de ML los indicadores que superan el umbral
#   de contribución promedio en los primeros ejes del MCA.
#   Referencia: Asselin (2009); Husson et al. (2011).
# ==============================================================================

library(tidyverse)
library(FactoMineR)   # MCA()
library(factoextra)   # Visualización de biplots
library(janitor)

message(">>> Iniciando Step 05: MCA anual para selección de variables...")

# 1. Carga de datos
# ------------------------------------------------------------------------------
eph_train <- readRDS("data/processed/eph_train_ml.rds")

# 2. Variables candidatas para el MCA
# ------------------------------------------------------------------------------
# El MCA solo admite variables CATEGÓRICAS (factores).
# Seleccionamos variables que describen el "Contexto Estructural" y Confort,
# pero que NO están en la fórmula del MPI (Alkire-Foster) para evitar circularidad.
VARS_MCA <- c(
  # --- Posesión de Activos y Confort (No están en el MPI) ---
  "v2",      # Tiene heladera
  "v13",     # Tiene auto/camioneta
  "v5",      # Tiene computadora 
  "v11",     # Tiene internet
  "v12",     # Tiene cable/satélite
  
  # --- Contexto de Vivienda (No definen privación  MPI) ---
  "iv1",     # Tipo de vivienda (Casa, departamento, pieza)
  "iv2",     # Material de las paredes (Ladrillo, madera, chapa)
  "iv5",     # Revestimiento interior del techo (Cielorraso)
  "ii7",     # Régimen de tenencia (Propietario, inquilino, etc.)
  "ii8",     # Combustible para cocinar (Gas red, garrafa, leña)
  
  # --- Saneamiento de Contexto ---
  "iv10",    # Tipo de descarga del baño (Botón, cadena, balde)
  "ii9"      # Ubicación del baño (Dentro/Fuera de la vivienda)
)
# Nota: iv3, iv4, iv6, iv11, iv12_1 e ii1 se omiten por ser parte del Target MPI.

# 3. Función: MCA para un año concreto
# ------------------------------------------------------------------------------
ejecutar_mca_anual <- function(datos_anio, anio, n_ejes = 2) {
  
  message("  → MCA para año: ", anio, " (n = ", nrow(datos_anio), " hogares)")
  
  # Preparar datos: solo columnas del MCA, como factores
  mca_input <- datos_anio %>%
    select(any_of(VARS_MCA)) %>%
    mutate(across(everything(), ~ factor(as.character(.)))) %>%
    # COLAPSAR CATEGORÍAS RARAS (< 5%) en "Otros" para evitar ruido
    mutate(across(everything(), ~ fct_lump_prop(., prop = 0.05, other_level = "Otros"))) %>%
    # Eliminar columnas con un solo nivel (sin varianza)
    select(where(~ nlevels(.) > 1))
  
  # Verificar que haya al menos 3 variables
  if (ncol(mca_input) < 3) {
    warning("Año ", anio, ": insuficientes variables para MCA. Se omite.")
    return(NULL)
  }
  
  # Ejecutar MCA
  res_mca <- MCA(mca_input, ncp = n_ejes, graph = FALSE)
  
  # Umbral de contribución: 100 / número de categorías activas
  n_cats <- sum(sapply(mca_input, nlevels))
  umbral <- 100 / n_cats
  
  # Contribuciones por VARIABLE (promedio de sus categorías)
  contrib_var <- res_mca$var$contrib %>%
    as.data.frame() %>%
    rownames_to_column("categoria") %>%
    as_tibble() %>%
    mutate(variable = str_replace(categoria, "_[^_]+$", "")) %>%
    group_by(variable) %>%
    summarise(
      contrib_dim1 = mean(`Dim 1`),
      contrib_dim2 = mean(`Dim 2`),
      contrib_media = (contrib_dim1 + contrib_dim2) / 2,
      .groups = "drop"
    ) %>%
    mutate(
      supera_umbral = contrib_media > umbral,
      anio          = anio
    ) %>%
    arrange(desc(contrib_media))
  
  # Coordenadas factoriales para el dataset
  coords <- res_mca$ind$coord %>%
    as_tibble() %>%
    rename_with(~ paste0("mca_dim", seq_along(.)), everything()) %>%
    mutate(
      codusu    = datos_anio$codusu,
      nro_hogar = datos_anio$nro_hogar,
      anio      = anio
    )
  
  list(
    res_mca     = res_mca,
    contrib_var = contrib_var,
    coords      = coords,
    umbral      = umbral,
    vars_selec  = contrib_var %>% filter(supera_umbral) %>% pull(variable)
  )
}

# 4. Ejecución anual
# ------------------------------------------------------------------------------
anios_train <- sort(unique(eph_train$ano4))

resultados_mca <- map(
  anios_train,
  ~ ejecutar_mca_anual(
    datos_anio = eph_train %>% filter(ano4 == .x),
    anio       = .x
  )
) %>%
  set_names(as.character(anios_train))

resultados_mca <- compact(resultados_mca)

# 5. Tabla consolidada de selección
# ------------------------------------------------------------------------------
tabla_vars_selec <- map_df(resultados_mca, ~ .x$contrib_var)

message("  Variables seleccionadas por año:")
print(tabla_vars_selec %>% filter(supera_umbral) %>% count(anio))

# ==============================================================================
# 6. SELECCIÓN DE VARIABLES Y MCA FINAL
# ==============================================================================

# A. Diagnóstico de disponibilidad (Verificación de Step 04)
diagnostico_mca <- eph_train %>%
  group_by(ano4) %>%
  summarise(across(any_of(VARS_MCA), ~ mean(!is.na(.))))

message(">>> Diagnóstico de presencia de variables por año:")
print(diagnostico_mca)

# B. Selección basada en estabilidad (>75%)
vars_estables <- diagnostico_mca %>%
  summarise(across(-ano4, ~ all(. > 0.75))) %>%
  pivot_longer(everything()) %>%
  filter(value == TRUE) %>%
  pull(name)

if(length(vars_estables) < 2) {
  stop("Error: Pocas variables estables detectadas. Verifica el Script 04.")
}

message(">>> Variables finales MCA por estabilidad: ", paste(vars_estables, collapse = ", "))

# C. Ejecución del MCA sobre el set completo
df_mca_input <- eph_train %>%
  select(all_of(vars_estables)) %>%
  mutate(across(everything(), ~ factor(as.character(.))))

res_mca_final <- MCA(df_mca_input, ncp = 2, graph = FALSE)

# 7. Integración de Coordenadas (Join validado)
# ------------------------------------------------------------------------------
coords_all <- map_df(names(resultados_mca), ~ {
  res <- resultados_mca[[.x]]
  res$coords %>% rename(ano4 = anio)
}) %>%
  group_by(codusu, nro_hogar, ano4) %>%
  slice(1) %>% 
  ungroup()

eph_train_mca <- eph_train %>%
  left_join(
    coords_all, 
    by = c("codusu", "nro_hogar", "ano4"),
    relationship = "many-to-one" 
  )

if(nrow(eph_train_mca) == nrow(eph_train)) {
  message("✅ ¡Éxito! Dataset íntegro.")
} else {
  stop("❌ Error: El join alteró el número de filas.")
}

# 8. Guardado y Visualización
# ------------------------------------------------------------------------------
saveRDS(resultados_mca,  "data/processed/mca_resultados_anuales.rds")
saveRDS(eph_train_mca,   "data/processed/eph_train_mca.rds")

anio_ref <- as.character(max(anios_train))
p_biplot <- fviz_mca_var(
  resultados_mca[[anio_ref]]$res_mca,
  repel = TRUE, col.var = "contrib",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  title = paste0("MCA – Contribución de variables (", anio_ref, ")")
)
ggsave(paste0("output/figures/mca_biplot_", anio_ref, ".png"), p_biplot, width = 10, height = 8)

message("✓ Step 05 (MCA) completado con ", length(vars_estables), " variables.")
