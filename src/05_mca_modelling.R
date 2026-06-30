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
  # --- Estructura de Ingresos ---
  "v2",      # Se vivió de Jubilacion o pensión
  "v13",     # Se vivió de gastar lo ahorrado
  "v5",      # Se vivió de asignación por hijo AUH o Asignación por embarazo
  "v11",     # Se vivió de una beca del gobierno para finalizar estudios progresAR
  "v12",     # De ayuda de personas que no viven en el hogar
  
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

# Sin otros.
library(factoextra)
library(dplyr)
library(stringr)

# Año de referencia
anio_ref <- as.character(max(anios_train))

# Objeto MCA
res_mca_obj <- resultados_mca[[anio_ref]]$res_mca

# Renombrar categorías
cats_originales <- rownames(res_mca_obj$var$coord)

cats_nuevas <- cats_originales %>%
  str_replace("iv2_1", "Pared_Ladrillo_Cal") %>%
  str_replace("iv2_2", "Pared_Ladrillo_Solo") %>%
  str_replace("iv2_3", "Pared_Madera_Chapa") %>%
  str_replace("iv2_4", "Pared_Chapa_Carton") %>%
  str_replace("iv2_5", "Pared_Otro_Precario") %>%
  str_replace("iv1_1", "Casa") %>%
  str_replace("iv1_2", "Casilla/Rancho") %>%
  str_replace("iv1_3", "Departamento") %>%
  str_replace("ii7_1", "Prop_Vivienda_y_Terreno") %>%
  str_replace("ii7_2", "Prop_Vivienda_Solo") %>%
  str_replace("ii7_3", "Inquilino") %>%
  str_replace("ii7_4", "Ocupante_pago_impuestos") %>%
  str_replace("ii7_6", "Ocupante_de_hecho") %>%
  str_replace("iv6_1", "Agua_Red_Interna") %>%
  str_replace("iv10_1", "Baño_Inodoro_con_Boton") %>%
  str_replace("iv12_1", "Cloaca_Red") %>%
  str_replace("ii8_1", "Gas_Red") %>%
  str_replace("ii8_2", "Gas_Garrafa") %>%
  str_replace("ii8_3", "Leña_Carbon") %>%
  str_replace("v12_1", "Tiene_Cable") %>%
  str_replace("v12_2", "No_Tiene_Cable") %>%
  str_replace("v2_1", "Vivio_Jubilación") %>%
  str_replace("v2_2", "No_vivió_Jubilación") %>%
  str_replace("v13_1", "Gastaron_ahorros") %>%
  str_replace("v13_2", "No_Gastaron_ahorros") %>%
  str_replace("iv5_1", "Piso_con_Revestimiento") %>%
  str_replace("iv5_2", "Piso_sin_Revestimiento") %>%
  str_replace("ii9_1", "Baño_Interno") %>%
  str_replace("ii9_2", "Baño_Externo")

rownames(res_mca_obj$var$coord) <- cats_nuevas
rownames(res_mca_obj$var$contrib) <- cats_nuevas

# Tabla con contribuciones
tabla_contrib <- data.frame(
  categoria = cats_nuevas,
  contrib = rowSums(res_mca_obj$var$contrib)
)

# Eliminar categorías con NA y otras poco informativas
tabla_contrib <- tabla_contrib %>%
  filter(
    !grepl("\\.NA$", categoria),
    !grepl("Otros", categoria)
  ) %>%
  arrange(desc(contrib))

# Quedarse con las 20 más importantes
cats_a_mostrar <- tabla_contrib %>%
  slice_head(n = 20) %>%
  pull(categoria)

# Biplot limpio
p_biplot_clean <- fviz_mca_var(
  res_mca_obj,
  select.var = list(name = cats_a_mostrar),
  repel = TRUE,
  col.var = "contrib",
  gradient.cols = c("#2C7BB6", "#FDAE61", "#D7191C")
) +
  labs(
    title = paste0("MCA: Estructura de la Privación Material (", anio_ref, ")"),
    subtitle = "Principales categorías de vivienda y servicios",
    x = "Dimensión 1",
    y = "Dimensión 2",
    color = "Contribución"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 11),
    legend.position = "right"
  )

ggsave(
  paste0("output/figures/mca_biplot_clean_", anio_ref, ".png"),
  p_biplot_clean,
  width = 11,
  height = 8,
  dpi = 300
)

print("✓ Biplot limpio generado.")

# ------------------------------------------------------------------------------
# Contribución de categorías al plano factorial
# ------------------------------------------------------------------------------

# Top 15 categorías con mayor contribución total
top_contrib <- tabla_contrib %>%
  slice_head(n = 15) %>%
  mutate(
    categoria = forcats::fct_reorder(categoria, contrib)
  )

p_contrib <- ggplot(
  top_contrib,
  aes(
    x = categoria,
    y = contrib,
    fill = contrib
  )
) +
  geom_col(width = 0.8) +
  coord_flip() +
  scale_fill_gradient(
    low = "#2C7BB6",
    high = "#D7191C"
  ) +
  labs(
    title = paste0(
      "Categorías con mayor contribución al MCA (",
      anio_ref,
      ")"
    ),
    subtitle = "Contribución acumulada sobre las dimensiones 1 y 2",
    x = NULL,
    y = "Contribución (%)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 15
    ),
    plot.subtitle = element_text(
      size = 11,
      colour = "grey40"
    ),
    legend.position = "none"
  )

# Guardar
ggsave(
  paste0(
    "output/figures/mca_contribuciones_",
    anio_ref,
    ".png"
  ),
  p_contrib,
  width = 8,
  height = 6,
  dpi = 300
)

print("✓ Gráfico de contribuciones generado.")