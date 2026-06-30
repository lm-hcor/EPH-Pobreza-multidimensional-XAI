# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 04_feature_engineering.R
# Propósito: Feature Engineering + Construcción del MPI Alkire-Foster
#
# CORRECCIONES (v2):
#   1. Diagnóstico explícito al inicio: imprime los nombres de eph_final para
#      detectar a tiempo si alguna columna de vivienda falta antes del guard.
#   2. El guard de variables_necesarias se ejecuta ANTES de cualquier
#      transformación, con mensaje de ayuda para el usuario.
#   3. Bug priv_salud: la privación de salud debe usar ch10 (cobertura de
#      salud del individuo), NO ch06 (edad). Se corrige la lógica.
#   4. Se añaden comentarios que explican cada decisión metodológica.
#   5. Se divide en 2016-2023 train set, 2024 como test set, validación temporal
#      y 2025 como validación externa para aislar el impacto del cambio censal
# BUGS RESUELTOS PREVIAMENTE (conservados):
#   1. es_pobre_mon / es_indigente_mon: usar first() porque son vars de hogar.
#   2. p21_real con max() → ahora sum() para ingreso total del hogar.
#   3. region_label duplicada (_x / _y): se resuelve con rename() explícito.

# NOTA METODOLÓGICA (Methods §3):
#   El target (Y) es un MPI basado en el método AF de doble umbral, que integra
#   tres dimensiones: vivienda, saneamiento y educación (pesos iguales 1/3 cada
#   una). Un hogar es "multidimensionalmente pobre" si su puntaje ponderado de
#   privaciones supera el umbral k = 1/3.
#   La pobreza monetaria se conserva como predictor, no como target.
# ==============================================================================
# ==============================================================================

library(tidyverse)
library(janitor)
library(mice)
source("src/00_utils.R")

message(">>> Iniciando Step 04: Feature Engineering + MPI Alkire-Foster...")

# 1. Carga y normalización de nombres
# ------------------------------------------------------------------------------
eph_final <- readRDS("data/processed/eph_final.rds") %>%
  janitor::clean_names()   # Garantiza minúsculas aunque el archivo venga en mayúsculas

# Diagnóstico temprano: mostrar columnas para detectar problemas de nombres
message("  Columnas en eph_final: ", ncol(eph_final))
vars_vivienda <- c("iv3", "iv4", "iv6", "iv11", "iv12_1", "ii1", "ix_tot")
presentes <- intersect(vars_vivienda, names(eph_final))
ausentes  <- setdiff(vars_vivienda, names(eph_final))

if (length(ausentes) > 0) {
  message("  DIAGNÓSTICO — columnas de vivienda NO encontradas: ",
          paste(ausentes, collapse = ", "))
  message("  Columnas disponibles con prefijo 'iv': ",
          paste(grep("^iv", names(eph_final), value = TRUE), collapse = ", "))
  message("  Columnas disponibles con prefijo 'ii': ",
          paste(grep("^ii", names(eph_final), value = TRUE), collapse = ", "))
  stop(
    "ERROR CRÍTICO en Step 04: faltan columnas de vivienda.\n",
    "  Causa probable: en 03_merging.R el select(any_of(VARS_HOGAR)) no encontró\n",
    "  las columnas porque los nombres del archivo crudo no coincidían con VARS_HOGAR.\n",
    "  SOLUCIÓN: verifica que 03_merging.R aplica clean_names() ANTES del select().\n",
    "  Columnas faltantes: ", paste(ausentes, collapse = ", ")
  )
}
message("  Columnas de vivienda OK: ", paste(presentes, collapse = ", "))

# 2. Privación educativa — se calcula a nivel hogar desde la base individual
# ------------------------------------------------------------------------------

# 2a. Niños/as en edad escolar obligatoria (6–17 años) sin asistencia
#     ch05: asistencia escolar (1 = asiste, 2 = asistió pero no asiste, 3 = nunca asistió)
priv_escolar_hogar <- eph_final %>%
  filter(ch06 >= 6, ch06 <= 17) %>%
  group_by(codusu, nro_hogar, ano4, trimestre) %>%
  summarise(
    # Privado si al menos un niño no asiste actualmente (ch05 != 1)
    priv_asistencia_esc = as.integer(any(ch05 %in% c(2, 3), na.rm = TRUE)),
    .groups = "drop"
  )

# 2b. Nivel educativo del jefe/a de hogar (ch03 == 1)
#     nivel_ed: 1 = sin instrucción, 2 = primaria incompleta, 3 = primaria completa, etc.
#     Se usa min() porque en algunos hogares puede haber más de un registro como jefe
#     (errores de captura); conservar el nivel más bajo es la opción conservadora.
nivel_ed_jefe <- eph_final %>%
  filter(ch03 == 1) %>%
  group_by(codusu, nro_hogar, ano4, trimestre) %>%
  summarise(
    nivel_ed_jefe = min(as.numeric(nivel_ed), na.rm = TRUE),
    .groups = "drop"
  )

# 3. Colapso a nivel HOGAR (una fila por hogar)
# ------------------------------------------------------------------------------
# slice(1) selecciona el primer miembro; las variables de hogar son idénticas
# para todos los miembros porque vienen de la base hogar unida en 03_merging.R.
eph_hogar <- eph_final %>%
  group_by(codusu, nro_hogar, ano4, trimestre) %>%
  slice(1) %>%
  ungroup() %>%
  # Incorporar privaciones educativas calculadas arriba
  left_join(priv_escolar_hogar, by = c("codusu", "nro_hogar", "ano4", "trimestre")) %>%
  left_join(nivel_ed_jefe,      by = c("codusu", "nro_hogar", "ano4", "trimestre")) %>%
  mutate(
    # Hogares sin niños en edad escolar: sin privación por definición
    priv_asistencia_esc = replace_na(priv_asistencia_esc, 0L),
    # Hogares sin jefe identificado: imputamos nivel medio (primaria completa = 3)
    nivel_ed_jefe       = replace_na(nivel_ed_jefe, 3L)
  )

# 4. Variables de estructura, trabajo y economía a nivel hogar
# ------------------------------------------------------------------------------
# Nota: Variables de hogar (mismo valor en todos los miembros) se recuperan con first().
# Variables individuales se agregan con sum(), mean() o max() según corresponda.

vars_nivel_hogar <- eph_final %>%
  group_by(codusu, nro_hogar, ano4, trimestre) %>%
  summarise(
    # ── 1. Demografía ────────────────────────────────────────────────────────────
    tamano_hogar = n(),
    ix_tot       = n(),                          # Requerido por construir_mpi()
    n_menores    = sum(ch06 < 18,  na.rm = TRUE),
    n_ancianos   = sum(ch06 >= 65, na.rm = TRUE),
    max_instruccion = max(as.numeric(nivel_ed), na.rm = TRUE),
    
    # ── 2. Mercado de trabajo ────────────────────────────────────────────────────
    n_ocupados        = sum(as.numeric(estado) == 1L, na.rm = TRUE),
    # PRECISIÓN: Ratio basado en Adulto Equivalente (adeq_hogar) vs Ocupados
    ratio_dependencia = first(adeq_hogar) / (n_ocupados + 0.5),
    prop_informal     = mean(pp07h == 2L & as.numeric(estado) == 1L, na.rm = TRUE),
    
    # ── 3. Variables para el TARGET (MPI Alkire-Foster) ──────────────────────────
    # Estas columnas alimentan directamente la función construir_mpi()
    ii1    = first(ii1),    # Cantidad de habitaciones (Hacinamiento)
    iv3    = first(iv3),    # Material de los pisos
    iv4    = first(iv4),    # Material de la cubierta del techo
    iv6    = first(iv6),    # Fuente de agua
    iv11   = first(iv11),   # Desagüe del baño (Cloaca/Pozo)
    iv12_1 = first(iv12_1), # Proximidad a basurales
    
    # ── 4. Variables para el MCA (Contexto y Confort) ────────────────────────────
    # Predictores de entorno socioeconómico que NO entran en la fórmula del MPI
    ii7    = first(ii7),    # Régimen de tenencia (Dueño, inquilino, ocupante)
    ii8    = first(ii8),    # Combustible para cocinar (Gas red, garrafa, leña)
    ii9    = first(ii9),    # Ubicación del baño (Dentro o fuera)
    iv1    = first(iv1),    # Tipo de vivienda (Casa, departamento, pieza)
    iv2    = first(iv2),    # Material predominante de las paredes
    iv5    = first(iv5),    # Revestimiento interior del techo (Cielorraso)
    iv7    = first(iv7),    # Ubicación del suministro de agua
    iv8    = first(iv8),    # ¿Tiene baño/retrete?
    iv10   = first(iv10),   # Tipo de descarga del baño (Botón, cadena, balde)
    v2     = first(v2),     # Se vivió de ayuda de personas que no viven en el hogar
    v13    = first(v13),    # Se vivió de gastar lo ahorrado
    
    # ── 5. Vulnerabilidades y Brechas ────────────────────────────────────────────
    # Salud: 1 si ningún miembro tiene cobertura (obra social/prepaga)
    priv_salud   = as.integer(all(as.numeric(ch10) == 3L, na.rm = TRUE)),
    # Digital: sin computadora Y sin internet en el hogar
    priv_digital = as.integer(first(v11) == 2L & first(v12) == 2L),
    
    # ── 6. Economía e Ingresos ───────────────────────────────────────────────────
    es_pobre_mon     = first(es_pobre_mon),
    es_indigente_mon = first(es_indigente_mon),
    itcf_real        = first(itcf_real),      # Ingreso Total Familiar Real
    adeq_hogar       = first(adeq_hogar),     # Unidades de Adulto Equivalente
    p21_real         = sum(pmax(as.numeric(p21_real), 0), na.rm = TRUE), # Ingreso Ocup. Ppal.
    
    .groups = "drop"
  ) %>%
  # Limpieza de valores extremos o errores de cálculo
  mutate(across(c(p21_real, itcf_real), ~ if_else(is.infinite(.x), 0, .x)))
# Unión al dataset de hogar
# Eliminamos solo las columnas que vamos a reemplazar con los nuevos cálculos
# para evitar duplicados (.x / .y). Las de vivienda (iv3, ii1, etc.) se
# conservan porque vars_nivel_hogar también las trae (son idénticas, pero
# necesitamos que estén en el dataset de hogar para construir_mpi()).
eph_hogar <- eph_hogar %>%
  select(-any_of(c(
    "tamano_hogar", "itcf_real", "p21_real",
    "es_pobre_mon", "es_indigente_mon", "ix_tot",
    # Eliminar las de vivienda del slice(1) para usar las de vars_nivel_hogar
    "iv3", "iv4", "iv6", "iv11", "iv12_1", "ii1", "ii7", "ii8", "ii9"
  ))) %>%
  left_join(vars_nivel_hogar, by = c("codusu", "nro_hogar", "ano4", "trimestre"))

# Guardado de seguridad: verificar que todas las variables del MPI están presentes
variables_necesarias <- c("iv3", "iv4", "iv6", "iv11", "iv12_1", "ii1", "ix_tot",
                          "priv_asistencia_esc", "nivel_ed_jefe")
missing_vars <- setdiff(variables_necesarias, names(eph_hogar))
if (length(missing_vars) > 0) {
  stop(
    "ERROR CRÍTICO en Step 04 (post-join): faltan variables para construir_mpi():\n  ",
    paste(missing_vars, collapse = ", "),
    "\n  Revisa que vars_nivel_hogar las produce correctamente."
  )
}
message("  Variables para construir_mpi(): OK")

# # 4.5 Imputación multivariante con MICE
# # ------------------------------------------------------------------------------
# # La imputación por mediana ignora las relaciones entre variables.
# # MICE modela cada variable faltante condicionando en las demás, produciendo
# # imputaciones más coherentes con la estructura multivariante del dataset.
# # Se usa m = 1 (una sola imputación) porque el objetivo es preparar un dataset
# # único para el pipeline ML, no propagar incertidumbre de imputación.
# # Referencia: van Buuren & Groothuis-Oudshoorn (2011), Journal of Statistical Software.
# # ------------------------------------------------------------------------------
# vars_imputar <- c(
#   # Variables del MPI
#   "iv3", "iv4", "iv6", "iv11", "iv12_1", "ii1", "ix_tot",
#   # Variables educativas
#   "nivel_ed_jefe", "priv_asistencia_esc",
#   # Variables económicas
#   "itcf_real", "p21_real", "adeq_hogar",
#   # Variables demográficas
#   "tamano_hogar", "n_menores", "n_ancianos", "n_ocupados",
#   "ratio_dependencia", "max_instruccion", "prop_informal"
# )
# 
# # Filtrar solo las que existen en eph_hogar Y tienen NAs
# vars_con_na <- vars_imputar %>%
#   keep(~ .x %in% names(eph_hogar) &&
#          sum(is.na(eph_hogar[[.x]])) > 0)
# 
# message("  Variables con NAs para MICE: ", length(vars_con_na))
# message("  ", paste(vars_con_na, collapse = ", "))
# 
# if (length(vars_con_na) > 0) {
#   
#   # Subconjunto para MICE: solo variables relevantes
#   # MICE es O(n × p²) — incluir todas las columnas lo haría inviable
#   df_mice_input <- eph_hogar %>%
#     select(all_of(vars_con_na))
#   
#   set.seed(42)
#   imp <- mice(
#     df_mice_input,
#     m         = 1,      # una sola imputación
#     maxit     = 5,      # 5 iteraciones de cadenas
#     method    = "pmm",  # predictive mean matching
#     printFlag = FALSE   # suprimir output verboso
#   )
#   
#   df_imputado <- complete(imp, action = 1)
#   
#   # Verificación: no deben quedar NAs
#   nas_restantes <- sum(is.na(df_imputado))
#   if (nas_restantes > 0) {
#     warning("MICE dejó ", nas_restantes, " NAs — se imputarán con mediana.")
#     df_imputado <- df_imputado %>%
#       mutate(across(everything(),
#                     ~ if_else(is.na(.x),
#                               median(.x, na.rm = TRUE), .x)))
#   }
#   
#   message("  MICE completado. NAs restantes: ", sum(is.na(df_imputado)))
#   
#   # Reemplazar columnas imputadas en eph_hogar
#   eph_hogar <- eph_hogar %>%
#     select(-all_of(vars_con_na)) %>%
#     bind_cols(df_imputado)
#   
# } else {
#   message("  No hay NAs en las variables relevantes — MICE omitido.") 
# } En modelado se hace step_impute_mode y step_impute_median
# 5. Construcción del MPI Alkire-Foster
# ------------------------------------------------------------------------------
eph_hogar <- construir_mpi(eph_hogar, k = AF_UMBRAL_K)

message("  Tasa de pobreza MPI (ponderada): ",
        round(
          weighted.mean(eph_hogar$mpi_pobre == "pobre",
                        eph_hogar$pondera, na.rm = TRUE) * 100, 1
        ), "%")

# 6. Resolución del duplicado region_label (_x / _y)
# ------------------------------------------------------------------------------
# region_label puede aparecer duplicada si tanto ponderadores_regiones como
# canastas_nacionales la traían en el merge de 03_merging.R.
# _x proviene de ponderadores_regiones (fuente canónica) → la conservamos.
if ("region_label_x" %in% names(eph_hogar)) {
  eph_hogar <- eph_hogar %>%
    rename(region_label = region_label_x) %>%
    select(-any_of("region_label_y"))
  message("  Duplicado region_label resuelto: se conserva _x (ponderadores_regiones).")
}

# 7. Selección del dataset para modelado con limpieza de sufijos
# ------------------------------------------------------------------------------
eph_model_data <- eph_hogar %>%
  select(
    # Identificadores
    codusu, nro_hogar, ano4, trimestre, region_label, aglomerado, pondera,
    
    # Target
    mpi_pobre, mpi_score,
    
    # Privaciones (Nombres limpios)
    priv_piso, priv_techo, priv_hacinamiento,
    priv_agua, priv_cloaca, priv_esc, priv_educ,
    prop_informal, priv_salud, priv_digital,
    
    # REPARACIÓN DE VARIABLES MCA: 
    # Mapeamos los nombres con sufijo (.x) al nombre limpio que pide el Script 06
    iv1 = iv1.x, 
    iv2 = iv2.x, 
    iv5 = iv5.x, 
    iv7 = iv7.x, 
    iv8 = iv8.x, 
    iv10 = iv10.x,
    v2 = v2.x, 
    v13 = v13.x,
    ii7 = any_of(c("ii7", "ii7.x")),
    ii8 = any_of(c("ii8", "ii8.x")),
    ii9 = any_of(c("ii9", "ii9.x")),
    
    # Variables que ya están limpias
    iv3, iv4, iv6, iv11, iv12_1, v1, v5, v11, v12,
    
    # Demografía y Economía
    tamano_hogar, n_menores, n_ancianos, max_instruccion,
    n_ocupados, ratio_dependencia, 
    adeq_hogar = adeq_hogar.x, # También tenía sufijo
    
    itcf_real, p21_real,
    es_pobre_mon, es_indigente_mon
  )

# 8. Segmentación temporal Train / Test y comprobacion variables NA.
# ------------------------------------------------------------------------------
# Train: 2016–2023 (marco censal 2010)
# Test: 2024 (marco censal 2010)
# Test Externo:  2025      (primer año con marco censal 2022)
eph_train <- eph_model_data %>% filter(ano4 < 2024)
eph_test <- eph_model_data |> filter(ano4 == 2024)
eph_externo  <- eph_model_data %>% filter(ano4 == 2025)

message("Verificación de variables MCA críticas:")
c("ii7", "ii8", "ii9", "iv1", "iv2", "iv5", "iv10") %>%
  map_df(~ tibble(
    variable = .x,
    presente = .x %in% names(eph_model_data),
    n_na     = if(.x %in% names(eph_model_data))
      sum(is.na(eph_model_data[[.x]])) else NA_integer_
  )) %>%
  print()

# 9. Guardado
# ------------------------------------------------------------------------------
saveRDS(eph_model_data, "data/processed/eph_model_data.rds")
saveRDS(eph_train,      "data/processed/eph_train_ml.rds")
saveRDS(eph_test,       "data/processed/eph_test_ml.rds")
saveRDS(eph_externo, "data/processed/eph_externo_ml.rds")

message("✓ Step 04 finalizado.")
message("  Hogares totales: ", nrow(eph_model_data))
message("  Train (2016-2023): ", nrow(eph_train))
message("  Test (2024): ", nrow(eph_test))
message("  Test Externo (2025):      ", nrow(eph_externo))

# Verificación: tasa de pobreza monetaria nunca debe ser 1.0 en ninguna región
message("\nVerificación de tasas por región:")
eph_model_data %>%
  group_by(region_label) %>%
  summarise(
    mpi_rate  = weighted.mean(mpi_pobre == "pobre", pondera, na.rm = TRUE),
    mon_rate  = weighted.mean(es_pobre_mon,          pondera, na.rm = TRUE),
    n_hogares = n(),
    .groups   = "drop"
  ) %>%
  print()
