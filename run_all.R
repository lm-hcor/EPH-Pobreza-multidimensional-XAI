# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Propósito: Ejecución INTEGRAL del pipeline - OPTIMIZADO PARA EVITAR REENTRENAMIENTO
# ==============================================================================
# ESTRATEGIA:
#   - Fases 1-4: Ejecución completa (preparación de datos)
#   - Fase 5: SALTO de modelado (usa modelos existentes)
#   - Fases 6-8: Ejecución completa (XAI y gráficos)
#
# ADVERTENCIA: Este script asume que los modelos ya fueron entrenados
# (fit_rf.rds, fit_xgb.rds, best_rf.rds, best_xgb.rds existen en output/models/)
# ==============================================================================

# Función para manejar errores graceful
safe_source <- function(file) {
  tryCatch({
    source(file)
    TRUE
  }, error = function(e) {
    message("⚠️ Error en ", file, ": ", e$message)
    FALSE
  })
}

library(tidyverse)
library(tidymodels)
library(doParallel)

# ==============================================================================
# 0. OPTIMIZACIÓN DE RECURSOS
# ==============================================================================
# Control de hilos nativos para evitar deadlock en Windows
Sys.setenv(
  OMP_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1"
)

n_cores <- parallel::detectCores(logical = FALSE)
workers_optimos <- max(1, ceiling(n_cores / 2))

cl <- parallel::makeCluster(2, type = "PSOCK", setup_strategy = "sequential")
doParallel::registerDoParallel(cl)

message(">>> Núcleos físicos: ", n_cores, " | Workers: 2")
message(">>> Configuración de rendimiento aplicada (evitar deadlock Windows)")

# ==============================================================================
# VERIFICACIÓN DE ARCHIVOS EXISTENTES
# ==============================================================================
message("\n", strrep("=", 60))
message("VERIFICANDO ARTEFACTOS EXISTENTES...")
message(strrep("=", 60))

artefactos_requeridos <- c(
  "output/models/fit_cart.rds",
  "output/models/fit_rf.rds",
  "output/models/fit_xgb.rds",
  "output/models/best_cart.rds",
  "output/models/best_rf.rds",
  "output/models/best_xgb.rds",
  "output/models/tune_cart.rds",
  "output/models/tune_rf.rds",
  "output/models/tune_xgb.rds"
  
)

artefactos_existentes <- file.exists(artefactos_requeridos)
if (all(artefactos_existentes)) {
  message("✅ Todos los artefactos de modelos existen. Pipeline optimizado activado.")
} else {
  faltantes <- artefactos_requeridos[!artefactos_existentes]
  message("⚠️  Faltan artefactos:")
  for (f in faltantes) message("   - ", f)
  message("   El pipeline completo de modelado será ejecutado.")
}

# ==============================================================================
# FASE 1: PREPARACIÓN Y LIMPIEZA
# ==============================================================================
message("\n", strrep("=", 60))
message("FASE 1: PREPARACIÓN Y LIMPIEZA")
message(strrep("=", 60))

message(">>> Paso 1/8: Funciones utilitarias...")
source("src/00_utils.R")

message(">>> Paso 2/8: Descarga de microdatos EPH...")
source("src/01_download.R")

message(">>> Paso 3/8: Limpieza de IPC...")
source("src/02_clean_ipc.R")

message(">>> Paso 4/8: Limpieza de canastas...")
source("src/02b_clean_canastas.R")

message("✅ Fase 1 completada: Datos base y canastas listos.")

# ==============================================================================
# FASE 2: PROCESAMIENTO
# ==============================================================================
message("\n", strrep("=", 60))
message("FASE 2: PROCESAMIENTO")
message(strrep("=", 60))

message(">>> Paso 5/8: Unión EPH + Canastas + IPC...")
source("src/03_merging.R")

message(">>> Paso 6/8: Análisis exploratorio...")
source("src/03b_exploratory_analysis.R")

message("✅ Fase 2 completada: Dataset unido y verificado.")

# ==============================================================================
# FASE 3: INGENIERÍA DE CARACTERÍSTICAS
# ==============================================================================
message("\n", strrep("=", 60))
message("FASE 3: INGENIERÍA DE CARACTERÍSTICAS")
message(strrep("=", 60))

message(">>> Paso 7/8: Feature Engineering + MPI Alkire-Foster...")
source("src/04_feature_engineering.R")

message("✅ Fase 3: Variables de vulnerabilidad y Target MPI creados.")
message("MCA proyectado correctamente sobre el test set")
# ==============================================================================
# FASE 4: SELECCIÓN DE CARACTERÍSTICAS (MCA)
# ==============================================================================
message("\n", strrep("=", 60))
message("FASE 4: SELECCIÓN DE CARACTERÍSTICAS (MCA)")
message(strrep("=", 60))

message(">>> Paso 8/8: MCA para reducción dimensional...")
source("src/05_mca_feature_selection.R")
source("src/05c_project_mca_simple.R")

message("✅ Fase 4: MCA finalizado.")

# ==============================================================================
# FASE 5: MODELADO (CONDICIONAL)
# ==============================================================================
message("\n", strrep("=", 60))
message("FASE 5: MODELADO")
message(strrep("=", 60))

if (all(artefactos_existentes)) {
  message("⏭️  SALTO DE MODELADO - Usando artefactos existentes:")
  message("   - fit_rf.rds, fit_xgb.rds, fit_cart.rds")
  message("   - best_rf.rds, best_xgb.rds, best_cart.rds")
  message("   Tiempo ahorrado: ~12-16 horas")
} else {
  message(">>> Ejecutando pipeline completo de modelado...")
  message("   ADVERTENCIA: Esto puede tardar 12-16 horas")
  source("src/06_modelling.R")
  message("✅ Fase 5: Modelos entrenados y evaluados.")
}

# ==============================================================================
# FASE 6: ANÁLISIS DE ROBUSTEZ Y SENSIBILIDAD
# ==============================================================================
message("\n", strrep("=", 60))
message("FASE 6: ANÁLISIS DE ROBUSTEZ Y SENSIBILIDAD")
message(strrep("=", 60))

message(">>> Análisis de robustez (sin max_instruccion_hogar)...")
source("src/06b_robustness.R")

message(">>> Validación de SMOTENC leakage...")
source("src/06c_smotenc_leakage.R")

message("✅ Fase 6: Análisis de robustez completado.")

# ==============================================================================
# FASE 7: EXPLICABILIDAD (XAI)
# ==============================================================================
message("\n", strrep("=", 60))
message("FASE 7: EXPLICABILIDAD (XAI)")
message(strrep("=", 60))

message(">>> XAI Regional (objetivo principal del TFM)...")
source("src/07_xai_regional.R")

message(">>> XAI de sensibilidad (modelos sin educación)...")
source("src/07b_xai_sensibilidad.R")

message(">>> XAI de robustez (modelos sin max_instruccion)...")
source("src/07b_xai_robustness.R")

message("✅ Fase 7: Análisis XAI completado.")

# ==============================================================================
# FASE 8: VISUALIZACIÓN Y GRÁFICOS
# ==============================================================================
message("\n", strrep("=", 60))
message("FASE 8: VISUALIZACIÓN Y GRÁFICOS")
message(strrep("=", 60))

message(">>> Gráficos principales para TFM...")
source("src/08_graphs.R")

message(">>> Gráficos adicionales...")
source("src/08b_graphs.R")

message(">>> Gráficos de sensibilidad...")
source("src/08c_graphs_sensitivity.R")

message("✅ Fase 8: Gráficos generados.")

# ==============================================================================
# CIERRE
# ==============================================================================
parallel::stopCluster(cl)

message("\n", strrep("=", 60))
message("RESUMEN DE EJECUCIÓN")
message(strrep("=", 60))

# Verificar outputs generados
outputs_figures <- list.files("output/figures/", recursive = TRUE, pattern = "\\.png$", full.names = FALSE)
outputs_tables <- list.files("output/tables/", recursive = TRUE, pattern = "\\.csv$", full.names = FALSE)
outputs_models <- list.files("output/models/", recursive = TRUE, pattern = "\\.rds$", full.names = FALSE)
outputs_results <- list.files("output/results/", recursive = TRUE, pattern = "\\.rds$", full.names = FALSE)

message("\n📊 ESTADÍSTICAS DE OUTPUT:")
message("   Figuras (.png):    ", length(outputs_figures))
message("   Tablas (.csv):     ", length(outputs_tables))
message("   Modelos (.rds):    ", length(outputs_models))
message("   Resultados (.rds): ", length(outputs_results))

if (all(artefactos_existentes)) {
  message("\n⏱️  MODO OPTIMIZADO: Pipeline completado sin reentrenar modelos.")
  message("   Tiempo total estimado: ~2-3 horas (vs 12-16 horas completo)")
} else {
  message("\n⏱️  MODO COMPLETO: Pipeline ejecutado íntegramente.")
  message("   Tiempo total estimado: ~12-16 horas")
}

message("\n🏁 Pipeline finalizado con éxito: ", Sys.time())
message("   El proyecto está listo para la redacción del TFM.")