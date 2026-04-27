# ==============================================================================
# Proyecto: Pobreza Multidimensional Argentina
# Propósito: Ejecución INTEGRAL basada en la estructura de archivos src/
# Autor: Luis Miguel Herrera Corrales
# Licencia: Apache 2.0
# ==============================================================================

library(tidyverse)
library(tidymodels)
library(doParallel)

# 0. OPTIMIZACIÓN DE RECURSOS
all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)

message("🚀 Iniciando Pipeline TOTAL: ", Sys.time())

# --- FASE 1: PREPARACIÓN Y LIMPIEZA ---
source("src/00_utils.R")
source("src/01_download.R")       # Carga de microdatos EPH
source("src/02_clean_ipc.R")      # Índices de precios
source("src/02b_clean_canastas.R") # Canastas regionales CBA/CBT
message("✅ Fase 1 completada: Datos base y canastas listos.")

# --- FASE 2: PROCESAMIENTO ---
source("src/03_merging.R")        # Unión EPH + Canastas + IPC
source("src/03b_exploratory_analysis.R") # Verificación de distribuciones
message("✅ Fase 2 completada: Dataset unido y verificado.")

# --- FASE 3: INGENIERÍA Y MODELADO ---
# Aquí se crean las nuevas variables de vulnerabilidad
source("src/04_feature_engineering.R") 
message("✅ Fase 3: Variables de vulnerabilidad y Target MPI creados.")

# Cálculo de dimensiones estructurales
source("src/05_mca_feature_selection.R")
message("✅ Fase 4: MCA finalizado.")

# Entrenamiento de modelos (CART, RF, XGBoost) con proyección real al Test
source("src/06b_optimized_modelling.R")
message("✅ Fase 5: Modelos entrenados y evaluados.")

# --- FASE 4: EXPLICABILIDAD (XAI) ---
source("src/07_xai.R")
message("✅ Fase 6: Análisis SHAP y explicabilidad finalizados.")

# --- CIERRE ---
stopImplicitCluster()

if(exists("tabla_resultados")) {
  message("\n📊 RENDIMIENTO DEL MODELO (Test Set):")
  print(tabla_resultados)
}

message("\n🏁 Pipeline finalizado con éxito: ", Sys.time())
message("Puedes proceder a cargar los .rds en la App Shiny.")