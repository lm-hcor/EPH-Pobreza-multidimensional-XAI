# ==============================================================================
# PROYECTO: Análisis Multidimensional de la Pobreza (EPH Argentina)
# OBJETIVO: Script Maestro de Ejecución (Pipeline Automatizado)
# UBICACIÓN: Raíz del Proyecto
# AUTOR: Luis Miguel Herrera Corrales
# LICENCIA: Apache 2.0
# ==============================================================================

# 0. Configuración del Entorno --------------------------------------------
options(scipen = 999) # Evitar notación científica en los resultados
message(">>> Iniciando Pipeline Profesional EPH...")

# Carga de librerías esenciales
libs <- c("tidyverse", "data.table", "parallel")
invisible(lapply(libs, library, character.only = TRUE))

# 1. Ejecución Secuencial del Proyecto ------------------------------------

# 00: Funciones auxiliares y configuración global
source("src/00_utils.R")

# 01: Descarga de microdatos (o verificación de archivos locales)
source("src/01_download.R")

# 02: Procesamiento de IPC y Canastas (Deflactación regional)
source("src/02_clean_ipc.R")
source("src/02b_clean_canastas.R")

# 03: Unión de bases Individual y Hogar + Cruce con Canastas
source("src/03_merging.R")

# 04: Ingeniería de Variables y Segmentación Metodológica (HOY)
# Aquí es donde separamos 2016-2024 de 2025 y creamos los predictores
 source("src/04_feature_engineering.R")

# 05: Modelado de Machine Learning (PRÓXIMO PASO)
# source("src/05_modelado_ml.R")

# 2. Verificación Final ---------------------------------------------------

if (exists("eph_final")) {
  message(">>> [OK] Pipeline ejecutado hasta Step 04.")
  message(">>> Registros totales: ", nrow(eph_final))
  message(">>> Periodos detectados: ", paste(unique(eph_final$periodo), collapse = ", "))
} else {
  warning(">>> [ERROR] El dataset final no se encuentra en memoria.")
}

message(">>> Listo para el Step 05: Modelado de Machine Learning.")