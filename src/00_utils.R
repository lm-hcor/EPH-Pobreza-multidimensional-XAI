# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 00_utils.R
# Propósito: Definición de diccionarios, constantes y funciones auxiliares
# ==============================================================================

# 1. Diccionario de Regiones EPH
# ------------------------------------------------------------------------------
get_region_labels <- function() {
  c(
    "1"  = "GBA",
    "40" = "Noroeste",
    "41" = "Nordeste",
    "42" = "Cuyo",
    "43" = "Pampeana",
    "44" = "Patagonia"
  )
}

# 2. Configuración de periodos y rutas
# ------------------------------------------------------------------------------
fecha_base_ipc <- "2024-12-01" # Fecha para llevar pesos nominales a constantes
año_inicio    <- 2016
año_fin       <- 2025

# 3. Función para limpiar y preparar el dataset básico
# ------------------------------------------------------------------------------
# (Aquí iremos agregando funciones de limpieza específicas según sean necesarias)

cat("✓ Utilidades cargadas correctamente.\n")