# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 01_download.R
# Propósito: Descarga automatizada de microdatos EPH (Individual y Hogar)
# ==============================================================================

library(eph)
library(tidyverse)

# Cargamos las utilidades (00_utils)
source("src/00_utils.R")

# 1. Definir los periodos a descargar
# ------------------------------------------------------------------------------
# Usamos las variables definidas en 00_utils
anios <- anio_inicio:anio_fin
trimestres <- 1:4

# 2. Función de descarga con control de existencia local
# ------------------------------------------------------------------------------
descargar_eph_segura <- function(anio, trimestre, tipo) {
  
  # Definimos la ruta del archivo
  archivo_nombre <- paste0("data/raw/eph_", tipo, "_", anio, "_T", trimestre, ".rds")
  
  # Si el archivo NO existe, lo descargamos
  if (!file.exists(archivo_nombre)) {
    message(paste("Iniciando descarga:", tipo, anio, "T", trimestre, "..."))
    
    # tryCatch evita que el script se detenga si un trimestre no existe en el servidor
    tryCatch({
      datos <- get_microdata(year = anio, trimester = trimestre, type = tipo)
      
      # Guardamos en formato RDS para ahorrar espacio y mantener tipos de datos
      saveRDS(datos, archivo_nombre)
      message(paste("✓ Guardado en:", archivo_nombre))
      
    }, error = function(e) {
      message(paste("! Aviso: El periodo", anio, "T", trimestre, "no está disponible o hubo un error de red."))
    })
    
  } else {
    # Si el archivo ya existe, saltamos la descarga
    message(paste("· El archivo", archivo_nombre, "ya existe. Omitiendo descarga."))
  }
}

# 3. Ejecución del proceso (Bucle anidado)
# ------------------------------------------------------------------------------
# Descargamos tanto bases Individuales (ingresos/educación) como de Hogares (vivienda)
for (a in anios) {
  for (t in trimestres) {
    descargar_eph_segura(a, t, "individual")
    descargar_eph_segura(a, t, "hogar")
  }
}

cat("\n==================================================\n")
cat("Proceso de descarga completado satisfactoriamente.\n")
cat("Revisa la carpeta 'data/raw/' para ver los archivos.\n")
cat("==================================================\n")