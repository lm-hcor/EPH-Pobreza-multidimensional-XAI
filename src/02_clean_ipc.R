# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 02_clean_ipc.R
# Propósito: Extraer índices regionales del IPC nacional
#
# NOTAS:
#   1. Lectura explícita con encoding cp1252 (el archivo INDEC usa Windows-1252)
#   2. Localización de regiones mediante su línea de encabezado propia
#      ("Región GBA;", "Región Pampeana;", etc.) en vez de asumir posición fija
#   3. Los labels de región ahora coinciden EXACTAMENTE con los de 03_merging.R:
#      GBA, Pampeana, Noroeste, Noreste, Cuyo, Patagonia
#   4. La serie de fechas se extrae de la línea de encabezado de CADA bloque,
#      no de la primera ocurrencia global
# ==============================================================================

library(tidyverse)
library(lubridate)

# 1. Lectura del archivo con encoding correcto
# ------------------------------------------------------------------------------
lineas <- readLines(
  "data/external/ipc_regional_indec.csv",
  warn     = FALSE,
  encoding = "UTF-8"   # readLines lee bytes; iconv() abajo convierte si es necesario
)

# Si el sistema no detecta bien el encoding, forzar la conversión desde cp1252
lineas <- iconv(lineas, from = "CP1252", to = "UTF-8", sub = "byte")

# 2. Función auxiliar: convertir línea CSV con comas decimales en vector numérico
# ------------------------------------------------------------------------------
extraer_valores <- function(linea) {
  partes <- str_split(linea, ";")[[1]]
  # Mantener solo campos con formato numérico INDEC: "100,0", "1203,5", etc.
  # El campo [1] es la etiqueta ("Nivel general"), los siguientes son números
  valores_raw <- partes[-1]                          # descartar etiqueta
  valores_raw <- str_trim(valores_raw)
  # Reemplazar coma decimal por punto y convertir
  as.numeric(str_replace(valores_raw, ",", "."))
}

# 3. Mapeo región → línea de "Nivel general" correspondiente
# ------------------------------------------------------------------------------
# Cada bloque regional en el CSV tiene esta estructura:
#   Línea N:   "Región GBA;dic-16;ene-17;..."    ← encabezado de fechas del bloque
#   Línea N+2: "Nivel general y divisiones COICOP;..."
#   Línea N+4: "Nivel general;100,0;101,3;..."   ← serie que nos interesa
#
# Identificamos los encabezados de bloque y derivamos la posición del Nivel general

# Patrones de encabezado regional (orden de aparición en el archivo INDEC)
bloques <- list(
  list(patron = "^Total nacional;",  label = "Nacional"),
  list(patron = "^Región GBA;",      label = "GBA"),
  list(patron = "^Región Pampeana;", label = "Pampeana"),
  list(patron = "^Región Noroeste;", label = "Noroeste"),
  list(patron = "^Región Noreste;",  label = "Noreste"),
  list(patron = "^Región Cuyo;",     label = "Cuyo"),
  list(patron = "^Región Patagonia;",label = "Patagonia")
)

# 4. Función que extrae la serie trimestral para un bloque dado
# ------------------------------------------------------------------------------
extraer_bloque <- function(lineas, patron_cabecera, label_region) {
  
  # Línea de encabezado del bloque (contiene las fechas propias del bloque)
  idx_cab <- which(str_detect(lineas, patron_cabecera))[1]
  if (is.na(idx_cab)) {
    warning("No se encontró el bloque: ", label_region)
    return(NULL)
  }
  
  # Fechas: están en la misma línea de cabecera, desde la segunda posición
  fechas_raw <- str_split(lineas[idx_cab], ";")[[1]][-1]
  fechas_raw <- str_trim(fechas_raw)
  fechas_raw <- fechas_raw[str_detect(fechas_raw, "^[a-z]{3}-[0-9]{2}$")]
  
  meses_map <- c(
    "ene" = "01", "feb" = "02", "mar" = "03", "abr" = "04",
    "may" = "05", "jun" = "06", "jul" = "07", "ago" = "08",
    "sep" = "09", "oct" = "10", "nov" = "11", "dic" = "12"
  )
  
  fechas <- map_chr(str_split(fechas_raw, "-"), ~ {
    mes_num   <- meses_map[.x[1]]
    anio      <- paste0("20", .x[2])
    paste0(anio, "-", mes_num, "-01")
  }) |> as.Date()
  
  # "Nivel general;" es la primera línea de datos dentro del bloque
  # Buscamos la primera ocurrencia DESPUÉS de idx_cab
  idx_ng <- which(str_detect(lineas, "^Nivel general;")
                  & seq_along(lineas) > idx_cab)[1]
  
  if (is.na(idx_ng)) {
    warning("No se encontró 'Nivel general' para: ", label_region)
    return(NULL)
  }
  
  valores <- extraer_valores(lineas[idx_ng])
  
  # Alinear longitudes por si el archivo tiene asimetrías menores
  n <- min(length(fechas), length(valores))
  
  tibble(
    fecha        = fechas[1:n],
    valor_ipc    = valores[1:n],
    region_label = label_region
  )
}

# 5. Construir la tabla completa
# ------------------------------------------------------------------------------
ipc_mensual <- map_df(bloques, ~ extraer_bloque(lineas, .x$patron, .x$label))

# Verificación de cobertura
cat("Regiones procesadas:", paste(unique(ipc_mensual$region_label), collapse = ", "), "\n")
cat("Rango temporal:", format(min(ipc_mensual$fecha)), "→", format(max(ipc_mensual$fecha)), "\n")
cat("Registros totales:", nrow(ipc_mensual), "\n")

# 6. Agregar a nivel trimestral y filtrar Nacional
# ------------------------------------------------------------------------------
ipc_trimestral <- ipc_mensual |>
  filter(region_label != "Nacional") |>
  mutate(
    ANO4      = year(fecha),
    TRIMESTRE = quarter(fecha)
  ) |>
  group_by(ANO4, TRIMESTRE, region_label) |>
  summarise(valor_ipc = mean(valor_ipc, na.rm = TRUE), .groups = "drop")

# Verificación final
cat("\nRegistros trimestrales:", nrow(ipc_trimestral), "\n")
print(ipc_trimestral |> filter(ANO4 == 2020, TRIMESTRE == 1))

# 7. Guardar
# ------------------------------------------------------------------------------
saveRDS(ipc_trimestral, "data/processed/ipc_trimestral.rds")
cat("✓ ipc_trimestral guardado correctamente.\n")
