# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 02_clean_ipc.R
# Propósito: Extraer índices regionales del IPC nacional
# ==============================================================================
library(tidyverse)
library(lubridate)

# 1. Leer el contenido (asumiendo que el archivo se llama "ipc_regional_indec.csv")
# Si lo tienes en una variable de texto, usa: lineas <- readLines(textConnection(tu_variable_de_texto))
lineas <- readLines("data/external/ipc_regional_indec.csv", warn = FALSE, encoding = "UTF-8")

# 2. Función para convertir una línea de INDEC en un vector numérico limpio
extraer_valores <- function(linea) {
  partes <- unlist(strsplit(linea, ";"))
  # Buscamos elementos que tengan el formato numérico (ej: "101,6" o "1203,0")
  # Reemplazamos coma por punto para que R los entienda como números
  valores <- partes[str_detect(partes, "[0-9],[0-9]")]
  valores <- as.numeric(str_replace(valores, ",", "."))
  return(valores)
}

# 3. Extraer las fechas (están en la línea 5 para Nacional y se repiten)
# Buscamos la línea que contiene los nombres de los meses
linea_fechas <- lineas[str_detect(lineas, "dic-16;ene-17")][1]
fechas_raw <- unlist(strsplit(linea_fechas, ";"))
fechas_raw <- fechas_raw[str_detect(fechas_raw, "-[0-9]{2}")] # Filtra dic-16, etc.

# Convertir meses de texto a fechas reales
meses_map <- c("ene"="01", "feb"="02", "mar"="03", "abr"="04", "may"="05", "jun"="06", 
               "jul"="07", "ago"="08", "sep"="09", "oct"="10", "nov"="11", "dic"="12")

fechas_list <- strsplit(fechas_raw, "-")
fechas_finales <- map_chr(fechas_list, ~ {
  mes_num <- meses_map[[.x[1]]]
  anio_completo <- ifelse(as.numeric(.x[2]) > 80, paste0("19", .x[2]), paste0("20", .x[2]))
  paste0(anio_completo, "-", mes_num, "-01")
}) %>% as.Date()

# 4. Localizar las líneas de "Nivel general" para cada región
# En tu archivo, el Nivel General de cada región aparece después del nombre de la región
indices_nivel_gral <- which(str_detect(lineas, "^Nivel general;"))

# Nombres de las regiones en el orden que aparecen en el archivo
nombres_regiones <- c("Nacional", "GBA", "Pampeana", "Noroeste", "Noreste", "Cuyo", "Patagonia")

# 5. Construir el Data Frame
ipc_lista <- list()

# Solo iteramos por la cantidad de índices encontrados (normalmente 7)
for(i in 1:min(length(indices_nivel_gral), length(nombres_regiones))) {
  valores <- extraer_valores(lineas[indices_nivel_gral[i]])
  
  # Solo guardamos si la cantidad de valores coincide con las fechas
  n <- min(length(fechas_finales), length(valores))
  
  ipc_lista[[i]] <- data.frame(
    fecha = fechas_finales[1:n],
    valor_ipc = valores[1:n],
    region_label = nombres_regiones[i]
  )
}

ipc_trimestral <- bind_rows(ipc_lista) %>%
  filter(region_label != "Nacional") %>% # Filtramos la nacional para quedarnos con regionales
  mutate(ANO4 = year(fecha), 
         TRIMESTRE = quarter(fecha)) %>%
  group_by(ANO4, TRIMESTRE, region_label) %>%
  summarise(valor_ipc = mean(valor_ipc, na.rm = TRUE), .groups = "drop")

# Verificación final
print(paste("Registros procesados:", nrow(ipc_trimestral)))
head(ipc_trimestral)

# Guardar ipc_trimestral en carpeta data/processed
saveRDS(ipc_trimestral, "data/processed/ipc_trimestral.rds")