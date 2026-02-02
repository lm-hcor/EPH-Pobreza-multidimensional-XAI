# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 02b_clean_canastas.R
# Propósito: Descarga, Limpieza e Integración de las CBA y CBT del INDEC 
#            (valores a promedio nacional de las canastas de pobreza)
# ==============================================================================

library(readxl)
library(tidyverse)
library(lubridate)

# 1. Leer el Excel (fuerza a que lea las columnas como texto para no perder nada)
canastas_raw <- read_excel("data/external/canastas_indec.xls", 
                           sheet = "CBA-CBT", 
                           skip = 4,
                           col_types = "text") # Leemos todo como texto primero

# 2. Limpieza y Conversión
canastas_clean <- canastas_raw %>%
  # Nos quedamos con las primeras 4 columnas y les damos nombre
  select(1:4) %>%
  setNames(c("mes_texto", "cba_nacional", "coef_engel", "cbt_nacional")) %>%
  # Limpieza de valores
  mutate(
    # Quitamos espacios, comas por puntos y forzamos a numérico
    cbt_nacional = as.numeric(gsub(",", ".", gsub("[^0-9,]", "", cbt_nacional))),
    cba_nacional = as.numeric(gsub(",", ".", gsub("[^0-9,]", "", cba_nacional)))
  ) %>%
  # Eliminamos filas que no tengan datos tras la conversión
  filter(!is.na(cbt_nacional)) %>%
  mutate(
    # Creamos la serie de fechas (ajustar fecha inicio si es necesario)
    fecha = seq(as.Date("2016-04-01"), by = "month", length.out = n()),
    ANO4 = year(fecha),
    TRIMESTRE = quarter(fecha)
  ) %>%
  # 3. Promedio Trimestral (Ahora sí funcionará)
  group_by(ANO4, TRIMESTRE) %>%
  summarise(
    cbt_nacional = mean(cbt_nacional, na.rm = TRUE),
    cba_nacional = mean(cba_nacional, na.rm = TRUE),
    .groups = "drop"
  )

# Guardar resultado
saveRDS(canastas_clean, "data/processed/canastas_nacionales.rds")
message(">>> Canastas procesadas correctamente sin warnings.")