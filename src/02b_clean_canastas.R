# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 02b_clean_canastas.R
# Propósito: Descarga de CBA y CBT regionales via paquete {eph}
#
# ESTRUCTURA de get_poverty_lines(regional = TRUE):
#   - CBA : canasta básica alimentaria en pesos por adulto equivalente
#   - CBT : canasta básica total       en pesos por adulto equivalente
#           (NO es el ICE — el paquete ya devuelve el valor monetario final)
# ==============================================================================

library(eph)
library(tidyverse)

canastas_raw <- get_poverty_lines(regional = TRUE)

canastas_nacionales <- canastas_raw |>
  rename(REGION_LABEL = region, REGION = codigo) |>
  mutate(
    cba_regional = CBA,   # pesos por adulto equivalente — usar directamente
    cbt_regional = CBT,   # pesos por adulto equivalente — usar directamente
    ANO4         = as.integer(str_extract(periodo, "^[0-9]{4}")),
    TRIMESTRE    = as.integer(str_extract(periodo, "[0-9]$"))
  ) |>
  select(ANO4, TRIMESTRE, REGION, REGION_LABEL, cba_regional, cbt_regional)

# Sanity check: 2020 T1 debe dar CBA ≈ 5.000-7.000 y CBT ≈ 12.000-18.000 ARS
cat("Sanity check 2020 T1:\n")
canastas_nacionales |> filter(ANO4 == 2020, TRIMESTRE == 1) |> print()

saveRDS(canastas_nacionales, "data/processed/canastas_nacionales.rds")
message("✓ Canastas guardadas: ", nrow(canastas_nacionales), " filas.")
