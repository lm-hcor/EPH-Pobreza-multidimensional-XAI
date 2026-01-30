# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 03_merging.R
# Propósito: Unir EPH Hogares e INdividual e IPC.
# ==============================================================================

# Cargamos ipc_trimestral.
ipc_trimestral <- readRDS("data/processed/ipc_trimestral.rds")

# ******************************************************************************
#                          Bloque EPH
# ******************************************************************************

library(tidyverse)

# 1. Listamos las rutas completas
archivos <- list.files("data/raw", full.names = TRUE)

# 2. Cargamos y unimos todas las bases INDIVIDUALES con limpieza de tipos
eph_indiv_all <- archivos[str_detect(archivos, "individual")] %>%
  map_df(~ {
    data <- readRDS(.x)
    # Convertimos todo a character para que el bind_rows no falle
    data %>% mutate(across(everything(), as.character))
  }) %>%
  # Una vez unidas, dejamos que R intente volver a poner números donde van
  type_convert() 

# 3. Hacemos lo mismo para HOGAR (por si acaso ocurre lo mismo)
eph_hogar_all <- archivos[str_detect(archivos, "hogar")] %>%
  map_df(~ {
    data <- readRDS(.x)
    data %>% mutate(across(everything(), as.character))
  }) %>%
  type_convert()

# 4. Unión de ambas (Hogar + Individual)
# Nota: Usamos solo las llaves necesarias para evitar duplicar columnas de tiempo/region
eph_unida <- eph_indiv_all %>%
  left_join(eph_hogar_all, by = c("CODUSU", "NRO_HOGAR", "ANO4", "TRIMESTRE", 
                                  "REGION"))

# ******************************************************************************
#                         Bloque IPC
# ******************************************************************************
# 6. Aseguramos que el IPC tenga el código REGION (basado en el paso anterior)
mapping_regiones <- data.frame(
  REGION = c(1, 40, 41, 42, 43, 44),
  region_label = c("GBA", "Noroeste", "Noreste", "Cuyo", "Pampeana", "Patagonia")
)

ipc_final <- ipc_trimestral %>%
  left_join(mapping_regiones, by = "region_label")

# ******************************************************************************
#                             EPH + IPC
# ******************************************************************************
# 7. Unión Final y Deflactación
eph_final <- eph_unida %>%
  # Pasamos todo a mayúsculas para unificar
  rename_with(toupper) %>% 
  left_join(ipc_final, by = c("ANO4", "TRIMESTRE", "REGION")) %>%
  mutate(
    # Usamos ITF.Y (la que viene de Hogares) o ITF.X como respaldo
    ingreso_hogar_nom = coalesce(as.numeric(ITF.Y), as.numeric(ITF.X)),
    P21 = as.numeric(P21)
  ) %>%
  mutate(
    p21_real = (P21 / valor_ipc) * 100,
    itcf_real = (ingreso_hogar_nom / valor_ipc) * 100
  ) %>%
  # Filtramos filas sin IPC o sin ingreso (no nos sirven para el modelo)
  filter(!is.na(valor_ipc), !is.na(itcf_real))

# 6. Guardar el trabajo.
saveRDS(eph_final, "data/processed/eph_final.rds")

message("¡Felicidades! Dataset unido y deflactado.")
print(paste("Registros totales:", nrow(eph_final)))

# ******************************************************************************
#                     Chequeo Merge y Deflactación
# ******************************************************************************

# Validación de rigor: Ingreso Real promedio por Región y Año
check_regiones <- eph_final %>%
  group_by(ANO4, REGION) %>%
  summarise(ingreso_medio_real = mean(itcf_real, na.rm = TRUE), .groups = "drop")
# Ver los primeros resultados
head(check_regiones)
