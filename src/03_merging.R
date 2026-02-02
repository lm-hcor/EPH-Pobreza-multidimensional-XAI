# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 03_merging.R
# Propósito: Unir EPH (Hog + Ind), IPC y Canastas con Ajuste Regional
# ==============================================================================

library(tidyverse)

# 0. Carga de datos auxiliares procesados en pasos anteriores
ipc_trimestral <- readRDS("data/processed/ipc_trimestral.rds")
canastas_nacionales <- readRDS("data/processed/canastas_nacionales.rds")

# 1. Listamos las rutas de bases raw
archivos <- list.files("data/raw", full.names = TRUE)

# ******************************************************************************
#                          Unión de bases INDIVIDUAL Y HOGAR
# ******************************************************************************
# 2. Carga y unión de bases INDIVIDUALES
message(">>> Cargando bases individuales...")
eph_indiv_all <- archivos[str_detect(archivos, "individual")] %>%
  map_df(~ {
    readRDS(.x) %>% mutate(across(everything(), as.character))
  }) %>%
  type_convert()

# 3. Carga y unión de bases HOGAR
message(">>> Cargando bases hogar...")
eph_hogar_all <- archivos[str_detect(archivos, "hogar")] %>%
  map_df(~ {
    readRDS(.x) %>% mutate(across(everything(), as.character))
  }) %>%
  type_convert()

# 4. Limpieza de duplicados antes del Merge
# Identificamos columnas repetidas (excepto las llaves) para evitar el caos de .X y .Y
col_repetidas <- intersect(names(eph_indiv_all), names(eph_hogar_all))
llaves <- c("CODUSU", "NRO_HOGAR", "ANO4", "TRIMESTRE", "REGION")
col_a_eliminar_hogar <- setdiff(col_repetidas, llaves)

# 5. Unión de ambas (Hogar + Individual)
eph_unida <- eph_indiv_all %>%
  left_join(eph_hogar_all %>% select(-all_of(col_a_eliminar_hogar)), 
            by = llaves)

# ******************************************************************************
# Bloque de Integración: IPC + Canastas + Regiones
# ******************************************************************************

# 6. Definición de Ponderadores Regionales (Ajuste de costo de vida)
# Nos aseguramos de que los nombres de las llaves estén en MAYÚSCULAS
ponderadores_regiones <- data.frame(
  REGION = c(1, 40, 41, 42, 43, 44),
  REGION_LABEL = c("GBA", "Noroeste", "Noreste", "Cuyo", "Pampeana", "Patagonia"),
  COEF_REGIONAL = c(1.00, 0.91, 0.92, 0.95, 1.02, 1.22)
)

# 7. Unión Final, Deflactación y Etiquetado
eph_final <- eph_unida %>%
  # PASO CLAVE: Todo a mayúsculas antes de empezar los joins
  rename_with(toupper) %>% 
  
  # Unimos con Ponderadores (REGION ya está en mayúsculas en ambos)
  left_join(ponderadores_regiones, by = "REGION") %>%
  
  # Unimos con IPC (asegurando que las llaves del IPC también coincidan)
  left_join(ipc_trimestral %>% rename_with(toupper), 
            by = c("ANO4", "TRIMESTRE", "REGION_LABEL")) %>%
  
  # Unimos con Canastas (asegurando mayúsculas en las llaves)
  left_join(canastas_nacionales %>% rename_with(toupper), 
            by = c("ANO4", "TRIMESTRE")) %>%
  
  mutate(
    # 1. Ingresos Nominales y Reales
    # Usamos ITF que ya está en mayúsculas por el rename_with anterior
    ingreso_hogar_nom = as.numeric(ITF),
    p21_real = (as.numeric(P21) / VALOR_IPC) * 100,
    itcf_real = (ingreso_hogar_nom / VALOR_IPC) * 100,
    
    # 2. Ajuste Regional de Canastas
    cbt_regional = CBT_NACIONAL * COEF_REGIONAL,
    cba_regional = CBA_NACIONAL * COEF_REGIONAL,
    
    # 3. Creación de Variable Target
    es_pobre = ifelse(ingreso_hogar_nom < cbt_regional, 1, 0),
    es_indigente = ifelse(ingreso_hogar_nom < cba_regional, 1, 0),
    
    # 4. Marca de tiempo
    periodo = as.Date(paste0(ANO4, "-", (as.numeric(TRIMESTRE)*3-2), "-01"))
  ) %>%
  # Filtramos registros vitales
  filter(!is.na(VALOR_IPC), !is.na(itcf_real), !is.na(es_pobre))

# ******************************************************************************
#                      Guardado y chequeo
# ******************************************************************************
# 8. Guardar el trabajo finalizado
saveRDS(eph_final, "data/processed/eph_final.rds")

message("¡Felicidades! Step 03 completado con éxito.")
message("Registros totales: ", nrow(eph_final))
message("Variables listas para el Step 04: itcf_real, es_pobre, cbt_regional")

# Verificación rápida
eph_final %>% 
  group_by(REGION_LABEL) %>% 
  summarise(pobreza_promedio = mean(es_pobre)) %>% 
  print()