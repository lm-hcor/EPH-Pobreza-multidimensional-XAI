# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 03_merging.R
# Propósito: Unir EPH (Hogar + Individual), IPC y Canastas con ajuste regional
#
# NOTAS:
#   - clean_names() se aplica INMEDIATAMENTE al leer cada archivo crudo.
#     El paquete {eph} puede entregar nombres en mayúsculas o minúsculas según
#     la versión instalada. Normalizar antes del select() garantiza que
#     any_of(VARS_HOGAR) — que ahora está en minúsculas en 00_utils.R —
#     siempre encuentre las columnas de vivienda (iv3, iv4, iv6, etc.).
#   - Se añade un chequeo explícito post-carga para detectar columnas faltantes
#     críticas antes de que el pipeline falle en pasos posteriores.
#   - Se elimina el duplicado "II3" en VARS_HOGAR (estaba dos veces en v1).
# ==============================================================================

library(tidyverse)
library(janitor)       # clean_names()
source("src/00_utils.R")

# 0. Carga de datos auxiliares
# ------------------------------------------------------------------------------
ipc_trimestral      <- readRDS("data/processed/ipc_trimestral.rds")
canastas_nacionales <- readRDS("data/processed/canastas_nacionales.rds")

# Lista de archivos RDS descargados por 01_download.R
archivos <- list.files("data/raw", full.names = TRUE, pattern = "\\.rds$")

# 1. Carga y Cálculo de Adulto Equivalente (base Individual)
# ------------------------------------------------------------------------------
message(">>> Procesando base individual y adulto equivalente (ADEQ)...")

eph_indiv_all <- archivos[str_detect(archivos, "individual")] %>%
  map_df(function(f) {
    readRDS(f) %>%
      # CORRECCIÓN: normalizar nombres ANTES del select para que VARS_INDIVIDUAL
      # (en minúsculas) siempre encuentre las columnas correctamente.
      janitor::clean_names() %>%
      select(any_of(VARS_INDIVIDUAL)) %>%
      mutate(across(everything(), as.character))
  }) %>%
  type_convert()

# Calcular coeficiente de adulto equivalente por persona y luego sumar por hogar
adeq_hogar <- eph_indiv_all %>%
  mutate(adeq_i = calcular_adeq_individual(ch04, ch06)) %>%
  group_by(codusu, nro_hogar, ano4, trimestre) %>%
  summarise(adeq_hogar = sum(adeq_i, na.rm = TRUE), .groups = "drop")

message("  ADEQ calculado para ", nrow(adeq_hogar), " hogares.")

# 2. Carga de base Hogar
# ------------------------------------------------------------------------------
message(">>> Procesando base hogar...")

eph_hogar_all <- archivos[str_detect(archivos, "hogar")] %>%
  map_df(function(f) {
    readRDS(f) %>%
      # CORRECCIÓN: normalizar nombres ANTES del select.
      janitor::clean_names() %>%
      select(any_of(VARS_HOGAR)) %>%
      mutate(across(everything(), as.character))
  }) %>%
  type_convert()

# Verificación temprana: columnas de vivienda críticas para el MPI
vars_vivienda_criticas <- c("iv3", "iv4", "iv6", "iv11", "iv12_1", "ii1", "ix_tot")
faltantes_hogar <- setdiff(vars_vivienda_criticas, names(eph_hogar_all))
if (length(faltantes_hogar) > 0) {
  stop(
    "ERROR en 03_merging: las siguientes columnas de vivienda no fueron ",
    "encontradas en los archivos crudos de hogar: ",
    paste(faltantes_hogar, collapse = ", "),
    "\n  → Verifica que VARS_HOGAR en 00_utils.R coincide con los nombres ",
    "que entrega get_microdata() en tu versión del paquete {eph}."
  )
}
message("  Columnas de vivienda críticas presentes: OK")

# 3. Merge Individual + Hogar y estandarización de llaves
# ------------------------------------------------------------------------------
# Notar que pondera viene de ambas bases; renombramos la del hogar para
# conservar ambas si fuera necesario y evitar ambigüedad en joins posteriores.
llaves <- c("codusu", "nro_hogar", "ano4", "trimestre", "region", "aglomerado")

eph_unida <- eph_indiv_all %>%
  left_join(
    eph_hogar_all %>%
      rename(pondera_hogar = pondera) %>%   # renombrar para evitar colisión
      select(all_of(llaves), pondera_hogar, any_of(VARS_HOGAR)),
    by = llaves
  ) %>%
  mutate(
    ano4      = as.numeric(ano4),
    trimestre = as.numeric(trimestre)
  )

message("  Filas tras merge individual-hogar: ", nrow(eph_unida))

# 4. Ponderadores Regionales
# ------------------------------------------------------------------------------
ponderadores_regiones <- tibble(
  region        = c(1L, 40L, 41L, 42L, 43L, 44L),
  region_label  = c("GBA", "Noroeste", "Nordeste", "Cuyo", "Pampeana", "Patagonia"),
  coef_regional = c(1.00, 0.91, 0.92, 0.95, 1.02, 1.22)
)

# 5. Preparación de tablas auxiliares para los joins
# ------------------------------------------------------------------------------

# IPC trimestral: una fila por region_label + ano4 + trimestre
ipc_limpio <- ipc_trimestral %>%
  janitor::clean_names() %>%
  mutate(
    ano4      = as.numeric(ano4),
    trimestre = as.numeric(trimestre)
  )
# Columnas resultantes: ano4, trimestre, region_label, valor_ipc

# Canastas: join por region (numérico) + ano4 + trimestre
# Excluimos region_label de canastas para evitar sufijos _x/_y al joinear;
# region_label vendrá de ponderadores_regiones (fuente canónica).
canastas_limpio <- canastas_nacionales %>%
  janitor::clean_names() %>%
  mutate(
    ano4      = as.numeric(ano4),
    trimestre = as.numeric(trimestre),
    region    = as.numeric(region)
  ) %>%
  select(ano4, trimestre, region, cba_regional, cbt_regional)

# 6. Integración Final
# ------------------------------------------------------------------------------
message(">>> Integrando IPC, Canastas y ADEQ...")

eph_final <- eph_unida %>%
  # Join 1: ponderadores → añade region_label y coef_regional
  left_join(ponderadores_regiones, by = "region") %>%
  
  # Join 2: IPC → por ano4 + trimestre + region_label
  left_join(ipc_limpio, by = c("ano4", "trimestre", "region_label")) %>%
  
  # Join 3: canastas → por ano4 + trimestre + region (sin many-to-many)
  left_join(canastas_limpio, by = c("ano4", "trimestre", "region")) %>%
  
  # Join 4: adulto equivalente del hogar
  left_join(adeq_hogar, by = c("codusu", "nro_hogar", "ano4", "trimestre")) %>%
  
  mutate(
    # Guardia contra NA y división por cero en el IPC
    valor_ipc    = if_else(is.na(valor_ipc) | valor_ipc <= 0, 100, valor_ipc),
    cbt_regional = as.numeric(cbt_regional),
    cba_regional = as.numeric(cba_regional),
    itf          = as.numeric(itf),
    adeq_hogar   = if_else(is.na(adeq_hogar) | adeq_hogar <= 0, 1, adeq_hogar),
    
    # Canasta nominal del hogar = canasta por AE × coeficiente regional × AE del hogar
    cbt_nominal_hogar = cbt_regional * coef_regional * adeq_hogar,
    cba_nominal_hogar = cba_regional * coef_regional * adeq_hogar,
    
    # Pobreza monetaria (comparación nominal para evitar doble deflación)
    es_pobre_mon     = as.integer(itf < cbt_nominal_hogar),
    es_indigente_mon = as.integer(itf < cba_nominal_hogar),
    
    # Deflactor e ingresos reales (base = fecha_base_ipc, valor_ipc = 100 en esa fecha)
    deflactor    = valor_ipc / 100,
    itcf_real    = itf / deflactor,
    cbt_real_reg = cbt_nominal_hogar / deflactor,
    p21_real     = as.numeric(p21) / deflactor,
    
    # Período como fecha para posibles análisis de series temporales
    periodo = as.Date(paste0(ano4, "-", (as.numeric(trimestre) * 3 - 2), "-01"))
  ) %>%
  # Filtrar registros sin peso muestral (no representativos)
  filter(!is.na(pondera))

message("  Filas en eph_final: ", nrow(eph_final))

# 7. Guardado
# ------------------------------------------------------------------------------
saveRDS(eph_final, "data/processed/eph_final.rds")

# ==============================================================================
# VERIFICACIONES DE CONSISTENCIA
# ==============================================================================
message("\n--- REPORTES DE CONSISTENCIA ---")

message("\nCBT promedio por año (debe crecer con la inflación):")
print(
  eph_final %>%
    group_by(ano4) %>%
    summarise(cbt_promedio = mean(cbt_regional, na.rm = TRUE)) %>%
    head(10)
)

message("\nDistribución de Pobreza Monetaria (no debe ser todo 1):")
print(table(eph_final$es_pobre_mon, useNA = "ifany"))

message("\nResumen ITCF real vs Canasta real:")
print(summary(eph_final$itcf_real))
print(summary(eph_final$cbt_real_reg))

message("\nResumen Adulto Equivalente por hogar:")
print(summary(eph_final$adeq_hogar))

# Verificación de que las columnas de vivienda llegaron al dataset final
vars_check <- c("iv3", "iv4", "iv6", "iv11", "iv12_1", "ii1", "ix_tot")
presentes  <- intersect(vars_check, names(eph_final))
ausentes   <- setdiff(vars_check, names(eph_final))
message("\nVariables de vivienda en eph_final: ",
        length(presentes), "/", length(vars_check))
if (length(ausentes) > 0) {
  warning("ATENCIÓN: faltan columnas de vivienda en eph_final: ",
          paste(ausentes, collapse = ", "))
}

message("\n✅ Step 03 completado.")
