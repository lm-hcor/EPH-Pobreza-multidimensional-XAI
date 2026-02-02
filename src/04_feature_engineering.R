# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 04_feature_engineering.R
# Propósito: Segmentación de periodos (2016-2024/2025) y creación de Target 
#            variable (Pobreza Multidimensional según Alkire-Foster)
# ==============================================================================

library(tidyverse)
library(janitor) # Librería clave para limpiar nombres

message(">>> Iniciando Step 04: Ingeniería de Variables (snake_case)...")

# 1. Cargar y estandarizar nombres a snake_case, para hacerlo robusto a cambios
# La función clean_names() de janitor convierte todo a minúsculas y pone guiones bajos
eph_final <- readRDS("data/processed/eph_final.rds") %>% 
  janitor::clean_names()

# 2. Creando variables a nivel hogar
message(">>> Creando variables a nivel hogar...")

eph_final <- eph_final %>%
  group_by(codusu, nro_hogar, ano4, trimestre) %>%
  mutate(
    # Demografía del hogar
    tamano_hogar      = n(),
    n_menores         = sum(ch06 < 18, na.rm = TRUE),
    n_ancianos        = sum(ch06 >= 65, na.rm = TRUE),
    
    # Educación y Trabajo
    # Convertimos a numérico por si vienen como caracteres
    max_instruccion   = max(as.numeric(nivel_ed), na.rm = TRUE),
    n_ocupados        = sum(estado == 1, na.rm = TRUE),
    
    # Ratio de dependencia: ¿Cuánta gente depende de los que trabajan?
    ratio_dependencia = tamano_hogar / (n_ocupados + 0.5) 
  ) %>%
  ungroup()

# 3. Selección de Features para el modelo
message(">>> Seleccionando variables para el modelo...")

eph_model_data <- eph_final %>%
  select(
    # Target e Identificadores
    es_pobre, ano4, trimestre, region_label,
    
    # Predictores creados
    tamano_hogar, n_menores, n_ancianos, max_instruccion, 
    n_ocupados, ratio_dependencia,
    
    # Variables originales de la EPH (ahora en minúsculas gracias a clean_names)
    # iv1: tipo vivienda, iv6: agua, iv11: cloacas, ii7: régimen tenencia
    any_of(c("iv1", "iv6", "iv11", "ii7", "ii1"))
  ) %>%
  # El target debe ser factor para clasificación en ML
  mutate(es_pobre = as.factor(es_pobre))

# 4. Segmentación Temporal (Entrenamiento vs Validación 2025)
message(">>> Segmentando universos: Entrenamiento y Test 2025...")

eph_train <- eph_model_data %>% filter(ano4 < 2025)
eph_test  <- eph_model_data %>% filter(ano4 == 2025)

# 5. Guardar datasets listos para la fase de Machine Learning
saveRDS(eph_train, "data/processed/eph_train_ml.rds")
saveRDS(eph_test,  "data/processed/eph_test_ml.rds")

message(">>> [OK] Step 04 finalizado con éxito (Formato snake_case).")
message(">>> Registros para entrenamiento: ", nrow(eph_train))
message(">>> Registros para validación 2025: ", nrow(eph_test))