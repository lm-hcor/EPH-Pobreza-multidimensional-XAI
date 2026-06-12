# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 05b_project_mca_simple.R
# Propósito: Proyección del MCA a los test set y esterno (2024-2025)
# Usa el enfoque de coordenadas por ano en lugar de MCA con individuos suplementarios
# ==============================================================================

library(tidyverse)
library(FactoMineR)

message(">>> Iniciando proyeccion MCA simplificada...")

# Cargar datos
eph_train_mca <- readRDS("data/processed/eph_train_mca.rds")
vars_estables <- readRDS("data/processed/mca_vars_estables.rds")
mca_resultados <- readRDS("data/processed/mca_resultados_anuales.rds")

message("  Train MCA: ", nrow(eph_train_mca), " filas, ", length(vars_estables), " variables")

# Funcion simplificada: proyectar usando todos los datos de train
proyectar_mca_simple <- function(train_data, test_data, vars_mca) {
  
  message("    Proyeccion usando MCA entrenado en todos los anos...")
  
  # Preparar datos de train (todos los anos)
  train_input <- train_data %>%
    select(all_of(vars_mca)) %>%
    mutate(across(everything(), ~ factor(as.character(.))))
  
  # Eliminar columnas con un solo nivel
  train_input <- train_input[, sapply(train_input, nlevels) > 1, drop = FALSE]
  
  message("    Variables train: ", paste(names(train_input), collapse = ", "))
  
  # Preparar datos de test con LOS MISMOS niveles que train
  test_input <- test_data %>%
    select(all_of(vars_mca)) %>%
    mutate(across(everything(), ~ factor(as.character(.))))
  
  # Aplicar los mismos niveles que train
  for (var in names(train_input)) {
    if (var %in% names(test_input)) {
      niveles_train <- levels(train_input[[var]])
      test_input[[var]] <- factor(as.character(test_input[[var]]), levels = niveles_train)
    }
  }
  
  # Mantener solo columnas comunes
  cols_comunes <- intersect(names(test_input), names(train_input))
  test_input <- test_input %>% select(all_of(cols_comunes))
  
  # Eliminar columnas con un solo nivel
  test_input <- test_input[, sapply(test_input, nlevels) > 1, drop = FALSE]
  
  message("    Variables test: ", paste(names(test_input), collapse = ", "))
  
  # Combinar datos para MCA
  n_train <- nrow(train_input)
  n_test <- nrow(test_input)
  
  all_data <- bind_rows(
    train_input %>% mutate(.source = "train"),
    test_input %>% mutate(.source = "test")
  )
  
  all_data$.source <- NULL
  
  message("    Total datos: ", nrow(all_data), " (train: ", n_train, ", test: ", n_test, ")")
  
  # Ejecutar MCA con train como activos y test como suplementarios
  res_mca <- tryCatch({
    MCA(all_data, ncp = 2, graph = FALSE, ind.sup = (n_train + 1):(n_train + n_test))
  }, error = function(e) {
    message("    ERROR en MCA: ", e$message)
    return(NULL)
  })
  
  if (!is.null(res_mca)) {
    # Extraer coordenadas suplementarias
    if (!is.null(res_mca$ind.sup$coord) && nrow(res_mca$ind.sup$coord) > 0) {
      coords <- as.data.frame(res_mca$ind.sup$coord)
      names(coords) <- paste0("mca_dim", 1:ncol(coords))
      
      coords$codusu <- test_data$codusu
      coords$nro_hogar <- test_data$nro_hogar
      coords$ano4 <- test_data$ano4
      coords$trimestre <- test_data$trimestre
      
      message("    Proyeccion completada: ", nrow(coords), " filas")
      return(coords)
    } else {
      stop("No se pudieron extraer coordenadas suplementarias")
    }
  }
  
  return(NULL)
}

# Proyectar test 2024
message("\n>>> Proyectando test 2024...")
eph_test <- readRDS("data/processed/eph_test_ml.rds")

coords_test <- tryCatch({
  proyectar_mca_simple(eph_train_mca, eph_test, vars_estables)
}, error = function(e) {
  message("  ERROR en proyeccion test: ", e$message)
  return(NULL)
})

if (!is.null(coords_test)) {
  message("  ✅ Proyeccion test completada: ", nrow(coords_test), " filas")
  
  # Asignar coordenadas directamente
  if (nrow(coords_test) == nrow(eph_test)) {
    eph_test$mca_dim1 <- coords_test$mca_dim1
    eph_test$mca_dim2 <- coords_test$mca_dim2
    
    message("  'mca_dim1' presente: ", "mca_dim1" %in% names(eph_test))
    message("  'mca_dim2' presente: ", "mca_dim2" %in% names(eph_test))
    
    saveRDS(eph_test, "data/processed/eph_test_ml.rds")
    message("  ✅ Test data actualizado guardado")
  } else {
    message("  ❌ ERROR: Numero de filas no coincide")
  }
}

# Proyectar externo 2025
if (file.exists("data/processed/eph_externo_ml.rds")) {
  message("\n>>> Proyectando externo 2025...")
  eph_externo <- readRDS("data/processed/eph_externo_ml.rds")
  
  coords_externo <- tryCatch({
    proyectar_mca_simple(eph_train_mca, eph_externo, vars_estables)
  }, error = function(e) {
    message("  ERROR en proyeccion externo: ", e$message)
    return(NULL)
  })
  
  if (!is.null(coords_externo)) {
    message("  ✅ Proyeccion externo completada: ", nrow(coords_externo), " filas")
    
    if (nrow(coords_externo) == nrow(eph_externo)) {
      eph_externo$mca_dim1 <- coords_externo$mca_dim1
      eph_externo$mca_dim2 <- coords_externo$mca_dim2
      
      message("  'mca_dim1' presente: ", "mca_dim1" %in% names(eph_externo))
      message("  'mca_dim2' presente: ", "mca_dim2" %in% names(eph_externo))
      
      saveRDS(eph_externo, "data/processed/eph_externo_ml.rds")
      message("  ✅ Externo data actualizado guardado")
    } else {
      message("  ❌ ERROR: Numero de filas no coincide")
    }
  }
}

message("\n>>> Proceso de proyeccion MCA completado")