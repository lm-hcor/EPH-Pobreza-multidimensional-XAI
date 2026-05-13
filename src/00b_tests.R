# ==============================================================================
# Script: 00b_tests.R
# Propósito: Tests de invariantes del pipeline
# Uso: source("src/00b_tests.R") en cualquier punto del pipeline
# ==============================================================================

library(tidyverse)
source("src/00_utils.R")

test_ok  <- 0
test_fail <- 0

check <- function(condicion, mensaje_ok, mensaje_fail) {
  if (condicion) {
    message("  ✅ ", mensaje_ok)
    test_ok  <<- test_ok  + 1
  } else {
    message("  ❌ ", mensaje_fail)
    test_fail <<- test_fail + 1
  }
}

message(">>> Ejecutando tests de invariantes...\n")

# ------------------------------------------------------------------------------
# TEST 1: Pesos AF suman exactamente 1
# ------------------------------------------------------------------------------
message("--- Tests: Pesos Alkire-Foster ---")

suma_pesos <- sum(unlist(AF_PESOS))
check(
  abs(suma_pesos - 1) < 1e-10,
  paste("Pesos AF suman 1 (suma =", round(suma_pesos, 10), ")"),
  paste("Pesos AF NO suman 1 (suma =", suma_pesos, ")")
)

check(
  AF_UMBRAL_K == 1/3,
  "Umbral k = 1/3 correcto",
  paste("Umbral k incorrecto:", AF_UMBRAL_K)
)

# ------------------------------------------------------------------------------
# TEST 2: Variables del MPI no están en PREDICTORES
# ------------------------------------------------------------------------------
message("\n--- Tests: Ausencia de leakage ---")

if (exists("PREDICTORES")) {
  vars_leakage <- c("priv_piso", "priv_techo", "priv_hacinamiento",
                    "priv_agua", "priv_cloaca", "priv_esc", "priv_educ",
                    "mpi_score", "iv3", "iv4", "iv6", "iv11", "iv12_1")
  
  leakage_encontrado <- intersect(PREDICTORES, vars_leakage)
  
  check(
    length(leakage_encontrado) == 0,
    "Ninguna variable de leakage en PREDICTORES",
    paste("LEAKAGE DETECTADO:", paste(leakage_encontrado, collapse = ", "))
  )
} else {
  message("  ⚠️  PREDICTORES no definido — omitiendo test de leakage")
}

# ------------------------------------------------------------------------------
# TEST 3: Dataset de modelado
# ------------------------------------------------------------------------------
message("\n--- Tests: Integridad del dataset ---")

if (exists("data_model")) {
  
  check(
    nrow(data_model) > 100000,
    paste("data_model tiene", nrow(data_model), "filas (> 100K)"),
    paste("data_model demasiado pequeño:", nrow(data_model), "filas")
  )
  
  check(
    all(c("mpi_pobre", "pondera") %in% names(data_model)),
    "Columnas mpi_pobre y pondera presentes",
    "Faltan columnas críticas en data_model"
  )
  
  check(
    all(levels(data_model$mpi_pobre) == c("no_pobre", "pobre")),
    "Niveles de mpi_pobre correctos",
    paste("Niveles incorrectos:", paste(levels(data_model$mpi_pobre), collapse = ", "))
  )
  
  tasa_mpi <- mean(data_model$mpi_pobre == "pobre")
  check(
    tasa_mpi > 0.02 & tasa_mpi < 0.15,
    paste("Tasa MPI en rango plausible:", round(tasa_mpi * 100, 1), "%"),
    paste("Tasa MPI fuera de rango:", round(tasa_mpi * 100, 1), "%")
  )
  
  check(
    sum(is.na(data_model$mpi_pobre)) == 0,
    "Sin NAs en el target mpi_pobre",
    paste("NAs en target:", sum(is.na(data_model$mpi_pobre)))
  )
  
  check(
    all(data_model$pondera > 0, na.rm = TRUE),
    "Todos los ponderadores son positivos",
    "Hay ponderadores negativos o cero"
  )
}

# ------------------------------------------------------------------------------
# TEST 4: Rangos plausibles de variables clave
# ------------------------------------------------------------------------------
message("\n--- Tests: Rangos de variables ---")

if (exists("eph_train")) {
  
  check(
    all(eph_train$itcf_real >= 0, na.rm = TRUE),
    "itcf_real no tiene valores negativos",
    paste("itcf_real tiene", sum(eph_train$itcf_real < 0, na.rm = TRUE), "valores negativos")
  )
  
  check(
    all(eph_train$adeq_hogar > 0, na.rm = TRUE),
    "adeq_hogar siempre positivo",
    "adeq_hogar tiene valores <= 0"
  )
  
  check(
    all(eph_train$tamano_hogar >= 1, na.rm = TRUE),
    "tamano_hogar >= 1 en todos los hogares",
    "Hay hogares con tamano_hogar < 1"
  )
  
  check(
    all(eph_train$ano4 >= 2016 & eph_train$ano4 <= 2024, na.rm = TRUE),
    "ano4 en rango 2016-2024 en train",
    paste("ano4 fuera de rango en",
          sum(eph_train$ano4 < 2016 | eph_train$ano4 > 2024, na.rm = TRUE),
          "filas")
  )
  
  check(
    all(eph_train$trimestre %in% 1:4, na.rm = TRUE),
    "trimestre en rango 1-4",
    "trimestre fuera de rango 1-4"
  )
}

# ------------------------------------------------------------------------------
# TEST 5: MPI construido correctamente
# ------------------------------------------------------------------------------
message("\n--- Tests: Construcción del MPI ---")

if (exists("eph_train")) {
  
  check(
    all(eph_train$mpi_score >= 0 & eph_train$mpi_score <= 1, na.rm = TRUE),
    "mpi_score en rango [0, 1]",
    "mpi_score fuera de rango [0, 1]"
  )
  
  tasa_ponderada <- weighted.mean(
    eph_train$mpi_pobre == "pobre",
    eph_train$pondera, na.rm = TRUE
  )
  
  check(
    tasa_ponderada > 0.03 & tasa_ponderada < 0.10,
    paste("Tasa MPI ponderada plausible:", round(tasa_ponderada * 100, 1), "%"),
    paste("Tasa MPI ponderada fuera de rango:", round(tasa_ponderada * 100, 1), "%")
  )
}

# ------------------------------------------------------------------------------
# TEST 6: Umbrales calibrados
# ------------------------------------------------------------------------------
message("\n--- Tests: Umbrales calibrados ---")

if (exists("tabla_umbrales")) {
  
  check(
    nrow(tabla_umbrales) == 3,
    "tabla_umbrales tiene 3 modelos",
    paste("tabla_umbrales tiene", nrow(tabla_umbrales), "modelos (esperado: 3)")
  )
  
  check(
    all(tabla_umbrales$umbral > 0.10 & tabla_umbrales$umbral < 0.60),
    "Todos los umbrales en rango plausible [0.10, 0.60]",
    paste("Umbral fuera de rango:",
          tabla_umbrales %>%
            filter(umbral <= 0.10 | umbral >= 0.60) %>%
            pull(modelo) %>%
            paste(collapse = ", "))
  )
}

# ------------------------------------------------------------------------------
# RESUMEN FINAL
# ------------------------------------------------------------------------------
message("\n", strrep("=", 50))
message("RESUMEN: ", test_ok, " tests OK | ", test_fail, " tests FALLIDOS")
if (test_fail == 0) {
  message("✅ Todos los invariantes verificados correctamente.")
} else {
  message("❌ Hay ", test_fail, " test(s) fallido(s) — revisar antes de continuar.")
}
message(strrep("=", 50))