# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 00_utils.R
# Propósito: Diccionarios, constantes y funciones auxiliares
# ==============================================================================

# 1. Diccionario de Regiones EPH
# ------------------------------------------------------------------------------
get_region_labels <- function() {
  c(
    "1"  = "GBA",
    "40" = "Noroeste",
    "41" = "Nordeste",
    "42" = "Cuyo",
    "43" = "Pampeana",
    "44" = "Patagonia"
  )
}

# 2. Configuración de periodos y rutas
# ------------------------------------------------------------------------------
fecha_base_ipc <- as.Date("2024-12-01") # Referencia para deflactar a pesos constantes
anio_inicio    <- 2016
anio_fin       <- 2025

# 3. Umbrales Alkire-Foster (AF)
# ------------------------------------------------------------------------------
# Ponderaciones para 7 indicadores distribuidos en 3 dimensiones con peso 1/3 c/u:
#   Vivienda (1/3)    = priv_piso (1/9) + priv_techo (1/9) + priv_hacinamiento (1/9)
#   Saneamiento (1/3) = priv_agua (1/6) + priv_cloaca (1/6)
#   Educación (1/3)   = priv_esc (1/6)  + priv_educ (1/6)

AF_PESOS <- list(
  priv_piso         = 1/9,
  priv_techo        = 1/9,
  priv_hacinamiento = 1/9,   # Suma Vivienda:     3/9 = 1/3 ✓
  priv_agua         = 1/6,
  priv_cloaca       = 1/6,   # Suma Saneamiento:  2/6 = 1/3 ✓
  priv_esc          = 1/6,
  priv_educ         = 1/6    # Suma Educación:    2/6 = 1/3 ✓
)

AF_UMBRAL_K <- 1/3  # Hogar "MPI-pobre" si su puntaje ponderado >= 33.3%

# 4. Columnas EPH utilizadas
# ------------------------------------------------------------------------------
# IMPORTANTE: nombres en MINÚSCULAS para ser robustos ante clean_names().
# any_of() en los scripts de lectura aplicará estos nombres tras normalizar.

VARS_HOGAR <- c(
  "codusu", "nro_hogar", "ano4", "trimestre", "region", "aglomerado",
  "pondera",
  "itf",        # ingreso total familiar
  
  # -- Vivienda --
  "iv1",        # tipo de vivienda
  "iv2",        # material de paredes
  "iv3",        # material de techo
  "iv4",        # material de piso
  "iv5",        # revestimiento de piso
  "iv6",        # agua: origen
  "iv7",        # agua: dentro/fuera
  "iv8",        # baño: tiene
  "iv9",        # baño: uso exclusivo
  "iv10",       # baño: descarga
  "iv11",       # desagüe del baño
  "iv12_1",     # desagüe a red cloacal  (1 = sí, 0 = no)
  "iv12_2",     # desagüe a cámara séptica
  "iv12_3",     # desagüe a pozo/hoyo
  
  # -- Hacinamiento y tenencia --
  "ii1",        # cantidad de ambientes totales
  "ii2",        # cantidad de habitaciones
  "ii3",        # habitaciones para dormir
  "ii7",        # régimen de tenencia
  "ii8",        # combustible para cocinar
  "ii9",        # baño: ubicación (dentro/fuera)
  
  # -- Servicios del hogar --
  "v1",         # combustible principal para cocinar
  "v2",         # tiene heladera
  "v5",         # tiene computadora
  "v6",         # tiene celular
  "v11",        # tiene internet
  "v12",        # tiene cable/satélite
  "v13",        # tiene auto/camioneta
  
  "ix_tot"      # total de personas en el hogar (para cálculo de hacinamiento)
)

VARS_INDIVIDUAL <- c(
  "codusu", "nro_hogar", "ano4", "trimestre", "region", "aglomerado",
  "componente",
  "ch03",       # relación con jefe de hogar (1 = jefe/a)
  "ch04",       # sexo
  "ch05",       # asistencia escolar
  "ch06",       # edad
  "ch10",       # cobertura de salud
  "ch13",       # completó el nivel educativo
  "ch14",       # años en el nivel educativo actual
  "nivel_ed",   # nivel educativo máximo alcanzado
  "estado",     # condición de actividad
  "cat_ocup",   # categoría ocupacional
  "pp07h",      # realiza aportes jubilatorios (1 = sí, 2 = no)
  "p21",        # ingreso de la ocupación principal
  "pondera"
)

# 5. Funciones auxiliares
# ------------------------------------------------------------------------------

#' construir_mpi()
#' Construye el MPI basado en Alkire-Foster (AF) con doble umbral.
#' @param df  Data frame a nivel hogar con variables de vivienda y educación
#'            codificadas (tras clean_names() y feature engineering).
#' @param k   Umbral de corte sobre el puntaje ponderado (default = 1/3).
#' @return    df con columnas nuevas:
#'              priv_piso, priv_techo, priv_hacinamiento, priv_agua,
#'              priv_cloaca, priv_esc, priv_educ,
#'              mpi_score (numérico [0,1]), mpi_pobre (factor).
# ------------------------------------------------------------------------------
construir_mpi <- function(df, k = AF_UMBRAL_K) {
  
  # Verificar que las columnas requeridas existen antes de operar
  vars_req <- c("iv3", "iv4", "iv6", "iv11", "iv12_1", "ii1", "ix_tot",
                "priv_asistencia_esc", "nivel_ed_jefe")
  faltantes <- setdiff(vars_req, names(df))
  if (length(faltantes) > 0) {
    stop("construir_mpi(): faltan columnas requeridas: ",
         paste(faltantes, collapse = ", "))
  }
  
  df %>%
    mutate(
      # ── DIMENSIÓN 1: Vivienda ──────────────────────────────────────────────
      
      # Piso inadecuado: iv3 == 3 (tierra/ladrillo suelto) o 4 (otro)
      priv_piso   = as.integer(as.numeric(iv3) %in% c(3, 4)),
      
      # Techo inadecuado: iv4 == 3 (chapa sin cielorraso) o 4 (otro precario)
      priv_techo  = as.integer(as.numeric(iv4) %in% c(3, 4)),
      
      # Hacinamiento crítico: más de 3 personas por ambiente habitable
      # ix_tot = personas totales; ii1 = ambientes (pmax evita división por 0)
      priv_hacinamiento = as.integer(
        (as.numeric(ix_tot) / pmax(as.numeric(ii1), 1)) > 3
      ),
      
      # ── DIMENSIÓN 2: Saneamiento ───────────────────────────────────────────
      
      # Agua insuficiente: origen distinto a red pública (iv6 > 1)
      priv_agua   = as.integer(as.numeric(iv6) > 1),
      
      # Sin cloaca: desagüe a cámara séptica/pozo (iv11 == 3)
      #             O sin conexión a red cloacal (iv12_1 == 0)
      priv_cloaca = as.integer(
        as.numeric(iv11) == 3 | as.numeric(iv12_1) == 0
      ),
      
      # ── DIMENSIÓN 3: Educación ─────────────────────────────────────────────
      
      # Niño/a en edad escolar sin asistencia
      priv_esc  = as.integer(priv_asistencia_esc == 1),
      
      # Jefe/a con nivel educativo insuficiente (primario incompleto o menos)
      priv_educ = as.integer(as.numeric(nivel_ed_jefe) <= 2),
      
      # ── Puntaje ponderado AF ───────────────────────────────────────────────
      mpi_score = (
        priv_piso         * AF_PESOS$priv_piso         +
          priv_techo        * AF_PESOS$priv_techo        +
          priv_hacinamiento * AF_PESOS$priv_hacinamiento +
          priv_agua         * AF_PESOS$priv_agua         +
          priv_cloaca       * AF_PESOS$priv_cloaca       +
          priv_esc          * AF_PESOS$priv_esc          +
          priv_educ         * AF_PESOS$priv_educ
      ),
      
      # ── Clasificación dual Alkire-Foster ───────────────────────────────────
      mpi_pobre = factor(
        as.integer(mpi_score >= k),
        levels = c(0L, 1L),
        labels = c("no_pobre", "pobre")
      )
    )
}

# ------------------------------------------------------------------------------
#' calcular_adeq_individual()
#' Devuelve el coeficiente de adulto equivalente (INDEC) según sexo y edad.
#' @param CH04  Vector numérico de sexo (1 = varón, 2 = mujer).
#' @param CH06  Vector numérico de edad en años.
#' @return      Vector numérico de coeficientes AE.
# ------------------------------------------------------------------------------
calcular_adeq_individual <- function(CH04, CH06) {
  s <- as.numeric(CH04)
  e <- as.numeric(CH06)
  
  dplyr::case_when(
    # Ambos sexos (primera infancia)
    e < 1  ~ 0.35,
    e == 1 ~ 0.37,
    e == 2 ~ 0.46,
    e == 3 ~ 0.51,
    e == 4 ~ 0.55,
    e == 5 ~ 0.60,
    e == 6 ~ 0.64,
    e == 7 ~ 0.66,
    e == 8 ~ 0.68,
    e == 9 ~ 0.69,
    
    # Varones (s == 1)
    s == 1 & e == 10 ~ 0.79,
    s == 1 & e == 11 ~ 0.82,
    s == 1 & e == 12 ~ 0.85,
    s == 1 & e == 13 ~ 0.90,
    s == 1 & e == 14 ~ 0.96,
    s == 1 & e == 15 ~ 1.00,
    s == 1 & e == 16 ~ 1.03,
    s == 1 & e == 17 ~ 1.04,
    s == 1 & e >= 18 & e <= 29 ~ 1.02,
    s == 1 & e >= 30 & e <= 45 ~ 1.00,
    s == 1 & e >= 46 & e <= 60 ~ 1.00,
    s == 1 & e >= 61 & e <= 75 ~ 0.83,
    s == 1 & e >  75            ~ 0.74,
    
    # Mujeres (s == 2)
    s == 2 & e == 10 ~ 0.70,
    s == 2 & e == 11 ~ 0.72,
    s == 2 & e == 12 ~ 0.74,
    s == 2 & e == 13 ~ 0.76,
    s == 2 & e == 14 ~ 0.76,
    s == 2 & e == 15 ~ 0.77,
    s == 2 & e == 16 ~ 0.77,
    s == 2 & e == 17 ~ 0.77,
    s == 2 & e >= 18 & e <= 29 ~ 0.76,
    s == 2 & e >= 30 & e <= 45 ~ 0.77,
    s == 2 & e >= 46 & e <= 60 ~ 0.76,
    s == 2 & e >= 61 & e <= 75 ~ 0.67,
    s == 2 & e >  75            ~ 0.63,
    
    # Valor por defecto para NAs o rangos no contemplados
    TRUE ~ 1.00
  )
}

cat("✓ Utilidades cargadas correctamente.\n")
