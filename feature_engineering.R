# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 04_feature_engineering.R
# Propósito: Segmentación de periodos (2016-2024/2025) y creación de Target 
#            variable (Pobreza Multidimensional según Alkire-Foster)
# ==============================================================================

# 1. Segmentación de periodos (Blindaje Metodológico ante metodologías del INDEC)
message(">>> Segmentando universos: Entrenamiento (2016-2024) y Test 2025...")

eph_final <- eph_final %>%
  mutate(
    periodo = ifelse(ANO4 <= 2024, "Entrenamiento", "Validacion_2025")
  )

# Verificación de volúmenes para el reporte
# Útil documentar el impacto del cambio censal (Censo 2010 vs Censo 2022)
table(eph_final$universo, eph_final$ANO4)

# 2. Creación de Variable Target (Labeling) -------------------------------
message(">>> Calculando umbrales de pobreza y etiquetas...")

# Nota: Usamos el ingreso deflactado para que la comparación sea justa
# a través de los años (Pobreza en términos reales).

eph_final <- eph_final %>%
  mutate(
    # Etiqueta de Pobreza (ITF vs CBT)
    # Asumimos que canasta_cbt_regional ya está pegada o es el valor base 2016
    es_pobre = ifelse(ingreso_hogar_deflactado < canasta_cbt_regional, 1, 0),
    
    # Etiqueta de Indigencia (ITF vs CBA)
    es_indigente = ifelse(ingreso_hogar_deflactado < canasta_cba_regional, 1, 0)
  )

# 3. Ratios de Dependencia Familiar ---------------------------------------
message(">>> Generando predictores de estructura del hogar...")

eph_final <- eph_final %>%
  group_by(CODUSU, NRO_HOGAR, ANO4, TRIMESTRE) %>%
  mutate(
    tamano_hogar = n(),
    n_menores    = sum(CH06 < 18),
    n_ocupados   = sum(ESTADO == 1),
    
    # Ratio de Dependencia Económica: clave para predecir pobreza
    # Usamos +0.5 para evitar errores de división por cero en hogares sin ocupados
    ratio_dependencia = tamano_hogar / (n_ocupados + 0.5),
    
    # Variable de vulnerabilidad infantil (clave para perfiles etarios)
    presencia_menores = ifelse(n_menores > 0, 1, 0)
  ) %>%
  ungroup()

message(">>> Paso 3.1 completado: Periodos marcados y Target definido.")

# 4. Pobreza Estructural (Vivienda y Hacinamiento) ------------------------
message(">>> Generando variables de infraestructura y hacinamiento...")

eph_final <- eph_final %>%
  mutate(
    # Hacinamiento Crítico (IV8: cantidad de habitaciones para dormir)
    # Definición estándar: más de 3 personas por habitación
    personas_por_cuarto = tamano_hogar / IV8,
    hacinamiento_critico = ifelse(personas_por_cuarto > 3, 1, 0),
    
    # Calidad de los materiales (IV3: Piso)
    # 1 = Mosaico/madera/etc, 2 = Cemento/Ladrillo fijo, 3 = Tierra/Ladrillo suelto
    piso_precario = ifelse(IV3 %in% c(3, 4), 1, 0),
    
    # Acceso a servicios (IV6: Agua, IV7: Baño)
    sin_agua_red = ifelse(IV6 != 1, 1, 0),
    sin_bano_privado = ifelse(IV7 != 1, 1, 0)
  )

# 5. Guardado del Dataset Listo para Modelar ------------------------------
message(">>> Guardando dataset final con Features...")
saveRDS(eph_final, "data/eph_modelo_ready.rds")