# ==============================================================================
# Proyecto: Pobreza Multidimensional en Argentina (ML + XAI)
# Script: 03b_eda_distribuciones.R
# Propósito: Análisis exploratorio sustentado
# ==============================================================================

library(tidyverse)
library(patchwork)
library(scales)
library(e1071) 

message(">>> Iniciando Step 03b: Análisis descriptivo con variables reales...")

# 1. Carga de datos
# ------------------------------------------------------------------------------
eph <- readRDS("data/processed/eph_final.rds")

# 2. Diagnóstico del Ingreso Real (itcf_real)
# ------------------------------------------------------------------------------
# Calculamos asimetría (Skewness)
skew_orig <- round(skewness(eph$itcf_real, na.rm = TRUE), 2)

# Filtramos para el logaritmo (solo valores > 0)
ingresos_pos <- eph$itcf_real[eph$itcf_real > 0 & !is.na(eph$itcf_real)]
skew_log <- round(skewness(log10(ingresos_pos), na.rm = TRUE), 2)

# Gráfico 1: Distribución Real (Natural)
p1 <- eph %>%
  filter(itcf_real > 0) %>%
  ggplot(aes(x = itcf_real)) +
  geom_density(fill = "steelblue", alpha = 0.6) +
  scale_x_continuous(labels = label_dollar(), 
                     limits = c(0, quantile(eph$itcf_real, 0.98, na.rm = TRUE))) +
  labs(title = "Distribución del Ingreso Total Familiar Real",
       subtitle = paste0("Asimetría (Skewness): ", skew_orig),
       x = "ITCF Real (Deflactado)", y = "Densidad") +
  theme_minimal()

# Gráfico 2: Distribución Logarítmica
p2 <- data.frame(itcf_log = log10(ingresos_pos)) %>%
  ggplot(aes(x = itcf_log)) +
  geom_density(fill = "darkgreen", alpha = 0.6) +
  labs(title = "Distribución Log10(Ingreso Real)",
       subtitle = paste0("Asimetría Post-Transformación: ", skew_log),
       x = "Log10(ITCF Real)", y = "Densidad") +
  theme_minimal()

# 3. Análisis de Frecuencias de Vivienda (Sustento MCA)
# ------------------------------------------------------------------------------
# Revisamos IV3 (Material de pisos) como ejemplo de desbalance
p3 <- eph %>%
  count(iv3) %>% # Aquí la columna está en MAYÚSCULAS
  mutate(perc = n / sum(n)) %>%
  ggplot(aes(x = factor(iv3), y = perc)) +
  geom_col(fill = "coral", alpha = 0.8) +
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
  scale_y_continuous(labels = percent) +
  labs(title = "Frecuencia de Categorías (IV3 - Pisos)",
       subtitle = "Categorías bajo 5% requieren colapsamiento para estabilidad del MCA",
       x = "Código Categoría", y = "Porcentaje") +
  theme_minimal()

# 4. Correlación: itcf_real vs Pobreza Monetaria
# ------------------------------------------------------------------------------
# Esto demuestra el "vagueo" si usamos ambas
p4 <- eph %>%
  select(itcf_real, p21_real, es_pobre_mon) %>%
  mutate(es_pobre_mon = as.numeric(es_pobre_mon)) %>%
  drop_na() %>%
  cor() %>%
  as.data.frame() %>%
  rownames_to_column("v1") %>%
  pivot_longer(-v1) %>%
  ggplot(aes(v1, name, fill = value)) +
  geom_tile() +
  geom_text(aes(label = round(value, 2)), color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white") +
  theme_minimal() + labs(title = "Matriz de Correlación Real", x=NULL, y=NULL)

# 5. Guardado Final
# ------------------------------------------------------------------------------
layout_final <- (p1 / p2) | (p3 / p4)
ggsave("output/plots/diagnostico_final_sustento.png", layout_final, width = 14, height = 10)

message("✓ Análisis completado. Usando 'itcf_real'. Skewness original: ", skew_orig)