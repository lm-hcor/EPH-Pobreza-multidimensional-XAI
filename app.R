library(shiny)
library(stringr)
library(DT)
library(dplyr)

fig_dir <- "output/figures"
reports_dir <- "output/reports"
notebooks_dir <- "notebooks"
src_dir <- "src"

if (!dir.exists(fig_dir)) {
  stop(paste(
    "No existe:", fig_dir,
    "\nDirectorio actual:", getwd(),
    "\nContenido:", paste(list.files("."), collapse = ", ")
  ))
}


addResourcePath("figures", fig_dir)

if (dir.exists(reports_dir))
  addResourcePath("reports", reports_dir)

if (dir.exists(notebooks_dir))
  addResourcePath("notebooks", notebooks_dir)

list_pngs <- function(dir_path) {
  files <- list.files(dir_path, recursive = TRUE, pattern = "\\.png$", full.names = FALSE)
  sort(files)
}

list_r_scripts <- function(dir_path) {
  files <- list.files(dir_path, pattern = "\\.R$", full.names = FALSE)
  sort(files)
}

figure_groups <- function(files) {
  groups <- list(
    "General" = character(0),
    "Tesis" = character(0),
    "Explicabilidad" = character(0),
    "Robustez" = character(0)
  )
  
  for (f in files) {
    lower <- tolower(f)
    if (str_detect(lower, "thesis/")) {
      groups[["Tesis"]] <- c(groups[["Tesis"]], f)
    } else if (str_detect(lower, "shap|breakdown|calibr|densidad|deciles|curva|comparativa|metricas|threshold|correlacion|privacion")) {
      groups[["Explicabilidad"]] <- c(groups[["Explicabilidad"]], f)
    } else if (str_detect(lower, "robust|sensitivity|comparativa|calibr")) {
      groups[["Robustez"]] <- c(groups[["Robustez"]], f)
    } else {
      groups[["General"]] <- c(groups[["General"]], f)
    }
  }
  
  groups[vapply(groups, length, integer(1)) > 0]
}

png_files <- list_pngs(fig_dir)
script_files <- list_r_scripts(src_dir)
figure_mapping <- figure_groups(png_files)

# Figuras específicas para la pestaña Resultados (métricas y evaluación de modelos)
resultados_pattern <- "^calibracion|^comparacion_|^comparativa_|^curva_|^deciles_|^f1_|^densidad_|^threshold_"
resultados_files <- sort(png_files[str_detect(png_files, resultados_pattern)])

model_metrics <- data.frame(
  Modelo = c("CART", "CART", "CART", "Random Forest", "Random Forest", "Random Forest", "XGBoost", "XGBoost", "XGBoost"),
  Conjunto = c("Externo 2025", "Test 2024", "Train", "Externo 2025", "Test 2024", "Train", "Externo 2025", "Test 2024", "Train"),
  Precision = c(0.1372369, 0.1622268, 0.8318877, 0.1901036, 0.2341768, 0.9403705, 0.1660230, 0.2084281, 0.9121884),
  Recall = c(0.07944922, 0.09809519, 0.78468244, 0.13727182, 0.17766550, 0.92270514, 0.10941334, 0.14722381, 0.88700520),
  AUC_ROC = c(0.6948438, 0.7110787, 0.9624492, 0.8378377, 0.8631544, 0.9968746, 0.8404838, 0.8652683, 0.9915128),
  Sensibilidad = c(0.5933260, 0.6231505, 0.7870895, 0.6922673, 0.7347521, 0.9247715, 0.7941372, 0.8538800, 0.8760259),
  Especificidad = c(0.7404462, 0.7417781, 0.9681351, 0.7942685, 0.8065059, 0.9872634, 0.7139868, 0.7298315, 0.9864655),
  stringsAsFactors = FALSE
)

metric_cards <- list(
  list(title = "Periodo analizado", value = "2016-2025", color = "#0d6efd"),
  list(title = "Modelos comparados", value = "CART, RF y XGBoost", color = "#198754"),
  list(title = "Mejor rendimiento", value = "XGBoost: AUC 0.865 (Test 2024) / 0.840 (Externo 2025)", color = "#fd7e14"),
  list(title = "Figuras disponibles", value = length(png_files), color = "#6f42c1")
)

pipeline_steps <- list(
  list(step = "1. Preparación", desc = "Descarga de microdatos EPH, limpieza de IPC y canastas"),
  list(step = "2. Procesamiento", desc = "Unión de bases, agregación a nivel hogar y análisis exploratorio"),
  list(step = "3. Ingeniería de features", desc = "Construcción del MPI y variables de vulnerabilidad"),
  list(step = "4. MCA y modelado", desc = "Reducción de dimensionalidad y ajuste de modelos"),
  list(step = "5. XAI y robustez", desc = "SHAP, break-down, sensibilidad y validación externa"),
  list(step = "6. Visualización", desc = "Gráficos de tesis, mapas y comparativas finales"),
  list(step = "7. Comunicación", desc = "Dashboard interactivo y reportes ejecutivos")
)

ui <- navbarPage(
  title = "EPH | Pobreza Multidimensional XAI",
  windowTitle = "EPH Pobreza Multidimensional XAI",
  collapsible = TRUE,
  fluid = TRUE,
  theme = NULL,
  header = tags$head(
    tags$link(rel = "stylesheet", href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"),
    tags$style(HTML("
            :root {
                --primary-dark: #0f4c81;
                --primary: #1e5a96;
                --accent: #2a9d8f;
                --accent-light: #48c9b0;
                --warning: #fd7e14;
                --danger: #dc3545;
                --success: #198754;
                --info: #0d6efd;
                --light-bg: #f8fafc;
                --border-color: #e0e7ff;
                --text-dark: #0f172a;
                --text-muted: #64748b;
            }

            * {
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }

            html, body {
                background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
                font-family: 'Segoe UI', 'Roboto', sans-serif;
                color: var(--text-dark);
                min-height: 100vh;
                margin: 0;
                padding: 0;
            }

            .navbar {
                background: linear-gradient(90deg, var(--primary-dark) 0%, var(--primary) 50%, var(--accent) 100%);
                border: none;
                box-shadow: 0 4px 20px rgba(15, 76, 129, 0.15);
                padding: 0.8rem 0;
            }

            .navbar-brand {
                font-weight: 700;
                font-size: 1.3rem;
                letter-spacing: -0.5px;
                color: white !important;
            }

            .nav-tabs {
                border-bottom: 2px solid var(--border-color);
            }

            .nav-link {
                color: var(--text-muted) !important;
                font-weight: 500;
                border-bottom: 3px solid transparent;
                transition: all 0.3s ease;
                padding: 0.75rem 1.25rem;
            }

            .nav-link:hover {
                color: var(--accent) !important;
                border-bottom-color: var(--accent-light) !important;
            }

            .nav-link.active {
                color: var(--accent) !important;
                border-bottom-color: var(--accent) !important;
                background-color: transparent;
            }

            .card-panel {
                background: white;
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 20px;
                box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
                border: 1px solid var(--border-color);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }

            .card-panel:hover {
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
                transform: translateY(-2px);
            }

            .metric-box {
                border-radius: 14px;
                padding: 20px;
                color: white;
                min-height: 130px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.12);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                transition: all 0.3s ease;
                border: none;
                position: relative;
                overflow: hidden;
            }

            .metric-box::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 3px;
                background: rgba(255, 255, 255, 0.3);
            }

            .metric-box:hover {
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.18);
                transform: translateY(-3px);
            }

            .metric-title {
                font-size: 0.9rem;
                opacity: 0.95;
                font-weight: 500;
                letter-spacing: 0.3px;
            }

            .metric-value {
                font-size: 1.5rem;
                font-weight: 700;
                margin-top: 8px;
                line-height: 1.2;
            }

            .metric-box-1 {
                background: linear-gradient(135deg, var(--info) 0%, #0a5ecf 100%);
            }

            .metric-box-2 {
                background: linear-gradient(135deg, var(--success) 0%, #157347 100%);
            }

            .metric-box-3 {
                background: linear-gradient(135deg, var(--warning) 0%, #e17c0f 100%);
            }

            .metric-box-4 {
                background: linear-gradient(135deg, #6f42c1 0%, #5a32a3 100%);
            }

            h1, h2, h3, h4, h5, h6 {
                color: var(--primary-dark);
                font-weight: 700;
                line-height: 1.3;
                margin-bottom: 12px;
            }

            h2 {
                font-size: 2rem;
            }

            h3 {
                font-size: 1.4rem;
                margin-bottom: 16px;
            }

            p {
                color: var(--text-muted);
                line-height: 1.6;
                margin-bottom: 12px;
            }

            strong {
                color: var(--text-dark);
                font-weight: 600;
            }

            .figure-frame {
                background: white;
                border-radius: 14px;
                padding: 12px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
                border: 1px solid var(--border-color);
            }

            .figure-frame img {
                width: 100%;
                height: auto;
                border-radius: 10px;
                display: block;
            }

            .step-item {
                border-left: 4px solid var(--accent);
                padding-left: 16px;
                margin-bottom: 14px;
                padding-bottom: 8px;
            }

            .step-item strong {
                font-size: 0.95rem;
                color: var(--primary-dark);
            }

            tags$ul, tags$li {
                color: var(--text-muted);
            }

            tags$li {
                margin-bottom: 8px;
                line-height: 1.6;
            }

            .dataTables_wrapper {
                font-size: 0.9rem;
            }

            .dataTables_info {
                color: var(--text-muted);
            }

            .dataTable {
                border-collapse: collapse;
                width: 100%;
            }

            .dataTable thead th {
                background: linear-gradient(90deg, var(--primary-dark) 0%, var(--primary) 100%);
                color: white;
                font-weight: 600;
                padding: 12px;
                text-align: left;
            }

            .dataTable tbody tr:hover {
                background-color: #f0f7ff;
            }

            .dataTable tbody td {
                padding: 11px;
                border-bottom: 1px solid var(--border-color);
            }

            .dataTable tbody tr:nth-child(even) {
                background-color: #fafbff;
            }

            .selectize-control {
                border-radius: 8px;
                border: 2px solid var(--border-color);
            }

            .selectize-control.focus {
                border-color: var(--accent);
                box-shadow: 0 0 0 3px rgba(42, 157, 143, 0.1);
            }

            .input-group {
                margin-bottom: 16px;
            }

            .form-control, .shiny-input-container .form-control {
                border-radius: 8px;
                border: 2px solid var(--border-color);
                padding: 10px 12px;
                font-size: 0.95rem;
                transition: all 0.3s ease;
            }

            .form-control:focus {
                border-color: var(--accent);
                box-shadow: 0 0 0 3px rgba(42, 157, 143, 0.1);
            }

            button.btn {
                border-radius: 8px;
                font-weight: 600;
                padding: 10px 20px;
                transition: all 0.3s ease;
                border: none;
                cursor: pointer;
            }

            .btn-primary {
                background: linear-gradient(135deg, var(--info) 0%, #0a5ecf 100%);
                color: white;
            }

            .btn-primary:hover {
                box-shadow: 0 4px 12px rgba(13, 110, 253, 0.3);
                transform: translateY(-2px);
            }

            .performance-badge {
                display: inline-block;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 0.85rem;
                font-weight: 600;
            }

            .badge-excellent {
                background: linear-gradient(135deg, var(--success) 0%, #157347 100%);
                color: white;
            }

            .badge-good {
                background: linear-gradient(135deg, var(--info) 0%, #0a5ecf 100%);
                color: white;
            }

            .badge-fair {
                background: linear-gradient(135deg, var(--warning) 0%, #e17c0f 100%);
                color: white;
            }

            .section-divider {
                height: 2px;
                background: linear-gradient(90deg, transparent, var(--border-color), transparent);
                margin: 32px 0;
            }

            .highlight-box {
                background: linear-gradient(135deg, #f0f7ff 0%, #e7f3ff 100%);
                border-left: 4px solid var(--accent);
                border-radius: 8px;
                padding: 16px;
                margin: 16px 0;
            }

            .highlight-box strong {
                color: var(--accent);
            }

            .container-fluid {
                padding-right: 30px;
                padding-left: 30px;
            }

            @media (max-width: 768px) {
                .navbar-brand {
                    font-size: 1rem;
                }
                
                .metric-box {
                    min-height: 100px;
                    margin-bottom: 12px;
                }
                
                .metric-value {
                    font-size: 1.2rem;
                }
                
                h2 {
                    font-size: 1.5rem;
                }
            }
        "))
  ),
  tabPanel(
    "Inicio",
    fluidRow(
      column(
        8,
        div(
          class = "card-panel",
          h2("📊 Dashboard Ejecutivo"),
          p("Análisis de pobreza multidimensional en Argentina mediante machine learning explicable (XAI). Esta plataforma integra el pipeline completo: desde la recolección de datos hasta la interpretabilidad del modelo."),
          div(
            class = "highlight-box",
            p(strong("🎯 Objetivo:"), "Identificar hogares vulnerables y sus determinantes estructurales para informar políticas públicas con precisión y transparencia.")
          )
        )
      ),
      column(
        4,
        div(
          class = "card-panel",
          h4("📋 Proyecto"),
          p(strong("Institución:"), "MUCSS - Universidad Carlos III de Madrid"),
          p(strong("Tipo:"), "Trabajo Fin de Máster"),
          p(strong("Período:"), "2016-2025"),
          p(strong("Región:"), "Argentina (Aglomerados EPH)")
        )
      )
    ),
    fluidRow(
      lapply(seq_along(metric_cards), function(i) {
        card <- metric_cards[[i]]
        column(
          3,
          div(
            class = paste0("metric-box metric-box-", i),
            div(class = "metric-title", card$title),
            div(class = "metric-value", card$value)
          )
        )
      })
    ),
    div(class = "section-divider"),
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h3("📈 Rendimiento de modelos"),
          p("Evaluación en tres conjuntos: train (2016-2023), test temporal (2024) y validación externa post-censal (2025)."),
          DTOutput("model_metrics_table_dt"),
          div(
            style = "margin-top: 20px;",
            img(src = "figures/comparacion_metricas_train_test_externo.png", style = "width: 100%; max-width: 750px; margin: 0 auto; display: block; border-radius: 12px;")
          )
        )
      )
    ),
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h3(" Hallazgos clave"),
          fluidRow(
            column(
              4,
              div(
                class = "highlight-box",
                strong("Mejor modelo:"), br(),
                p("XGBoost destaca en generalización con AUC 0.865 en test y 0.840 en externo.")
              )
            ),
            column(
              4,
              div(
                class = "highlight-box",
                strong("Robustez temporal:"), br(),
                p("La degradación controlada del rendimiento (Δ≈0.024 AUC) valida la robustez ante cambios de marco censal.")
              )
            ),
            column(
              4,
              div(
                class = "highlight-box",
                strong("Calibración:"), br(),
                p("Probabilidades de XGBoost alineadas con frecuencias empíricas para focalización precisa.")
              )
            )
          )
        )
      )
    ),
    div(class = "section-divider"),
    fluidRow(
      column(
        7,
        div(
          class = "card-panel",
          h3("⚙️ Pipeline del análisis"),
          lapply(seq_along(pipeline_steps), function(idx) {
            item <- pipeline_steps[[idx]]
            div(
              class = "step-item",
              strong(item$step), br(),
              p(item$desc, style = "margin: 6px 0 0 0; color: #64748b;")
            )
          })
        )
      ),
      column(
        5,
        div(
          class = "card-panel",
          h3("📦 Componentes del proyecto"),
          tags$ul(
            tags$li(
              strong("Scripts:"), " ",
              span(length(script_files), style = "background: #e0e7ff; padding: 2px 8px; border-radius: 4px; font-weight: 600; color: #0f4c81;"),
              " archivos en src/"
            ),
            tags$li(
              strong("Figuras:"), " ",
              span(length(png_files), style = "background: #e0e7ff; padding: 2px 8px; border-radius: 4px; font-weight: 600; color: #0f4c81;"),
              " gráficos en output/figures"
            ),
            tags$li(
              strong("Reportes:"), " Resumen ejecutivo y documento final (PDF)"
            ),
            tags$li(
              strong("Base de datos:"), " Microdatos EPH procesados y canastas nacionales"
            )
          ),
          div(
            class = "highlight-box",
            p(strong("💡 Nota:"), " Los datos crudos están excluidos por políticas de secreto estadístico del INDEC.")
          )
        )
      )
    )
  ),
  tabPanel(
    "Metodología",
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h2("🔬 Diseño metodológico"),
          p("El proyecto integra técnicas estadísticas rigurosas con machine learning y explicabilidad para construir un modelo robusto de predicción de pobreza multidimensional.")
        )
      )
    ),
    fluidRow(
      column(
        6,
        div(
          class = "card-panel",
          h3("📊 Construcción del índice MPI"),
          p("Se adaptó el método Alkire-Foster con tres dimensiones:"),
          tags$ul(
            tags$li(strong("Vivienda:"), " calidad de pisos, techos, hacinamiento"),
            tags$li(strong("Saneamiento:"), " acceso a agua, cloaca, basura"),
            tags$li(strong("Educación:"), " asistencia escolar, nivel educativo del jefe")
          ),
          p("Un hogar es pobre multidimensionalmente si su puntaje ponderado de privaciones (k=1/3) indica carencias severas.")
        )
      ),
      column(
        6,
        div(
          class = "card-panel",
          h3("🔄 Procesamiento de datos"),
          p("Pasos clave:"),
          tags$ul(
            tags$li(strong("Deflactación regional:"), " sincronización de ingresos con IPC desagregado"),
            tags$li(strong("Agregación por hogar:"), " consolidación desde base individual"),
            tags$li(strong("MCA:"), " reducción de dimensionalidad de variables categóricas"),
            tags$li(strong("Reponderación:"), " ajuste del marco muestral 2010 al 2022")
          )
        )
      )
    ),
    fluidRow(
      column(
        6,
        div(
          class = "card-panel",
          h3("🤖 Modelos comparados"),
          p("Tres algoritmos evaluados bajo el mismo protocolo:"),
          tags$ul(
            tags$li(strong("CART:"), " baseline interpretable, árboles de decisión"),
            tags$li(strong("Random Forest:"), " ensemble robusto, reducción de varianza"),
            tags$li(strong("XGBoost:"), " boosting gradual, máximo rendimiento")
          ),
          p("Ajuste de hiperparámetros mediante ANOVA Racing CV con validación por bloques geográficos.")
        )
      ),
      column(
        6,
        div(
          class = "card-panel",
          h3("✅ Validación y robustez"),
          p("Estrategia multi-capas:"),
          tags$ul(
            tags$li(strong("Block CV:"), " evita data leakage espacial por aglomerado"),
            tags$li(strong("Chronological split:"), " train 2016-2023, test 2024, externo 2025"),
            tags$li(strong("Sensibilidad:"), " evaluación sin variables educativas"),
            tags$li(strong("Calibración:"), " umbrales optimizados por F1-Score")
          )
        )
      )
    ),
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h3("📐 Análisis de explicabilidad (XAI)"),
          fluidRow(
            column(
              4,
              div(
                class = "highlight-box",
                strong("SHAP Global"), br(),
                p("Importancia promedio de variables a nivel población. Identifica qué predictores más explican las decisiones del modelo en aggregate.")
              )
            ),
            column(
              4,
              div(
                class = "highlight-box",
                strong("SHAP Regional"), br(),
                p("Análisis diferenciado por región geográfica. Revela heterogeneidad territorial en los drivers de pobreza.")
              )
            ),
            column(
              4,
              div(
                class = "highlight-box",
                strong("Break-down local"), br(),
                p("Descomposición secuencial de predicciones para hogares tipo. Muestra cómo cada variable contribuye a la clasificación individual.")
              )
            )
          )
        )
      )
    ),
    br()
  ),
  tabPanel(
    "Mapas y Análisis Temporal",
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h2("🗺️ Análisis geoespacial y temporal de la pobreza")
        )
      )
    ),
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h3("📍 Tasa de pobreza multidimensional por región"),
          p("Mapa de Argentina mostrando la distribución espacial del MPI. Las regiones se colorean según la prevalencia de pobreza multidimensional en el conjunto de entrenamiento (2016-2023)."),
          img(src = "figures/mapa_tasa_mpi.png", style = "width: 100%; max-width: 750px; margin: 0 auto; display: block; border-radius: 12px;")
        )
      )
    ),
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h3("📈 Evolución temporal del MPI por región"),
          p("Tendencias del Índice de Pobreza Multidimensional a través de los años, desagregadas por región geográfica. Permite identificar patrones de cambio y regiones particularmente vulnerables a shocks macroeconómicos."),
          img(src = "figures/evolucion_mpi_regional.png", style = "width: 100%; max-width: 750px; margin: 0 auto; display: block; border-radius: 12px;")
        )
      )
    ),
    div(class = "section-divider"),
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h3("💡 Interpretación de los mapas"),
          fluidRow(
            column(
              6,
              div(
                class = "highlight-box",
                strong("Mapa de tasas (arriba):")
              ),
              tags$ul(
                tags$li("Muestra la prevalencia de MPI en cada región."),
                tags$li("Colores más oscuros = mayor tasa de pobreza multidimensional."),
                tags$li("Permite focalizar recursos en regiones con mayor privación estructural.")
              )
            ),
            column(
              6,
              div(
                class = "highlight-box",
                strong("Evolución temporal (abajo):")
              ),
              tags$ul(
                tags$li("Líneas por región muestran cómo cambió el MPI a lo largo de 2016-2023."),
                tags$li("Algunos picos coinciden con crisis macroeconómicas conocidas (2018, 2020, 2023), pero no todos los máximos se alinean con shocks agregados a nivel nacional: varias regiones muestran picos en años sin un shock macro identificado, lo que sugiere la presencia de factores idiosincráticos, estacionales o de medición regional."),
                tags$li("Heterogeneidad regional: algunas zonas más resilientes que otras, y la relación pico-shock debe interpretarse con cautela sin un análisis formal de series temporales (p. ej., detección de puntos de quiebre).")
              )
            )
          )
        )
      )
    )
  ),
  tabPanel(
    "Resultados",
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h2("📊 Resultados de modelos"),
          p("Explora las figuras de evaluación de modelos: calibración, comparación de métricas, curvas y densidades de probabilidad.")
        )
      )
    ),
    fluidRow(
      column(
        3,
        div(
          class = "card-panel",
          h4("🔍 Seleccionar figura"),
          selectInput(
            "result_file",
            label = strong("Figura:"),
            choices = resultados_files,
            selected = resultados_files[1]
          ),
          div(
            class = "highlight-box",
            p("Estas figuras muestran el rendimiento predictivo: discriminación (ROC), calibración, curvas Precisión-Recall y densidades de probabilidad.")
          )
        )
      ),
      column(
        9,
        div(
          class = "card-panel",
          uiOutput("result_view"),
          p(style = "font-size: 0.85rem; color: #64748b; margin-top: 12px;",
            "Sugerencia: Abre en pestaña nueva para visualizar en mayor resolución.")
        )
      )
    )
  ),
  tabPanel(
    "Explicabilidad",
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h2("🔍 Análisis de Explicabilidad (XAI)"),
          p("La inteligencia artificial explicable permite abrir la 'caja negra' del modelo y entender qué variables generan cada predicción. Esto es crítico para ganar confianza en una herramienta destinada a políticas públicas.")
        )
      )
    ),
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h3("📍 Importancia de variables por región (SHAP)"),
          p("Cuáles son las variables que más explican la pobreza en cada región. Nótese que la heterogeneidad territorial es importante para política pública diferenciada."),
          img(src = "figures/mapa_shap_top1_region.png", style = "width: 100%; max-width: 700px; margin: 0 auto; display: block; border-radius: 12px;")
        )
      )
    ),
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h3("👥 Descomposición de casos tipo (Break-down)"),
          p("Cómo el modelo llega a su predicción para hogares específicos. Muestra el aporte secuencial de cada variable respecto al baseline promedio."),
          img(src = "figures/breakdown_panel_4_hogares_xgb.png", style = "width: 100%; max-width: 900px; margin: 0 auto; display: block; border-radius: 12px;"),
          div(
            class = "highlight-box",
            strong("Interpretación:"), br(),
            p("Las barras azules suben la probabilidad de pobreza, las rojas la bajan. La longitud indica magnitud de impacto. El baseline es la probabilidad media del modelo (~35% en test 2024).")
          )
        )
      )
    ),
    div(class = "section-divider"),
    fluidRow(
      column(
        6,
        div(
          class = "card-panel",
          h3("💡 Hallazgos principales"),
          tags$ul(
            tags$li(strong("Ingresos:"), " La variable más discriminativa. Hogares con menor ingreso familiar real (itcf_real) tienen mayor riesgo."),
            tags$li(strong("Educación:"), " Máximo nivel educativo del hogar explica significativamente el riesgo."),
            tags$li(strong("Infraestructura:"), " Acceso a servicios (agua, cloaca) y material de vivienda son predictores robustos."),
            tags$li(strong("Mercado de trabajo:"), " Proporción de informalidad y ratio de dependencia importan.")
          )
        )
      ),
      column(
        6,
        div(
          class = "card-panel",
          h3("🌍 Heterogeneidad regional"),
          p("Los drivers de pobreza no son uniformes:"),
          tags$ul(
            tags$li(strong("GBA (Buenos Aires):"), " Énfasis en ingresos y acceso a servicios."),
            tags$li(strong("Noroeste/Nordeste:"), " Infraestructura y educación más relevantes."),
            tags$li(strong("Patagonia:"), " Estructura demográfica y mercado laboral."),
            tags$li(strong("Cuyo/Pampeana:"), " Balance entre ingresos e infraestructura.")
          )
        )
      )
    )
  ),
  tabPanel(
    "Explorador de figuras",
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h2("🖼️ Galería completa de gráficos"),
          p("Explora todas las figuras generadas durante el análisis. La galería se organiza por categoría para facilitar la navegación.")
        )
      )
    ),
    fluidRow(
      column(
        2,
        div(
          class = "card-panel",
          h4("🔍 Filtros"),
          selectInput("figure_group", strong("Categoría:"), choices = names(figure_mapping), selected = names(figure_mapping)[1]),
          selectInput("figure_file", strong("Figura:"), choices = figure_mapping[[1]], selected = figure_mapping[[1]][1])
        )
      ),
      column(
        10,
        div(
          class = "card-panel",
          uiOutput("figure_view"),
          p(style = "font-size: 0.85rem; color: #64748b; margin-top: 12px;", 
            "Sugerencia: Abre en pestaña nueva para visualizar en mayor resolución.")
        )
      )
    ),
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h3("📂 Resumen de figuras disponibles"),
          p("Total de imágenes disponibles: ", strong(length(png_files))),
          fluidRow(
            lapply(names(figure_mapping), function(category) {
              count <- length(figure_mapping[[category]])
              column(
                3,
                div(
                  style = "background: #f0f7ff; padding: 12px; border-radius: 8px; text-align: center; margin-bottom: 12px;",
                  strong(category), br(),
                  span(style = "font-size: 1.3rem; font-weight: 700; color: #0f4c81;", count)
                )
              )
            })
          )
        )
      )
    )
  ),
  tabPanel(
    "Archivos del proyecto",
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h2("📁 Estructura del proyecto"),
          p("Acceso a todos los artefactos del análisis: scripts, documentación y reportes.")
        )
      )
    ),
    fluidRow(
      column(
        6,
        div(
          class = "card-panel",
          h3("🔧 Scripts del pipeline"),
          p("Módulos de R que implementan cada fase del análisis:"),
          tags$div(
            lapply(script_files, function(x) {
              tags$div(
                style = "padding: 8px; margin-bottom: 8px; background: #f0f7ff; border-left: 3px solid #2a9d8f; border-radius: 4px;",
                code(x)
              )
            })
          )
        )
      ),
      column(
        6,
        div(
          class = "card-panel",
          h3("📄 Reportes y documentación"),
          tags$ul(
            tags$li(
              a("📋 Resumen ejecutivo (PDF)", href = "reports/resumen_ejecutivo.pdf", target = "_blank", 
                style = "text-decoration: none; color: #2a9d8f; font-weight: 600;"),
              " - Vista rápida de hallazgos clave"
            ),
            tags$li(
              a("📚 Documento final TFM (PDF)", href = "reports/tfm_mucss.pdf", target = "_blank",
                style = "text-decoration: none; color: #2a9d8f; font-weight: 600;"),
              " - Tesis completa con metodología detallada"
            ),
            tags$li(
              a("📊 README.md", href = "https://github.com/lm-hcor/EPH-Pobreza-multidimensional-XAI#readme", target = "_blank",
                style = "text-decoration: none; color: #2a9d8f; font-weight: 600;"),
              " - Descripción del proyecto en Markdown"
            ),
            tags$li(
              a("🎨 Infografía interactiva (HTML)", href = "notebooks/infografia.html", target = "_blank",
                style = "text-decoration: none; color: #2a9d8f; font-weight: 600;"),
              " - Visualización ejecutiva del análisis"
            )
          )
        )
      )
    ),
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h3("🏗️ Arquitectura de directorios"),
          tags$pre(
            style = "background: #1a202c; color: #a0aec0; padding: 16px; border-radius: 8px; overflow-x: auto;",
            "EPH-Pobreza-multidimensional-XAI/
├── src/                    # Scripts del pipeline
│   ├── 00_utils.R         # Funciones auxiliares
│   ├── 01_download.R      # Descarga de microdatos EPH
│   ├── 02_clean_ipc.R     # Limpieza de IPC
│   ├── 03_merging.R       # Unión de bases
│   ├── 04_feature_engineering.R
│   ├── 05_mca_modelling.R
│   ├── 06_modelling.R     # Entrenamiento de modelos
│   ├── 07_xai.R           # Análisis de explicabilidad
│   └── 08_graphs.R        # Generación de gráficos
├── data/
│   ├── raw/               # Microdatos crudos (EPH)
│   └── processed/         # Datos procesados y limpios
├── output/
│   ├── figures/           # Gráficos (PNG)
│   ├── models/            # Objetos de modelos (RDS)
│   ├── reports/           # Documentos finales (PDF)
│   └── results/           # Tablas de métricas
├── app.R                  # Esta aplicación Shiny
└── README.md              # Documentación del proyecto"
          )
        )
      )
    ),
    fluidRow(
      column(
        12,
        div(
          class = "card-panel",
          h3("ℹ️ Información del proyecto"),
          fluidRow(
            column(
              6,
              p(strong("Institución:"), " MUCSS (Master's in Computational Social Science)"),
              p(strong("Universidad:"), " Carlos III de Madrid"),
              p(strong("Tipo:"), " Trabajo Fin de Máster (TFM)"),
              p(strong("Periodo de datos:"), " 2016-2025")
            ),
            column(
              6,
              p(strong("Fuentes:"), " INDEC (EPH, IPC), Canastas Nacionales"),
              p(strong("Cobertura:"), " Aglomerados metropolitanos de Argentina"),
              p(strong("Licencia:"), " Apache 2.0"),
              p(strong("Contacto:"), " lm.hcor@gmail.com")
            )
          )
        )
      )
    )
  )
)

server <- function(input, output, session) {
  # Observer for Explorador de figuras tab
  observe({
    group_files <- figure_mapping[[input$figure_group]]
    updateSelectInput(session, "figure_file", choices = group_files, selected = group_files[1])
  })
  
  # Render figure view for Resultados tab
  output$result_view <- renderUI({
    req(input$result_file)
    file_name <- input$result_file
    file_path <- file.path("figures", file_name)
    tags$div(
      class = "figure-frame",
      tags$img(src = file_path, style = "cursor: pointer;")
    )
  })
  
  output$model_metrics_table_dt <- renderDT({
    datatable(
      model_metrics,
      options = list(
        pageLength = 9,
        dom = 'ftp',
        search = list(regex = FALSE, caseInsensitive = TRUE),
        columnDefs = list(
          list(className = 'dt-center', targets = '_all'),
          list(render = JS("function(data, type, row) {
                        if(type === 'display') {
                            return parseFloat(data).toFixed(4);
                        }
                        return data;
                    }"), targets = 2:6)
        ),
        rowCallback = JS("
                    function(row, data, index) {
                        var modelo = data[0];
                        var conjunto = data[1];
                        var auc = parseFloat(data[3]);
                        
                        if(modelo === 'XGBoost' && conjunto === 'Test 2024') {
                            $('td', row).css('background-color', '#fef3c7');
                        }
                    }
                ")
      ),
      colnames = c('Modelo', 'Conjunto', 'Precisión', 'Recall', 'AUC-ROC', 'Sensibilidad', 'Especificidad'),
      class = 'cell-border stripe',
      filter = 'top',
      rownames = FALSE
    ) %>%
      formatRound(columns = c(3:7), digits = 4)
  })
  
  output$figure_view <- renderUI({
    req(input$figure_file)
    file_name <- input$figure_file
    file_path <- file.path("figures", file_name)
    tags$div(
      class = "figure-frame",
      tags$img(src = file_path, style = "cursor: pointer;")
    )
  })
}

shinyApp(ui = ui, server = server)