# =============================================================================
# Shared theme, palette and export helpers for all manuscript figures.
#
# Built on the figure contract:
#   - one restrained palette per figure: one neutral family, one signal family,
#     one accent family reserved for directional cues (gain / loss / failure)
#   - white background, no panel grid, direct labels where possible
#   - editable text on export (SVG + PDF), 600 dpi TIFF for raster fallback
#
# Every figure sources this file; nothing here draws anything on its own.
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(readr)
})

# ---------------------------------------------------------------- palette ----
# Three families, one job each (per the figure-contract stance):
#   neutral  -> structure, de-emphasised categories
#   signal   -> the two sound domains. Low-saturation periwinkle / rose from the
#               NMI pastel family. The stance file prefers a unified low-
#               saturation family over maximal hue separation; saturated
#               mid-tone pairings (navy+ochre, teal+violet) read as dated at
#               print size and fight with the directional accents.
#   accent   -> direction only (transfer holds / transfer fails, feature
#               class that survives / does not). Never used for plain categories.
pal <- c(
  neutral_dark  = "#272727",
  neutral_mid   = "#767676",
  neutral_light = "#D8D8D8",
  music         = "#7884B4",  # signal: music domain (periwinkle)
  music_light   = "#B4C0E4",
  ambient       = "#D08C9C",  # signal: ambient domain (dusty rose)
  ambient_light = "#E4CCD8",
  success       = "#4C9A6A",  # accent: holds / survives
  failure       = "#B64342",  # accent: fails / invalid
  ceiling       = "#111111"
)

pal_domain <- c("Music" = unname(pal["music"]),
                "Ambient" = unname(pal["ambient"]))

pal_class <- c("Temporal variability" = unname(pal["success"]),
               "Spectral level"       = unname(pal["neutral_mid"]))

# ------------------------------------------------------------------ theme ----
# Typography is fixed here and must NOT be overridden per panel — uniform axis
# type across every figure is part of the submission contract.
#
#   axis titles   7.0 pt  Arial regular
#   axis text     6.2 pt  Arial regular, pure black
#   panel title   7.5 pt  Arial bold
#   legend text   6.0 pt
#   panel tag     8.0 pt  Arial bold
#
# All sizes are chosen for the 183 mm double-column width at 100% scale; nothing
# falls below 6 pt, which is the usual lower bound for Nature-family artwork.
FONT <- "Arial"
SZ_AXIS_TITLE <- 7.0
SZ_AXIS_TEXT  <- 6.2
SZ_PANEL_TITLE <- 7.5
SZ_LEGEND <- 6.0
SZ_TAG <- 8.0

theme_ms <- function(base_family = FONT) {
  theme_classic(base_size = SZ_AXIS_TITLE, base_family = base_family) +
    theme(
      axis.line       = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks      = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks.length = unit(1.2, "pt"),
      axis.title      = element_text(size = SZ_AXIS_TITLE, colour = "black",
                                     face = "plain"),
      axis.text       = element_text(size = SZ_AXIS_TEXT, colour = "black",
                                     face = "plain"),
      legend.title    = element_text(size = SZ_LEGEND + 0.3),
      legend.text     = element_text(size = SZ_LEGEND),
      legend.key.size = unit(6, "pt"),
      legend.margin   = margin(0, 0, 0, 0),
      legend.background = element_blank(),
      strip.text      = element_text(size = SZ_LEGEND + 0.3, face = "bold"),
      strip.background = element_blank(),
      plot.title      = element_text(size = SZ_PANEL_TITLE, face = "bold",
                                     margin = margin(b = 3)),
      plot.margin     = margin(3, 3, 3, 3),
      panel.grid      = element_blank()
    )
}

theme_set(theme_ms())

# Panel tags for patchwork assemblies.
tag_theme <- function(size = SZ_TAG) {
  theme(plot.tag = element_text(size = size, face = "bold", family = FONT),
        plot.tag.position = c(0, 1))
}

# --------------------------------------------------------------- geometry ----
# Nature column widths.
W_SINGLE <- 89    # mm
W_DOUBLE <- 183   # mm

# ----------------------------------------------------------------- export ----
# Editable text in SVG/PDF; TIFF only as a raster fallback for submission
# systems that reject vectors.
save_ms <- function(plot, filename, width_mm = W_DOUBLE, height_mm = 120,
                    dpi = 600, tiff = TRUE) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)

  svglite::svglite(paste0(filename, ".svg"), width = w, height = h)
  print(plot); dev.off()

  grDevices::cairo_pdf(paste0(filename, ".pdf"), width = w, height = h)
  print(plot); dev.off()

  if (tiff) {
    ragg::agg_tiff(paste0(filename, ".tiff"), width = w, height = h,
                   units = "in", res = dpi, compression = "lzw")
    print(plot); dev.off()
  }
  # PNG for on-screen review. Kept at >= 300 dpi rather than a lighter preview
  # resolution: editors and reviewers do sometimes pull the PNG instead of the
  # TIFF, and a 200 dpi file would fail a journal raster check.
  ragg::agg_png(paste0(filename, ".png"), width = w, height = h,
                units = "in", res = 300)
  print(plot); dev.off()

  message(sprintf("  %s  [%.0f x %.0f mm]", basename(filename), width_mm, height_mm))
}

# ------------------------------------------------------------------- data ----
SRC <- file.path("..", "reports", "source_data")

read_src <- function(name) {
  readr::read_csv(file.path(SRC, paste0(name, ".csv")),
                  show_col_types = FALSE, progress = FALSE)
}

# Consistent corpus labels across every panel.
corpus_label <- c(
  deam    = "DEAM\n(music)",
  pmemo   = "PMEmo\n(music)",
  emo_mix = "Emo-Soundscapes\n(ambient)"
)
corpus_short <- c(deam = "DEAM", pmemo = "PMEmo", soundtracks = "Soundtr.",
                  emo_mix = "Emo-Sound.")
