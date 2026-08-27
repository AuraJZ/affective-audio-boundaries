# =============================================================================
# Figure: the four boundaries, and the two kinds of failure.
#
# This is the figure the paper is organised around, and the only one that shows
# all four boundaries on one set of axes. Everything else in the manuscript
# measures one crossing; this one puts them side by side so the claim — that the
# failures divide into two kinds with mutually exclusive remedies — can be read
# rather than asserted.
#
#   a  survival. What fraction of the attainable ceiling is left after each
#      crossing. Ordered near-to-far, which is also the order an application
#      meets them.
#   b  price. What it costs to buy the gap back, on a common axis of target-side
#      observations. Two boundaries have a finite price and are drawn as bars;
#      on two, no purchase was found, and those are drawn as an open marker with
#      the reason. Drawing an unfound price as a very long bar would be a lie of
#      the same kind as clipping a negative value — it would suggest the purchase
#      exists and is merely expensive.
#
#      The marker says "none found", not "none exists". Only the priced
#      boundaries have a target-side budget curve behind them; for the other two
#      the claim is that every representation and route we tested failed to
#      recover the gap, which is weaker than saturation and the figure must not
#      read as though it were stronger.
#   c  the falsification test. The dichotomy is worth nothing if it cannot fail,
#      and it would fail if a gap yielded to both remedies. The panel shows the
#      one case we found where a representation closes an information-limited
#      gap, against the seven axes where it does almost nothing.
#
# LAYOUT NOTE. The first draft used facet_grid to group rows by boundary. That
# produces two columns of y labels — strip text on the far left, row labels
# beside the panel — with an empty gutter between them, and it detaches the axis
# line from the bars. Both panels now carry the boundary in the row label itself
# and use no facets and no y axis line: for a horizontal bar chart the baseline
# IS the information, and a second vertical rule at the panel edge is noise that
# reads as a broken axis.
#
# Sources: fig_domain_wall.csv, ast_domain_wall.csv, domain_adaptation.csv,
# canonical_tau_table.csv, ds002721_eeg_reliability.csv, fig_tonality_axes.csv.
# =============================================================================

source("_theme.R")

# Colour carries the diagnosis, not the boundary. Two families only: a gap that
# can be bought back, and one that cannot.
KIND <- c(Budget = unname(pal["music"]), Information = unname(pal["failure"]))

# Horizontal bars: no y axis line, and let the bars start exactly at zero.
theme_hbar <- theme(axis.line.y = element_blank(),
                    axis.ticks.y = element_blank())

# ------------------------------------------------------------------ panel a --
# Cross-corpus contributes two rows because it is the one boundary on which both
# kinds appear inside a single experiment; that is why it leads the paper.
#
# The cross-synthesis row is a special case: what survives is the SIGN of the
# effect and what does not is its calibration. A zero-height bar would leave a
# floating label attached to nothing, so that arm is drawn as an explicit
# "0%" marker rather than as an absent bar.
surv <- tibble::tribble(
  ~ord, ~arm,                                  ~survival, ~kind,
  1, "1 · cross-corpus, within domain",          0.798,   "Budget",
  2, "1 · cross-corpus, across domains",         0.061,   "Information",
  3, "2 · cross-synthesis, sign of the edit",    1.000,   "Budget",
  4, "2 · cross-synthesis, its calibration",     0.000,   "Budget",
  5, "3 · cross-channel, EEG vs ratings",        0.416,   "Information",
  6, "4 · cross-individual, soundscapes",        0.630,   "Budget",
  7, "4 · cross-individual, electrodermal",      0.132,   "Budget"
) |>
  mutate(arm = factor(arm, levels = rev(arm)))

pa <- ggplot(surv, aes(survival, arm, fill = kind)) +
  geom_col(width = 0.68) +
  geom_text(aes(label = percent(survival, 1)), hjust = -0.22,
            size = SZ_LEGEND / .pt, colour = pal[["neutral_dark"]]) +
  scale_fill_manual(values = KIND, name = NULL) +
  scale_x_continuous(labels = percent_format(1), limits = c(0, 1.16),
                     breaks = seq(0, 1, .25), expand = expansion(0)) +
  theme_hbar +
  theme(legend.position = "top", legend.justification = "left",
        legend.margin = margin(0, 0, -2, 0)) +
  labs(title = "What survives the crossing",
       x = "Fraction of the attainable ceiling retained", y = NULL)

# ------------------------------------------------------------------ panel b --
# `price` is NA where no purchase is on offer; those rows get a marker, not a
# bar. The reason sits on the same line as the marker rather than below it —
# a second line of text under each row was landing inside the next row's band.
price <- tibble::tribble(
  ~ord, ~arm,                                    ~price, ~note,
  1, "1 · within domain",                          100,   "labels, for 2/3 of the gap",
  2, "1 · across domains",                          NA,   "4 representations tested, none crossed",
  3, "2 · calibration",                             NA,   "not a quantity of data",
  # 🔴 was "every source tested stops at 28% of the ceiling". That figure came
  # from dividing each route by a single ceiling (phasic_mean's 0.592) even when
  # its correlation was computed on a different electrodermal measure. Rescored
  # per measure, the best route reaches 30.6%, so the bound is 31%, not 28%.
  4, "3 · any information source",                  NA,   "no source tested exceeds 31% of the ceiling",
  5, "4 · soundscape ratings",                      67,   "observations per person",
  6, "4 · music ratings",                          166,   "observations per person",
  7, "4 · music electrodermal",                    318,   "observations per person"
) |>
  mutate(arm = factor(arm, levels = rev(arm)))

XMAX <- 760          # room for the note text to the right of the longest bar
pb <- ggplot(price, aes(y = arm)) +
  geom_col(aes(x = price), fill = KIND[["Budget"]], width = 0.68,
           na.rm = TRUE) +
  geom_text(aes(x = price, label = paste0(price, "  ", note)), hjust = -0.08,
            na.rm = TRUE, size = SZ_LEGEND / .pt,
            colour = pal[["neutral_dark"]]) +
  geom_point(data = ~ dplyr::filter(.x, is.na(price)), aes(x = 14),
             shape = 21, size = 2.2, stroke = 0.7,
             colour = KIND[["Information"]], fill = "white") +
  geom_text(data = ~ dplyr::filter(.x, is.na(price)),
            aes(x = 34, label = paste0("no price found  —  ", note)), hjust = 0,
            size = SZ_LEGEND / .pt, colour = KIND[["Information"]]) +
  scale_x_continuous(limits = c(0, XMAX), breaks = seq(0, 600, 200),
                     expand = expansion(0)) +
  theme_hbar +
  theme(plot.subtitle = element_text(size = SZ_LEGEND - 0.4,
                                     colour = pal[["neutral_mid"]],
                                     margin = margin(b = 3))) +
  labs(title = "What it costs to buy the gap back",
       subtitle = "open markers: no purchase found under the remedies tested; no budget curve was measured there",
       x = "Target-side observations", y = NULL)

# ------------------------------------------------------------------ panel c --
tg <- read_src("fig_tonality_axes") |>
  dplyr::select(axis, gain = gain_rho) |>
  dplyr::mutate(is_happy = grepl("happ", axis, ignore.case = TRUE))

pc <- ggplot(tg, aes(reorder(axis, gain), gain, fill = is_happy)) +
  geom_col(width = 0.66, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%+.3f", gain)), hjust = -0.18,
            size = SZ_LEGEND / .pt, colour = pal[["neutral_dark"]]) +
  scale_fill_manual(values = c(`FALSE` = unname(pal["neutral_light"]),
                               `TRUE` = unname(pal["success"]))) +
  scale_y_continuous(limits = c(0, 0.165), expand = expansion(0)) +
  coord_flip() +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank()) +
  # Two short lines rather than one long one: at this panel width a single-line
  # title is clipped at the right edge.
  labs(title = "The falsification test:\none information-limited gap does close",
       x = NULL, y = expression("Gain from 39 tonal descriptors ("*Delta*rho*")"))

fig <- (pa / pb) | pc
fig <- fig + plot_layout(widths = c(1.75, 1)) +
  plot_annotation(tag_levels = "a") & tag_theme()

save_ms(fig, file.path("..", "reports", "figures_r", "Fig8_boundaries"),
        width_mm = W_DOUBLE, height_mm = 124)

# Guards against the two ways this figure could quietly become a lie: a boundary
# silently acquiring a price it does not have, and the falsification panel
# losing the concentration that made it a test rather than a demonstration.
stopifnot(
  sum(is.na(price$price)) == 3,
  identical(sort(price$price[!is.na(price$price)]), c(67, 100, 166, 318)),
  tg$gain[tg$is_happy] > 3 * median(tg$gain[!tg$is_happy])
)
message("  Fig8_boundaries: dichotomy and falsification checks passed")
