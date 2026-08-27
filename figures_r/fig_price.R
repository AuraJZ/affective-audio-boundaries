# =============================================================================
# Figure: the price of the first boundary, and what it does not buy.
#
# This figure exists to cash a claim. Calling a failure "budget-limited" means
# nothing unless the budget can be quoted, so this is the panel that quotes it —
# and, equally, the panel that shows where the purchase stops.
#
#   a  what target-side labels buy, as a fraction of the gap between zero-shot
#      transfer and training on the target corpus itself. The denominator is the
#      target's OWN within-corpus ceiling, not same-domain transfer: the question
#      an engineer faces is "if I am labelling k examples anyway, how close does
#      that get me to labelling all of them".
#
#   b  the same in marginal terms. Each doubling of the budget returns a roughly
#      constant 14-16 points up to k = 100, then 5. This is the panel that keeps
#      the section honest — the curve flattens well below the ceiling, so the
#      last third is either expensive or not for sale, and this experiment does
#      not say which.
#
#   c  zero-shot alignment, with its sham control. Correlation alignment appears
#      to recover 22% of the gap; aligning instead to an unrelated third corpus,
#      which cannot carry target-specific information by construction, recovers
#      14%. Roughly two-thirds of the apparent gain is not target-specific, and
#      subspace alignment lands below doing nothing at all. Without the sham bar
#      this panel would read as a modest success.
#
# Source: reports/source_data/domain_adaptation.csv (script 80), with the
# per-corpus ceilings from fig_domain_wall.csv.
# =============================================================================

source("_theme.R")

DA <- read_src("domain_adaptation")
WALL <- read_src("fig_domain_wall")

ceilings <- WALL |>
  filter(domain_match == "within-corpus") |>
  group_by(target) |>
  summarise(ceiling = median(spearman), .groups = "drop")

# Ridge only: the tree ensembles are reported in the supplement. Mixing them
# here would blend two different sensitivities to the alignment step.
X <- DA |>
  filter(cross_domain, model == "ridge") |>
  mutate(tgt = sub("^reg_", "", target)) |>
  left_join(ceilings, by = c("tgt" = "target"))

zero <- X |>
  filter(method == "none") |>
  select(source, target, zero = rho)

recovered <- X |>
  inner_join(zero, by = c("source", "target")) |>
  mutate(frac = (rho - zero) / (ceiling - zero))

# ------------------------------------------------------------------ panel a --
# NB: compute the quantiles BEFORE the median overwrites `frac`. Written the
# other way round, dplyr feeds the already-summarised scalar to quantile() and
# the ribbon silently collapses onto the line.
fs <- recovered |>
  filter(method == "fewshot") |>
  group_by(k) |>
  summarise(lo = quantile(frac, .25), hi = quantile(frac, .75),
            frac = median(frac), .groups = "drop")

pa <- ggplot(fs, aes(k, frac)) +
  geom_hline(yintercept = 1, linetype = "22", linewidth = 0.3,
             colour = pal["ceiling"]) +
  annotate("text", x = 10, y = 1.04, hjust = 0, size = SZ_LEGEND / .pt,
           colour = pal["ceiling"], label = "training on the target corpus") +
  # 🔴 Colour: the earlier version drew this curve in pal["music"]. That member
  # is the SIGNAL family for the music domain (see _theme.R), and this curve is
  # not a domain -- it is a budget trajectory. Using it here put a large block of
  # periwinkle in the one figure whose neighbours are all grey-and-accent, and
  # broke the contract's rule that the signal family names the two sound domains
  # and nothing else. A recovered gap is a direction, so it takes the accent.
  geom_ribbon(aes(ymin = lo, ymax = hi), fill = pal["success"], alpha = 0.16) +
  geom_line(colour = pal["success"], linewidth = 0.6) +
  geom_point(colour = pal["success"], size = 1.4) +
  geom_text(aes(label = percent(frac, 1)), vjust = -1.1,
            size = SZ_LEGEND / .pt, colour = pal["neutral_dark"]) +
  scale_x_log10(breaks = fs$k, labels = fs$k) +
  scale_y_continuous(labels = percent_format(1), limits = c(0, 1.12),
                     breaks = seq(0, 1, .25)) +
  labs(title = "Target-side labels buy back most of the gap",
       x = "Labelled target examples", y = "Gap recovered")

# ------------------------------------------------------------------ panel b --
# Marginal return per doubling. The point of the panel is the last bar.
marg <- fs |>
  arrange(k) |>
  mutate(prev = lag(frac), step = (frac - prev) * 100,
         lab = paste0(lag(k), "→", k)) |>
  filter(!is.na(prev))

pb <- ggplot(marg, aes(reorder(lab, k), step)) +
  geom_col(aes(fill = step < 8), width = 0.62, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%+.0f", step)), vjust = -0.4,
            size = SZ_LEGEND / .pt, colour = pal["neutral_dark"]) +
  # Directional, so the accent family is the right one: the return holds for
  # three doublings and then collapses. Panel c below is grey-and-red instead,
  # because there the bars are methods -- plain categories -- with one of them
  # invalid by construction.
  scale_fill_manual(values = c(`FALSE` = unname(pal["success"]),
                               `TRUE`  = unname(pal["failure"]))) +
  scale_y_continuous(limits = c(0, 20), breaks = seq(0, 20, 5)) +
  labs(title = "Return per doubling collapses after 100",
       x = "Budget doubling", y = "Additional gap recovered (pp)")

# ------------------------------------------------------------------ panel c --
ALIGN <- c(coral = "CORAL", coral_sham = "CORAL to a wrong corpus (sham)",
           coral_diag = "CORAL, diagonal", zscore = "Per-corpus z-score",
           sa = "Subspace alignment")

za <- recovered |>
  filter(method %in% names(ALIGN)) |>
  group_by(method) |>
  summarise(frac = median(frac), .groups = "drop") |>
  mutate(label = ALIGN[method],
         sham = method == "coral_sham")

pc <- ggplot(za, aes(reorder(label, frac), frac)) +
  geom_col(aes(fill = sham), width = 0.62, show.legend = FALSE) +
  geom_text(aes(label = percent(frac, 1),
                hjust = ifelse(frac < 0, 1.18, -0.18)),
            size = SZ_LEGEND / .pt, colour = pal["neutral_dark"]) +
  scale_fill_manual(values = c(`FALSE` = unname(pal["neutral_mid"]),
                               `TRUE`  = unname(pal["failure"]))) +
  # Limits must admit the negative bar. Subspace alignment lands BELOW doing
  # nothing, and clipping it would turn "actively harmful" into "small".
  geom_hline(yintercept = 0, linewidth = 0.3, colour = pal["neutral_dark"]) +
  scale_y_continuous(labels = percent_format(1), limits = c(-0.10, 0.30),
                     breaks = seq(-0.1, 0.3, .1)) +
  coord_flip() +
  # theme_classic draws the category axis at the PANEL edge. With a negative
  # bar the baseline is mid-panel, so that line floats away from the data and
  # reads as a broken axis. The zero line above is the real baseline; drop the
  # decorative one. Same reason as in the other horizontal panels.
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank()) +
  labs(title = "Alignment without labels: most of the gain is not target-specific",
       x = NULL, y = "Gap recovered")

fig <- (pa | pb) / pc +
  plot_layout(heights = c(1, 0.78)) +
  plot_annotation(tag_levels = "a") & tag_theme()

save_ms(fig, file.path("..", "reports", "figures_r", "Fig2_price"),
        width_mm = W_DOUBLE, height_mm = 118)

# A guard, not decoration: the section's headline numbers are read off this
# figure, so the figure must agree with the text or one of them is wrong.
# The alignment fractions are computed PER PAIR and then pooled. Doing it the
# other way round -- pool the correlations, then take one ratio against one
# median ceiling -- gives 36% and 27% instead of 22% and 14%, because it mixes
# pairs whose baselines differ by more than the effect being measured. The text
# quoted the pooled numbers until this guard caught them.
stopifnot(
  abs(fs$frac[fs$k == 50]  - 0.50) < 0.03,
  abs(fs$frac[fs$k == 200] - 0.69) < 0.03,
  abs(za$frac[za$method == "coral"]      - 0.22) < 0.03,
  abs(za$frac[za$method == "coral_sham"] - 0.14) < 0.03,
  za$frac[za$method == "sa"] < 0
)
message("  Fig2_price: text/figure agreement checks passed")
