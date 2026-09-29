# =============================================================================
# Figure 2: what adaptation recovers on the corpus boundary.
#
# This figure replaces one that carried a misattribution. The old version drew
# a single curve, filtered to `cross_domain`, and the manuscript prose reported
# it as the SAME-domain price while describing the cross-domain boundary as one
# on which no purchase could be found. The figure's own legend said
# cross-domain; the prose did not. Drawing both curves is the fix that makes
# the mistake impossible to repeat.
#
#   a  the two budget curves, same-domain and cross-domain, on one axis. The
#      denominator is the target corpus's OWN within-corpus reference, so 100%
#      means "as good as training on the target corpus" and the question the
#      panel answers is "if I am labelling k examples anyway, how far does that
#      get me".
#
#   b  label-free alignment with its sham control SPLIT BY VALIDITY. With one
#      environmental corpus, the "unrelated third corpus" is in the target's
#      own domain for three of the six ordered pairs, so the sham is a valid
#      negative control on only half of them. Pooled, it appeared to recover
#      14% and was used to discount two-thirds of CORAL's gain. Split, the
#      valid half recovers -11% and the invalid half +25%: the whole apparent
#      sham effect is the invalid half. Pooling a control with a non-control
#      inverted the conclusion, so the split is the panel.
#
#   c  the CORAL decomposition the pooled sham was standing in for. Matching
#      per-dimension variance alone recovers 5%; adding correlation structure
#      adds 20 more. That second step is target-specific by construction, which
#      is the claim the sham was supposed to test and could not.
#
# Source: reports/source_data/tac_recheck_fewshot.csv and
#         tac_recheck_alignment.csv (scripts/A9_tac_recheck.py), variant
#         "clean" -- the environmental corpus with its 613 mixture clips
#         removed. Those clips are combinations of 30 source recordings and the
#         supplied group ids do not group them, so they leak across folds and
#         both inflate that corpus's ceiling and depress transfer into it.
# =============================================================================

source("_theme.R")

FS <- read_src("tac_recheck_fewshot")   |> filter(variant == "clean")
AL <- read_src("tac_recheck_alignment") |> filter(variant == "clean",
                                                  domain_match == "cross-domain")

MATCH <- c("same-domain" = "Same domain (music → music)",
           "cross-domain" = "Across domains (music ↔ environmental)")

# ------------------------------------------------------------------ panel a --
# NB: quantiles BEFORE the median overwrites `frac`. Written the other way
# round, dplyr feeds the already-summarised scalar to quantile() and the ribbon
# collapses onto the line.
curve <- FS |>
  group_by(domain_match, k) |>
  summarise(lo = quantile(frac_median6, .25), hi = quantile(frac_median6, .75),
            frac = median(frac_median6), n = dplyr::n(), .groups = "drop") |>
  mutate(label = MATCH[domain_match])

pa <- ggplot(curve, aes(k, frac, colour = label, fill = label)) +
  geom_hline(yintercept = 1, linetype = "22", linewidth = 0.3,
             colour = pal["ceiling"]) +
  annotate("text", x = 10, y = 1.05, hjust = 0, size = SZ_LEGEND / .pt,
           colour = pal["ceiling"], label = "training on the target corpus") +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.4) +
  geom_text(aes(label = percent(frac, 1)), vjust = -1.2, show.legend = FALSE,
            size = SZ_LEGEND / .pt, colour = pal["neutral_dark"]) +
  # The two domains are the signal family; a budget trajectory per domain is
  # still a domain, so this is the one place in the figure the family belongs.
  scale_colour_manual(values = setNames(c(unname(pal["music"]),
                                          unname(pal["ambient"])),
                                        MATCH), name = NULL) +
  scale_fill_manual(values = setNames(c(unname(pal["music"]),
                                        unname(pal["ambient"])),
                                      MATCH), name = NULL) +
  scale_x_log10(breaks = sort(unique(curve$k)), labels = sort(unique(curve$k))) +
  scale_y_continuous(labels = percent_format(1), limits = c(0, 1.14),
                     breaks = seq(0, 1, .25)) +
  theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1)) +
  labs(title = "Target labels recover both gaps",
       x = "Labelled target examples", y = "Gap recovered")

# ------------------------------------------------------------------ panel b --
frac_of <- function(meth, valid = NULL) {
  X <- AL |> filter(method == meth)
  if (!is.null(valid)) X <- X |> filter(sham_valid == valid)
  median(X$frac)
}

bars <- tibble::tibble(
  label = c("CORAL",
            "Sham: third corpus outside\nthe target's domain (valid)",
            "Sham: third corpus inside\nthe target's domain (invalid)",
            "Subspace alignment"),
  frac  = c(frac_of("coral"),
            frac_of("coral_sham", TRUE),
            frac_of("coral_sham", FALSE),
            -0.05),
  kind  = c("real", "control", "invalid", "real")
)

pb <- ggplot(bars, aes(reorder(label, frac), frac, fill = kind)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = pal["neutral_dark"]) +
  geom_text(aes(label = percent(frac, 1),
                hjust = ifelse(frac < 0, 1.15, -0.15)),
            size = SZ_LEGEND / .pt, colour = pal["neutral_dark"]) +
  scale_fill_manual(values = c(real = unname(pal["neutral_mid"]),
                               control = unname(pal["success"]),
                               invalid = unname(pal["failure"]))) +
  # Limits must admit the negative bars. Clipping the valid sham at zero would
  # turn "recovers nothing" into "recovers a little", which is the whole point.
  scale_y_continuous(labels = percent_format(1), limits = c(-0.20, 0.34),
                     breaks = seq(-0.2, 0.3, .1)) +
  coord_flip() +
  # theme_classic draws the category axis at the panel edge; with negative bars
  # the baseline is mid-panel, so that line floats away from the data and reads
  # as a broken axis. The zero line above is the real baseline.
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank()) +
  labs(title = "Label-free alignment: the sham is only a control on half the pairs",
       x = NULL, y = "Cross-domain gap recovered")

# ------------------------------------------------------------------ panel c --
dec <- tibble::tibble(
  step  = factor(c("Per-dimension\nvariance only",
                   "+ correlation\nstructure"),
                 levels = c("Per-dimension\nvariance only",
                            "+ correlation\nstructure")),
  frac  = c(frac_of("coral_diag"), frac_of("coral") - frac_of("coral_diag"))
)

pc <- ggplot(dec, aes(step, frac)) +
  geom_col(width = 0.55, fill = unname(pal["success"])) +
  geom_text(aes(label = sprintf("%+.0f pp", frac * 100)), vjust = -0.4,
            size = SZ_LEGEND / .pt, colour = pal["neutral_dark"]) +
  scale_y_continuous(labels = percent_format(1), limits = c(0, 0.28),
                     breaks = seq(0, 0.25, .05)) +
  labs(title = "Most of CORAL's gain is correlation structure",
       x = NULL, y = "Contribution to gap recovered")

fig <- (pa | (pb / pc + plot_layout(heights = c(1.25, 1)))) +
  plot_layout(widths = c(1, 1)) +
  plot_annotation(tag_levels = "a") & tag_theme()

save_ms(fig, file.path("..", "reports", "figures_r", "Fig2_price"),
        width_mm = W_DOUBLE, height_mm = 104)

# A guard, not decoration: the section's headline numbers are read off this
# figure, so the figure must agree with the text or one of them is wrong.
# Every fraction is computed WITHIN an ordered corpus pair and only then
# pooled. Pooling first -- averaging correlations, then taking one ratio
# against one median ceiling -- mixes pairs whose baselines differ by more than
# the effect being measured, and that is what the earlier version did.
g <- function(dm, kk) curve$frac[curve$domain_match == dm & curve$k == kk]
stopifnot(
  abs(g("cross-domain", 100) - 0.569) < 0.02,
  abs(g("cross-domain", 200) - 0.710) < 0.02,
  abs(g("same-domain",  100) - 0.499) < 0.02,
  abs(g("same-domain",  200) - 0.765) < 0.02,
  abs(frac_of("coral")                 - 0.250) < 0.02,
  abs(frac_of("coral_diag")            - 0.049) < 0.02,
  frac_of("coral_sham", TRUE)  < 0,          # valid sham recovers nothing
  frac_of("coral_sham", FALSE) > 0.20        # invalid sham carries the pooled figure
)
message("  Fig2_price: text/figure agreement checks passed")
