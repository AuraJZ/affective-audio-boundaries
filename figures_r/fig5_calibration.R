# =============================================================================
# Figure 5 | Reference distributions and positive controls.
#
# Every panel instantiates one claim: a statistic is uninterpretable until it
# is placed against the distribution it would take under the null hypothesis,
# and a negative result is uninterpretable until the instrument is shown to
# detect a known effect.
#
#   a, b   A consensus statistic against its permutation null
#   c      The same negative conclusion from two pipelines, one of which fails
#          its positive controls
#   d      The positive control that licenses interpretation
#   e      The resulting comparison, matched for subjects, trials and estimator
#
# Panels c-e replace an earlier version built on an electrodermal analysis that
# was retracted after its pipeline failed positive controls; the substitution
# is documented in reports/step14 and step16.
# =============================================================================

source("_theme.R")

perm_null <- read_src("fig_calibration_permutation_null")
perm_obs  <- read_src("fig_calibration_permutation_obs")
ctrl      <- read_src("fig_calibration_controls")
erp       <- read_src("fig_calibration_erp")
# 🔴 Read the audited table, not the original one.
#
# `fig_calibration_icc.csv` predates run 83. It computes each measure on
# whatever trials that measure has (n runs 628-736 across the rating scales)
# and does NOT z-score within listener, so its self-report maximum is
# tender = 0.335 and the EEG-to-rating gap reads as 3.6-fold. Run 83 fixed
# both -- one common set of 1,240 trials, within-listener z-scoring -- giving
# pleasant = 0.221 and a 2.4-fold gap, and it is those numbers that are in the
# Results text and in this figure's own legend.
#
# The figure kept reading the old file, so for a day the panel said 0.335 and
# 3.6-fold while the caption beneath it said 0.221 and 2.4-fold. Nothing caught
# it: both files existed, both were current, and every assertion passed.
icc_raw   <- read_src("group_level_power")

# ------------------------------------------- panel a: permutation null -------
cnt <- perm_null %>% count(consensus_count)
obs <- perm_obs$observed_consensus[1]
pv  <- perm_obs$p_value[1]

p_a <- ggplot(cnt, aes(consensus_count, n)) +
  geom_col(width = 0.78, fill = pal[["neutral_light"]]) +
  geom_vline(xintercept = perm_obs$null_mean[1], linewidth = 0.35,
             linetype = "dashed", colour = pal["neutral_mid"]) +
  geom_vline(xintercept = obs, linewidth = 0.7, colour = pal[["failure"]]) +
  annotate("text", x = obs + 0.25, y = max(cnt$n) * 0.95,
           label = sprintf("observed = %d\np = %.2f", obs, pv),
           hjust = 0, size = 2.0, family = FONT, lineheight = 0.95,
           colour = pal[["failure"]], fontface = "bold") +
  annotate("text", x = perm_obs$null_mean[1] - 0.25, y = max(cnt$n) * 0.55,
           label = sprintf("null mean\n%.2f", perm_obs$null_mean[1]),
           hjust = 1, size = 1.9, family = FONT, lineheight = 0.95,
           colour = pal["neutral_mid"]) +
  scale_x_continuous(breaks = seq(0, 8, 2)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(x = "Features in the three-model SHAP top-20 intersection",
       y = "Label permutations")

# --------------------------------------- panel b: shuffling really worked ----
p_b <- ggplot(perm_null, aes(cv_auc)) +
  geom_density(fill = pal[["neutral_light"]], colour = pal["neutral_mid"],
               linewidth = 0.35, alpha = 0.7, bw = 0.02) +
  geom_vline(xintercept = 0.5, linewidth = 0.3, linetype = "dashed",
             colour = pal["neutral_mid"]) +
  geom_vline(xintercept = perm_obs$observed_cv_auc[1], linewidth = 0.7,
             colour = pal[["success"]]) +
  annotate("text", x = perm_obs$observed_cv_auc[1] - 0.02, y = 4,
           label = sprintf("intact labels\nAUC = %.3f", perm_obs$observed_cv_auc[1]),
           hjust = 1, size = 1.9, family = FONT, lineheight = 0.95,
           colour = pal[["success"]]) +
  scale_x_continuous(limits = c(0.38, 1.0), breaks = seq(0.4, 1.0, 0.2)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.06))) +
  labs(x = "Cross-validated AUC under permuted labels", y = "Density")

# ------------------------------------------- panel c: positive controls ------
# Effect sizes are on incommensurate scales (a correlation, a ratio, a t
# statistic), so the axis is -log10(p) with the direction of the effect carried
# by fill. A control that is significant in the wrong direction is the worst
# case and must be visually distinct from a null result.
ctrl <- ctrl %>%
  mutate(
    neglogp = pmin(-log10(pmax(p, 1e-10)), 10),
    outcome = factor(
      ifelse(passed, "Passed",
             ifelse(p < 0.05, "Significant, wrong direction", "Not detected")),
      levels = c("Passed", "Not detected", "Significant, wrong direction")),
    control = factor(control, levels = rev(unique(control))),
    dataset = factor(dataset, levels = unique(dataset)))

p_c <- ggplot(ctrl, aes(neglogp, control, fill = outcome)) +
  geom_vline(xintercept = -log10(0.05), linewidth = 0.3,
             linetype = "dashed", colour = pal["neutral_mid"]) +
  geom_col(width = 0.55) +
  # Long bars carry the label inside; short ones would clip it, so those sit
  # outside the bar end in dark text.
  geom_text(data = filter(ctrl, neglogp > 4), aes(label = label),
            x = 0.3, hjust = 0, size = 1.85, family = FONT, colour = "white") +
  geom_text(data = filter(ctrl, neglogp <= 4),
            aes(x = neglogp + 0.3, label = label), hjust = 0,
            size = 1.85, family = FONT, colour = pal[["neutral_dark"]]) +
  # Horizontal strips above each group. A vertical strip on either side costs
  # width that patchwork then reserves in panel a as well, and releasing that
  # with free() clips the strip entirely.
  facet_wrap(~ dataset, ncol = 1, scales = "free_y", strip.position = "top") +
  scale_fill_manual(values = c(
    "Passed" = unname(pal["success"]),
    "Not detected" = unname(pal["neutral_light"]),
    "Significant, wrong direction" = unname(pal["failure"])), guide = "none") +
  scale_x_continuous(limits = c(0, 10.6), expand = expansion(mult = c(0, 0.02)),
                     breaks = c(-log10(0.05), 5, 10),
                     labels = c("0.05", expression(10^-5), expression(10^-10))) +
  labs(x = expression(italic(p)), y = NULL) +
  theme(strip.text = element_text(size = SZ_LEGEND, family = FONT,
                                  face = "bold", hjust = 0,
                                  colour = pal[["neutral_dark"]],
                                  margin = margin(b = 0.8, t = 0.4, unit = "mm")),
        strip.background = element_blank(),
        panel.spacing.y = unit(2.2, "mm"))

# ------------------------------- panel d: the control that licenses reading --
# Grand-average auditory response at fronto-central sites, with the amplitude
# corresponding to the 95th percentile of the sign-flip max-statistic null.
sig_span <- erp %>% filter(significant)
p_d <- ggplot(erp, aes(time_s, amplitude_uV)) +
  {if (nrow(sig_span))
    annotate("rect", xmin = min(sig_span$time_s), xmax = max(sig_span$time_s),
             ymin = -Inf, ymax = Inf, fill = pal[["success"]], alpha = 0.09)} +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = pal["neutral_mid"]) +
  geom_vline(xintercept = 0, linewidth = 0.3, colour = pal["neutral_mid"]) +
  geom_ribbon(aes(ymin = -threshold_uV, ymax = threshold_uV),
              fill = pal[["neutral_light"]], alpha = 0.55) +
  geom_ribbon(aes(ymin = amplitude_uV - sem_uV, ymax = amplitude_uV + sem_uV),
              fill = pal[["music"]], alpha = 0.30) +
  geom_line(linewidth = 0.5, colour = pal[["music"]]) +
  annotate("text", x = 0.452, y = 1.72,
           label = "italic(p)==0.0001", parse = TRUE,
           size = 2.0, family = FONT, colour = pal[["neutral_dark"]]) +
  annotate("text", x = -0.19, y = -0.62, hjust = 0,
           label = "shaded band: sign-flip null (95th pct)",
           size = 1.8, family = FONT, colour = pal["neutral_mid"]) +
  scale_x_continuous(breaks = seq(-0.2, 0.6, 0.2)) +
  labs(x = "Time from sound onset (s)",
       y = expression("Fronto-central amplitude ("*mu*"V)"))

# ------------------ panel e: the interpretable comparison, matched design ----
# Self-report on top: it is the reference against which the EEG row is read.
# The EEG row spans all 156 measures. Restricting it to band power would
# understate it: the coherence family reaches more than twice the band-power
# maximum, and coherence is where the original report of these data located
# its effect.
SELF <- c("pleasant", "energetic", "tense", "angry",
          "afraid", "happy", "sad", "tender")
icc <- icc_raw %>%
  filter(kind == "observed", !is.na(icc)) %>%
  mutate(kind = factor(ifelse(measure %in% SELF, "Self-report", "EEG"),
                       levels = c("EEG", "Self-report")),
         measure = gsub("_", " ", measure))
# The 8 rating scales and 156 EEG measures are the population the maxima are
# selected from; if either count changes the selection correction in the text
# no longer applies to this panel.
stopifnot(sum(icc$kind == "Self-report") == 8,
          sum(icc$kind == "EEG") == 156)
best <- icc %>% group_by(kind) %>% slice_max(icc, n = 1) %>% ungroup()
fold <- best$icc[best$kind == "Self-report"] / best$icc[best$kind == "EEG"]
gap_y <- 1.5

p_e <- ggplot(icc, aes(icc, kind, colour = kind)) +
  geom_vline(xintercept = 0, linewidth = 0.3, colour = pal["neutral_mid"]) +
  geom_point(position = position_jitter(height = 0.15, seed = 1),
             size = 0.7, alpha = 0.45, stroke = 0) +
  stat_summary(fun = median, geom = "point", shape = 124,
               size = 3.6, stroke = 1.2, show.legend = FALSE) +
  # Labels point away from the connector so neither collides with it.
  geom_text(data = best, aes(label = sprintf("%s  %.3f", measure, icc),
                             vjust = ifelse(kind == "EEG", 2.1, -1.3)),
            hjust = -0.14, size = 1.95, family = FONT, show.legend = FALSE) +
  annotate("segment", x = best$icc[best$kind == "EEG"],
           xend = best$icc[best$kind == "Self-report"], y = gap_y, yend = gap_y,
           linewidth = 0.3, colour = pal["neutral_mid"],
           arrow = arrow(ends = "both", length = unit(1.1, "mm"))) +
  annotate("label", x = mean(best$icc), y = gap_y,
           label = sprintf("%.1f-fold", fold), size = 2.0, family = FONT,
           colour = pal["neutral_dark"], fill = "white",
           label.padding = unit(0.5, "mm")) +
  scale_colour_manual(values = c("Self-report" = unname(pal["music"]),
                                 "EEG" = unname(pal["ambient"])), guide = "none") +
  scale_x_continuous(limits = c(-0.06, 0.32), breaks = seq(0, 0.3, 0.1)) +
  scale_y_discrete(expand = expansion(add = c(0.55, 0.85)),
                   labels = c("EEG" = "EEG\n156 measures",
                              "Self-report" = "Self-report\n8 scales")) +
  labs(x = "Stimulus-level intraclass correlation (same subjects, same trials)",
       y = NULL)

# ------------------------------------------------------------------ assemble --
design <- "
AB
CD
EE
"
fig <- p_a + p_b + p_c + p_d + p_e +
  plot_layout(design = design, heights = c(1, 1.05, 0.42)) +
  plot_annotation(tag_levels = "a") & tag_theme()

# Guard the exact numbers the Results text and the legend quote from this panel.
# A figure that disagrees with its own caption is the one defect neither the
# LaTeX build nor a timestamp check can see.
stopifnot(
  abs(best$icc[best$kind == "Self-report"] - 0.221) < 0.003,
  abs(best$icc[best$kind == "EEG"]         - 0.092) < 0.003,
  abs(fold - 2.4) < 0.1
)
message("  Fig6_channel: ICC panel agrees with the text (0.221 / 0.092, 2.4-fold)")

save_ms(fig, file.path("..", "reports", "figures_r", "Fig6_channel"),
        width_mm = W_DOUBLE, height_mm = 132)
