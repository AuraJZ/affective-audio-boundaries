# =============================================================================
# Extended Data figures 1-7.
#
# These carry robustness and method evidence that would crowd the main figures.
# Same theme, palette and export contract as Fig. 1-5; nothing here introduces
# a new colour language.
#
#   ED1  Loudness is a corpus fingerprint, not signal
#   ED2  Sample utilisation: regression versus binary framing
#   ED3  Glass-box models cost almost no accuracy
#   ED4  EBM shape functions: thresholds read directly off the curve
#   ED5  Rashomon-set feature stability
#   ED6  Individual differences in the physiological response
#   ED7  Hand-crafted descriptors versus pretrained embeddings
# =============================================================================

source("_theme.R")

OUT <- file.path("..", "reports", "figures_r")

# ================================================================= ED1 =======
ed1 <- function() {
  d <- read_src("ed_loudness")
  r <- read_src("ed_loudness_ranks")

  p1 <- ggplot(d, aes(reorder(model, -auc), auc, fill = condition)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.64) +
    coord_cartesian(ylim = c(0.75, 0.96)) +
    scale_fill_manual(values = c("Raw" = unname(pal["neutral_light"]),
                                 "Loudness-normalised" = unname(pal["music"])),
                      name = NULL) +
    labs(x = NULL, y = "Cross-corpus AUC (DEAM → PMEmo)") +
    theme(legend.position = c(0.98, 0.98), legend.justification = c(1, 1))

  long <- r %>%
    pivot_longer(c(rank_raw, rank_norm), names_to = "cond", values_to = "rank") %>%
    mutate(cond = ifelse(cond == "rank_raw", "Raw", "Loudness-normalised"))

  # 🔴 The two point colours had no key, and they were a THIRD encoding: panel a
  # uses blue for normalised and grey for raw, while this panel used grey and
  # red. A reader could not tell which end of each arrow was which. Same colours
  # as panel a, and a legend.
  p2 <- ggplot(r, aes(y = reorder(feature, -rank_raw))) +
    geom_segment(aes(x = rank_raw, xend = rank_norm, yend = reorder(feature, -rank_raw)),
                 linewidth = 1.4, colour = pal["neutral_light"], lineend = "round",
                 arrow = arrow(length = unit(2.4, "pt"), type = "closed")) +
    geom_point(aes(x = rank_raw, colour = "Raw"), size = 1.7) +
    geom_point(aes(x = rank_norm, colour = "Loudness-normalised"), size = 1.7) +
    scale_colour_manual(values = c("Raw" = unname(pal["neutral_light"]),
                                   "Loudness-normalised" = unname(pal["music"])),
                        name = NULL) +
    scale_x_continuous(limits = c(0, 108)) +
    theme(legend.position = c(0.98, 0.06), legend.justification = c(1, 0)) +
    labs(x = "Mean SHAP rank (higher = less important)", y = NULL)

  fig <- (p1 | p2) + plot_annotation(tag_levels = "a") & tag_theme()
  save_ms(fig, file.path(OUT, "ED1_loudness"), W_DOUBLE, 62)
}

# ================================================================= ED2 =======
ed2 <- function() {
  d <- read_src("ed_data_use") %>%
    pivot_longer(c(available, regression, binary),
                 names_to = "framing", values_to = "n") %>%
    mutate(framing = factor(framing, levels = c("available", "regression", "binary"),
                            labels = c("Annotated and available",
                                       "Used by regression framing",
                                       "Used by binary framing")))

  p1 <- ggplot(d, aes(n, corpus, fill = framing)) +
    geom_col(position = position_dodge(width = 0.74), width = 0.66) +
    scale_fill_manual(values = c(unname(pal["neutral_light"]),
                                 unname(pal["success"]),
                                 unname(pal["failure"])), name = NULL) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.05))) +
    labs(x = "Clips", y = NULL) +
    theme(legend.position = "top", legend.margin = margin(b = -3))

  tot <- d %>% group_by(framing) %>% summarise(n = sum(n), .groups = "drop")
  p2 <- ggplot(tot, aes(framing, n, fill = framing)) +
    geom_col(width = 0.55) +
    geom_text(aes(label = n), vjust = -0.4, size = 2.0, family = FONT,
              colour = pal["neutral_dark"]) +
    scale_fill_manual(values = c(unname(pal["neutral_light"]),
                                 unname(pal["success"]),
                                 unname(pal["failure"])), guide = "none") +
    scale_x_discrete(labels = function(x) gsub(" ", "\n", x)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(x = NULL, y = "Total clips") +
    theme(axis.text.x = element_text(lineheight = 0.85))

  fig <- (p1 | p2) + plot_layout(widths = c(1.4, 1)) +
    plot_annotation(tag_levels = "a") & tag_theme()
  save_ms(fig, file.path(OUT, "ED2_data_use"), W_DOUBLE, 62)
}

# ================================================================= ED3 =======
ed3 <- function() {
  d <- read_src("ed_glassbox") %>%
    mutate(model = factor(model, levels = c(
      "RF", "XGBoost", "XGBoost(topk,no constraint)",
      "XGBoost+monotone", "EBM")),
      glassbox = model %in% c("XGBoost+monotone", "EBM"))

  p1 <- ggplot(d, aes(model, spearman, fill = glassbox)) +
    geom_col(width = 0.62) +
    facet_wrap(~ domain, nrow = 1) +
    coord_cartesian(ylim = c(0.70, 0.80)) +
    scale_fill_manual(values = c(`FALSE` = unname(pal["neutral_light"]),
                                 `TRUE` = unname(pal["success"])),
                      labels = c("Black box", "Glass box"), name = NULL) +
    scale_x_discrete(labels = c("RF", "XGB", "XGB\n(top-k)",
                                "XGB\n+monotone", "EBM")) +
    labs(x = NULL, y = expression("Spearman "*rho)) +
    theme(axis.text.x = element_text(lineheight = 0.85),
          legend.position = "top", legend.margin = margin(b = -3))

  cost <- d %>%
    filter(model %in% c("XGBoost(topk,no constraint)", "XGBoost+monotone")) %>%
    select(domain, model, spearman) %>%
    pivot_wider(names_from = model, values_from = spearman) %>%
    mutate(delta = `XGBoost+monotone` - `XGBoost(topk,no constraint)`)

  p2 <- ggplot(cost, aes(domain, delta)) +
    geom_hline(yintercept = 0, linewidth = 0.35, colour = pal["neutral_dark"]) +
    geom_col(width = 0.4, fill = pal[["success"]]) +
    geom_text(aes(label = sprintf("%+.4f", delta),
                  vjust = ifelse(delta > 0, -0.6, 1.5)),
              size = 2.0, family = FONT, colour = pal["neutral_dark"]) +
    scale_y_continuous(limits = c(-0.01, 0.01)) +
    labs(x = NULL, y = expression(Delta*rho*" from monotone constraint"))

  fig <- (p1 | p2) + plot_layout(widths = c(1.7, 1)) +
    plot_annotation(tag_levels = "a") & tag_theme()
  save_ms(fig, file.path(OUT, "ED3_glassbox"), W_DOUBLE, 62)
}

# ================================================================= ED4 =======
ed4 <- function() {
  d <- read_src("ed_shape_functions")
  # Threshold = where the additive contribution crosses zero.
  cross <- d %>%
    group_by(feature) %>%
    arrange(value, .by_group = TRUE) %>%
    filter(sign(contribution) != lag(sign(contribution))) %>%
    slice_head(n = 1) %>%
    ungroup()

  fig <- ggplot(d, aes(value, contribution)) +
    geom_hline(yintercept = 0, linewidth = 0.3, linetype = "dashed",
               colour = pal["neutral_mid"]) +
    geom_vline(data = cross, aes(xintercept = value), linewidth = 0.35,
               colour = pal["failure"]) +
    geom_line(aes(colour = feature_class), linewidth = 0.6) +
    geom_text(data = cross, aes(x = value, y = Inf,
                                label = sprintf("%.3g", value)),
              vjust = 1.4, hjust = -0.1, size = 1.85, family = FONT,
              colour = pal["failure"]) +
    facet_wrap(~ feature, scales = "free_x", nrow = 2) +
    scale_colour_manual(values = pal_class, name = NULL) +
    labs(x = "Descriptor value", y = "Additive contribution to arousal") +
    theme(legend.position = "top", legend.margin = margin(b = -3),
          panel.spacing = unit(6, "pt"))

  save_ms(fig, file.path(OUT, "ED4_shape_functions"), W_DOUBLE, 78)
}

# ================================================================= ED5 =======
ed5 <- function() {
  d <- read_src("ed_rashomon") %>%
    mutate(feature = reorder(feature, frequency))

  fig <- ggplot(d, aes(frequency, feature, fill = feature_class)) +
    annotate("rect", xmin = 0.99, xmax = 1.02, ymin = -Inf, ymax = Inf,
             fill = pal[["success"]], alpha = 0.10) +
    geom_col(width = 0.68) +
    geom_vline(xintercept = 1, linewidth = 0.35, linetype = "dashed",
               colour = pal["neutral_dark"]) +
    annotate("text", x = 0.985, y = 1.5, label = "recovered by every\nnear-optimal model",
             hjust = 1, size = 1.85, family = FONT, lineheight = 0.95,
             colour = pal["neutral_mid"]) +
    scale_fill_manual(values = pal_class, name = NULL) +
    scale_x_continuous(limits = c(0, 1.02), expand = c(0, 0),
                       labels = percent_format(accuracy = 1)) +
    labs(x = "Share of Rashomon-set models ranking the feature in the top 15",
         y = NULL) +
    # Legend on top rather than inside: the only empty region inside the panel
    # is the bottom-right corner, and the annotation already occupies it.
    theme(legend.position = "top", legend.justification = "left",
          legend.key.size = unit(2.8, "mm"),
          legend.margin = margin(b = -1.5, unit = "mm"))

  save_ms(fig, file.path(OUT, "ED5_rashomon"), W_SINGLE * 1.6, 86)
}

# ================================================================= ED6 =======
# Rebuilt on ds002721. The earlier version used an electrodermal analysis whose
# pipeline failed its positive controls and was retracted; the replacement runs
# on a pipeline that passes all four (Fig. 5c, d).
ed6 <- function() {
  d <- read_src("ed_individual_eeg") %>%
    mutate(family = factor(family, levels = c("delta", "theta", "alpha",
                                              "beta", "gamma", "faa")),
           question = factor(question,
                             levels = c("pleasant", "energetic", "tense",
                                        "angry", "afraid", "happy", "sad",
                                        "tender")))

  # Volcano-style: effect against evidence. With no combination surviving FDR
  # correction the point is the absence of any excursion, so the FDR threshold
  # must be drawn even though nothing crosses it.
  thr <- -log10(max(d$p[d$q < 0.05], 0.05 / nrow(d)))

  fig <- ggplot(d, aes(mean_rho, -log10(p))) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = thr, ymax = Inf,
             fill = pal[["success"]], alpha = 0.06) +
    geom_hline(yintercept = thr, linewidth = 0.3, linetype = "dashed",
               colour = pal["neutral_mid"]) +
    geom_vline(xintercept = 0, linewidth = 0.3, colour = pal["neutral_light"]) +
    geom_point(aes(colour = question, size = frac_same_sign),
               alpha = 0.75, stroke = 0) +
    annotate("text", x = -0.115, y = thr, hjust = 0, vjust = -0.6,
             label = "FDR < 0.05 (0 of 208 combinations)",
             size = 1.9, family = FONT, colour = pal["neutral_mid"]) +
    scale_size_continuous(range = c(0.7, 2.6), name = "Sign agreement\nacross subjects",
                          breaks = c(0.5, 0.6, 0.7), labels = percent) +
    scale_colour_manual(values = colorRampPalette(
      c(pal[["music"]], pal[["neutral_mid"]], pal[["ambient"]]))(8),
      name = "Rating scale") +
    scale_x_continuous(limits = c(-0.125, 0.125)) +
    labs(x = expression("Mean per-subject partial "*rho*
                        " (EEG measure vs own rating, trial index removed)"),
         y = expression(-log[10]*"("*italic(p)*")")) +
    guides(colour = guide_legend(override.aes = list(size = 1.6), ncol = 1)) +
    theme(legend.position = "right",
          legend.key.size = unit(2.8, "mm"),
          legend.title = element_text(lineheight = 0.9))

  save_ms(fig, file.path(OUT, "ED6_individual"), W_SINGLE * 1.7, 72)
}

# ================================================================= ED7 =======
ed7 <- function() {
  d <- read_src("ed_representations") %>%
    filter(representation %in% c("A", "C")) %>%
    mutate(representation = factor(representation, levels = c("A", "C"),
                                   labels = c("Hand-crafted (122-d)",
                                              "CLAP embedding (512-d)")),
           corpus = factor(corpus))

  p1 <- ggplot(d, aes(reorder(algorithm, spearman), spearman,
                      fill = representation)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.64) +
    facet_wrap(~ corpus, nrow = 1, labeller = label_wrap_gen(28)) +
    # Floor set below the weakest bar. At 0.68 the hand-crafted k-NN bar on DEAM
    # (0.666) was clipped to nothing and read as a missing condition.
    coord_cartesian(ylim = c(floor(min(d$spearman) * 50) / 50, 0.88)) +
    scale_fill_manual(values = c(unname(pal["neutral_light"]),
                                 unname(pal["music"])), name = NULL) +
    labs(x = NULL, y = expression("Spearman "*rho)) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1),
          legend.position = "top", legend.margin = margin(b = -3),
          strip.text = element_text(lineheight = 0.9))

  # Take `contaminated` from the source data rather than re-deriving it from the
  # corpus name: the name no longer carries the "(Freesound-derived)" suffix the
  # old grepl relied on, and the silent failure coloured every bar alike.
  gain <- d %>%
    group_by(corpus, contaminated, representation) %>%
    summarise(best = max(spearman), .groups = "drop") %>%
    pivot_wider(names_from = representation, values_from = best) %>%
    mutate(delta = `CLAP embedding (512-d)` - `Hand-crafted (122-d)`)
  stopifnot(sum(gain$contaminated) == 1)

  clean_mean <- mean(gain$delta[!gain$contaminated])

  p2 <- ggplot(gain, aes(reorder(corpus, delta), delta, fill = contaminated)) +
    geom_hline(yintercept = clean_mean, linewidth = 0.35, linetype = "dashed",
               colour = pal["neutral_mid"]) +
    geom_col(width = 0.45) +
    geom_text(aes(label = sprintf("%+.3f", delta)), vjust = -0.5,
              size = 2.0, family = FONT, colour = pal["neutral_dark"]) +
    # 🔴 Two fixes. The wording said "the two INDEPENDENT corpora" — the only
    # independence claim this project still supports is the loudness control's
    # three feature pipelines, and these two corpora share their descriptors,
    # folds and estimator. What they are is OUTSIDE CLAP's pretraining data;
    # that is the property being asserted, and it is not independence.
    # Second, at vjust = -0.45 the caption sat on the dashed line and collided
    # with the "+0.034" bar label; it now clears the line.
    annotate("text", x = 0.52, y = clean_mean, hjust = 0, vjust = -1.5,
             label = sprintf("mean of the two corpora outside\nthe pretraining data  %+.3f",
                             clean_mean),
             size = 1.8, family = FONT, lineheight = 0.95,
             colour = pal["neutral_mid"]) +
    scale_fill_manual(values = c(`FALSE` = unname(pal["success"]),
                                 `TRUE` = unname(pal["failure"])),
                      labels = c(`FALSE` = "No pretraining overlap",
                                 `TRUE` = "Shares data with pretraining"),
                      name = NULL) +
    scale_x_discrete(labels = label_wrap_gen(18)) +
    scale_y_continuous(limits = c(0, 0.092),
                       expand = expansion(mult = c(0, 0.1))) +
    labs(x = NULL, y = expression("Best CLAP"-"best hand-crafted "*Delta*rho)) +
    theme(legend.position = "top", legend.margin = margin(b = -3))

  fig <- (p1 | p2) + plot_layout(widths = c(1.6, 1)) +
    plot_annotation(tag_levels = "a") & tag_theme()
  save_ms(fig, file.path(OUT, "ED7_representations"), W_DOUBLE, 70)
}

# ------------------------------------------------------------------- run -----
for (f in list(ed1, ed2, ed3, ed4, ed5, ed6, ed7)) f()
