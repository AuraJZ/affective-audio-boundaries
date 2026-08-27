# =============================================================================
# Figure 4 | Counterfactual audio interventions move both domain models in the
#            same direction, but only at the level of a feature *dimension*.
#
# Layout: hero dose-response with the two model domains as line families,
# plus a manipulation check and a specificity diagnostic. The specificity panel
# is deliberately included even though it weakens the claim: audio interventions
# cannot move one descriptor in isolation, and the figure has to show that.
#
# Evidence chain
#   a  dose-response, both domains       -> the causal direction agrees
#   b  manipulation check                -> the intervention did what it claims
#   c  specificity                       -> what else moved, and how much
#   d  per-excerpt reliability           -> whether the mean shift in (a) is a
#                                           real per-excerpt effect or an average
#                                           over excerpts that disagree
#
# Panel d exists because panel a alone cannot tell those apart, and for two of
# the three interventions the answer is that it cannot. The aggregate curve in
# (a) is computed over 80 excerpts; the earlier version of this analysis
# correlated the four dose means directly, a statistic whose smallest attainable
# two-sided p at n = 4 is 0.083 and which therefore could never have reached
# significance. Panel d replaces it with a test at the level that has power.
# =============================================================================

source("_theme.R")

library(ggrepel)

cur <- read_src("fig_intervention_curves")
met <- read_src("fig_intervention_metrics")
spc <- read_src("fig_intervention_specificity")

iv_lab <- c(spectral_smooth = "Spectral smoothing\n(slower timbre change)",
            transient_inject = "Transient injection\n(higher burst density)",
            envelope_smooth = "Envelope compression\n(failed manipulation)")
iv_lv <- names(iv_lab)

# ------------------------------------------------------- panel a (hero) ------
# Strip the " model" suffix so the values match the names in `pal_domain`;
# otherwise scale_colour_manual silently falls back to grey.
cur <- cur %>%
  mutate(intervention = factor(intervention, levels = iv_lv),
         model_domain = factor(sub(" model$", "", model_domain),
                               levels = c("Music", "Ambient")))

ends <- cur %>% filter(dose == max(dose))

p_a <- ggplot(cur, aes(dose, p_sleep, colour = model_domain)) +
  geom_hline(data = cur %>% filter(dose == 0) %>%
               group_by(intervention) %>% summarise(y = mean(p_sleep), .groups = "drop"),
             aes(yintercept = y), linewidth = 0.3, linetype = "dotted",
             colour = pal["neutral_light"]) +
  geom_line(linewidth = 0.55) +
  geom_point(size = 1.3) +
  geom_text_repel(data = ends, aes(label = paste0(model_domain, " model")),
                  nudge_x = 0.30, direction = "y", hjust = 0, size = 1.85,
                  family = FONT, segment.size = 0.2,
                  segment.colour = pal[["neutral_light"]],
                  min.segment.length = 0, seed = 3) +
  facet_wrap(~ intervention, nrow = 1,
             labeller = labeller(intervention = iv_lab)) +
  scale_colour_manual(values = pal_domain, guide = "none") +
  scale_x_continuous(breaks = 0:3, limits = c(-0.15, 4.6),
                     expand = c(0, 0)) +
  labs(x = "Intervention dose (0 = unmodified control)",
       y = "P(low arousal)") +
  theme(panel.spacing.x = unit(6, "pt"),
        strip.text = element_text(lineheight = 0.9))

# ------------------------------------------------------------------ panel b --
# Manipulation check: did the dose actually move the target descriptor?
mc <- met %>%
  mutate(intervention = factor(intervention, levels = iv_lv),
         ok = manip_consistency >= 0.75 & abs(manip_rho) >= 0.5,
         lab = paste0(target))

p_b <- ggplot(mc, aes(manip_rho, reorder(lab, manip_rho))) +
  annotate("rect", xmin = -0.5, xmax = 0.5, ymin = -Inf, ymax = Inf,
           fill = pal[["failure"]], alpha = 0.07) +
  geom_vline(xintercept = 0, linewidth = 0.3, colour = pal["neutral_mid"]) +
  geom_segment(aes(x = 0, xend = manip_rho, yend = reorder(lab, manip_rho)),
               linewidth = 0.35, colour = pal["neutral_light"]) +
  geom_point(aes(colour = ok, size = manip_consistency)) +
  scale_colour_manual(values = c(`TRUE` = unname(pal["success"]),
                                 `FALSE` = unname(pal["neutral_mid"])),
                      guide = "none") +
  scale_size_continuous(range = c(0.9, 2.3), name = "Consistency\nacross clips",
                        breaks = c(0.6, 0.8, 1.0), labels = percent) +
  scale_x_continuous(limits = c(-1.15, 1.15), breaks = seq(-1, 1, 0.5)) +
  labs(x = expression("Dose"%->%"target descriptor, within-clip "*rho),
       y = NULL) +
  theme(legend.position = "right", legend.key.height = unit(6, "pt"),
        legend.title = element_text(lineheight = 0.9))

# ------------------------------------------------------------------ panel c --
# Specificity: the five descriptors that moved most, per intervention.
spc <- spc %>%
  mutate(intervention = factor(intervention, levels = iv_lv),
         feature = reorder(feature, shift_sd))

iv_short <- c(spectral_smooth = "Spectral\nsmoothing",
              transient_inject = "Transient\ninjection",
              envelope_smooth = "Envelope\ncompression")

# Descriptor names are long enough that three side-by-side facets in 60% of the
# figure width left ~11 mm for the bars: the strip labels were clipped to
# fragments and the axis ticks ran together. Panel c therefore gets a row of its
# own, and the strips are one line rather than two.
iv_short <- c(spectral_smooth = "Spectral smoothing",
              transient_inject = "Transient injection",
              envelope_smooth = "Envelope compression")

p_c <- ggplot(spc, aes(shift_sd, feature, fill = feature_class)) +
  geom_col(width = 0.66) +
  facet_wrap(~ intervention, nrow = 1, scales = "free_y",
             labeller = labeller(intervention = iv_short)) +
  scale_fill_manual(values = pal_class, name = NULL) +
  scale_x_continuous(breaks = c(0, 4, 8),
                     expand = expansion(mult = c(0, 0.12))) +
  labs(x = "Standardised shift at maximum dose (SD)", y = NULL) +
  theme(panel.spacing.x = unit(10, "pt"),
        legend.position = "top", legend.margin = margin(b = -3))


# ------------------------------------------------------------------ panel d --
# The panel that decides how much of (a) may be believed.
#
# Two quantities, deliberately shown together. The permutation p says the median
# excerpt slope differs from zero; the sign consistency says what fraction of
# individual excerpts move the same way. They come apart: transient injection is
# significant on the first and at chance on the second, which is the signature
# of a mean shift produced by a minority of excerpts rather than an effect the
# intervention has on audio in general. Reporting only the p would licence a
# claim the data do not support.
clip <- read_src("intervention_dose_response_clip_level") %>%
  mutate(intervention = factor(intervention, levels = iv_lv),
         model_domain = factor(sub(" model$", "", domain),
                               levels = c("Music", "Ambient")),
         reliable = sign_consistency >= 0.65 & signflip_p < 0.01)

p_d <- ggplot(clip, aes(sign_consistency, -log10(signflip_p),
                        colour = model_domain, shape = reliable)) +
  # Chance is 50% of excerpts, not zero. A point near this line is an average,
  # not an effect, however small its p value.
  geom_vline(xintercept = 0.5, linewidth = 0.3, linetype = "dashed",
             colour = pal[["neutral_mid"]]) +
  annotate("text", x = 0.5, y = Inf, hjust = 0.5, vjust = 1.4,
           size = SZ_LEGEND / .pt, colour = pal[["neutral_mid"]],
           label = "chance") +
  geom_point(size = 2.0, stroke = 0.7) +
  ggrepel::geom_text_repel(
    aes(label = sub("\n.*", "", iv_lab[as.character(intervention)])),
    size = 1.85, family = FONT, segment.size = 0.2, seed = 5,
    segment.colour = pal[["neutral_light"]], min.segment.length = 0,
    show.legend = FALSE) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 1), guide = "none") +
  scale_colour_manual(values = pal_domain, name = NULL) +
  scale_x_continuous(labels = percent_format(1), limits = c(0.40, 0.86)) +
  scale_y_continuous(expand = expansion(mult = c(0.12, 0.14))) +
  labs(title = "Significant is not the same as reliable",
       x = "Excerpts moving in the intended direction",
       y = expression(-log[10]~italic(p)[perm])) +
  theme(legend.position = c(0.14, 0.88))



# ------------------------------------------------------------------ assemble --
# The specificity panel needs the full width: it carries three facets of long
# descriptor names, and at half width the names crowd the points off the panel
# entirely -- the first attempt at this layout produced three columns of text
# with no visible data. The reliability check sits beside the manipulation
# check instead, which still puts it on the same eye-line as the curve above.
fig <- p_a / (p_b | p_d) / p_c
fig <- fig +
  plot_layout(heights = c(1, 0.78, 0.95)) +
  plot_annotation(tag_levels = "a") & tag_theme()

save_ms(fig, file.path("..", "reports", "figures_r", "Fig5_intervention"),
        width_mm = W_DOUBLE, height_mm = 205)

# Guards. The section now rests on the separation between "significant" and
# "reliable", so the build must fail if that separation disappears.
ss <- clip$sign_consistency[clip$intervention == "spectral_smooth"]
ti <- clip$sign_consistency[clip$intervention == "transient_inject"]
stopifnot(all(ss > 0.65), all(ti < 0.62),
          all(clip$signflip_p[clip$intervention == "spectral_smooth"] < 0.01))
message("  Fig4: significant-vs-reliable separation holds")
