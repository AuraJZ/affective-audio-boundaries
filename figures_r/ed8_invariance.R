# =============================================================================
# Figure 2 (hero) | The only cross-domain-invariant correlates of low arousal
#                   are temporal-variability measures, not spectral timbre.
#
# Layout: hero scatter with marginal densities, plus a dumbbell panel.
# Dumbbells replace paired bars: with only two series per feature the bar pair
# wastes ink and the reader has to compare bar *lengths* across a gap, whereas
# the dumbbell shows the gap itself.
#
# Evidence chain
#   a  effect size in one domain vs the other, with marginals -> which features
#      keep their sign, and how the two classes separate
#   b  ranked sign-preserving features                        -> effect magnitude
# =============================================================================

source("_theme.R")

library(ggrepel)

d <- read_src("fig_invariance")
lim <- 1.8

# ------------------------------------------------------- panel a (hero) ------
quad <- data.frame(xmin = c(-lim, 0), xmax = c(0, lim),
                   ymin = c(-lim, 0), ymax = c(0, lim))

lab_pts <- d %>% filter(same_direction, min_abs_d > 0.35)

p_main <- ggplot(d, aes(d_music_domain, d_ambient_domain)) +
  geom_rect(data = quad, inherit.aes = FALSE,
            aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            fill = pal[["success"]], alpha = 0.06) +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = pal["neutral_mid"]) +
  geom_vline(xintercept = 0, linewidth = 0.3, colour = pal["neutral_mid"]) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted",
              linewidth = 0.3, colour = pal["neutral_light"]) +
  geom_point(aes(colour = feature_class, shape = same_direction),
             size = 1.7, stroke = 0.55) +
  geom_text_repel(data = lab_pts, aes(label = feature),
                  size = 1.75, family = FONT, colour = pal[["neutral_dark"]],
                  segment.size = 0.2, segment.colour = pal[["neutral_light"]],
                  min.segment.length = 0, box.padding = 0.28,
                  max.overlaps = 20, seed = 7) +
  annotate("text", x = -lim + 0.08, y = -lim + 0.12, label = "sign preserved",
           hjust = 0, size = 2.0, colour = pal[["success"]],
           fontface = "bold", family = FONT) +
  scale_colour_manual(values = pal_class, name = NULL) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 1),
                     labels = c(`TRUE` = "Sign preserved", `FALSE` = "Sign reversed"),
                     name = NULL) +
  coord_fixed(xlim = c(-lim, lim), ylim = c(-lim, lim), expand = FALSE) +
  labs(x = "Cohen's d, music domain", y = "Cohen's d, ambient domain") +
  guides(colour = guide_legend(order = 1, override.aes = list(shape = 16)),
         shape = guide_legend(order = 2)) +
  theme(legend.position = c(0.98, 0.98), legend.justification = c(1, 1),
        legend.spacing.y = unit(0, "pt"), legend.key.height = unit(6, "pt"))

# Marginal densities carry the actual message: the two classes sit in
# different places on both axes.
marg <- function(var, flip = FALSE) {
  p <- ggplot(d, aes(.data[[var]], fill = feature_class, colour = feature_class)) +
    geom_density(alpha = 0.32, linewidth = 0.3, bw = 0.32) +
    scale_fill_manual(values = pal_class, guide = "none") +
    scale_colour_manual(values = pal_class, guide = "none") +
    scale_x_continuous(limits = c(-lim, lim), expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    theme_void()
  if (flip) p + coord_flip(xlim = c(-lim, lim), expand = FALSE) else p
}

p_a <- (marg("d_music_domain") + plot_spacer() +
        p_main + marg("d_ambient_domain", flip = TRUE)) +
  plot_layout(ncol = 2, widths = c(1, 0.16), heights = c(0.16, 1))

# ------------------------------------------------------------------ panel b --
top <- d %>%
  filter(same_direction) %>%
  slice_max(min_abs_d, n = 8) %>%
  mutate(feature = reorder(feature, min_abs_d),
         lo = pmin(abs(d_music_domain), abs(d_ambient_domain)),
         hi = pmax(abs(d_music_domain), abs(d_ambient_domain)))

pts <- top %>%
  pivot_longer(c(d_music_domain, d_ambient_domain),
               names_to = "domain", values_to = "d") %>%
  mutate(domain = ifelse(domain == "d_music_domain", "Music", "Ambient"),
         domain = factor(domain, levels = c("Music", "Ambient")))

p_b <- ggplot(top, aes(y = feature)) +
  geom_segment(aes(x = lo, xend = hi, yend = feature),
               linewidth = 1.5, colour = pal["neutral_light"],
               lineend = "round") +
  geom_point(data = pts, aes(abs(d), feature, colour = domain), size = 1.9) +
  geom_text(aes(x = hi, label = sprintf("%.2f", min_abs_d)),
            hjust = -0.4, size = 1.85, family = FONT,
            colour = pal["neutral_mid"]) +
  scale_colour_manual(values = pal_domain, name = NULL) +
  scale_x_continuous(limits = c(0, 1.6), breaks = seq(0, 1.5, 0.5),
                     expand = c(0, 0)) +
  labs(x = "|Cohen's d|", y = NULL) +
  theme(legend.position = c(0.99, 0.04), legend.justification = c(1, 0),
        legend.direction = "horizontal")

# ------------------------------------------------------------------ assemble --
fig <- wrap_elements(p_a) | p_b
fig <- fig + plot_layout(widths = c(1.15, 1)) +
  plot_annotation(tag_levels = list(c("a", "b"))) & tag_theme()

save_ms(fig, file.path("..", "reports", "figures_r", "ED8_invariance"),
        width_mm = W_DOUBLE, height_mm = 82)
