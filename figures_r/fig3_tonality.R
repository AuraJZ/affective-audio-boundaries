# =============================================================================
# Figure 3 | A gap identified by the data, closed by a targeted representation.
#
# The figure is a closed loop, not a performance table:
#   the data name a deficit -> a falsifiable account is proposed -> the criterion
#   is fixed before the test -> the gain lands where predicted -> the mechanism
#   check dissociates.
#
#   a  eight affect axes, spectro-temporal descriptors alone versus with tonal
#      descriptors added; ordered by gain
#   b  tonal descriptors alone (39) against spectro-temporal alone (122)
#   c  the mechanism check: mode and consonance dissociate across two axes
#
# Panel c is the panel that distinguishes "the gap was tonality" from "39 more
# features helped", so it is drawn at the same weight as a and b rather than as
# a supporting inset.
# =============================================================================

source("_theme.R")

library(ggrepel)

ax  <- read_src("fig_tonality_axes")
rk  <- read_src("fig_tonality_ranks")

# ------------------------------------------------------------------ panel a --
a <- ax %>%
  mutate(axis = factor(axis, levels = rev(axis)),
         grp = ifelse(mode_dependent, "Mode-dependent", "Other"))

p_a <- ggplot(a, aes(y = axis)) +
  geom_segment(aes(x = rho_A, xend = rho_AD, yend = axis, colour = grp),
               linewidth = 1.7, alpha = 0.35, lineend = "round") +
  geom_point(aes(x = rho_A), size = 1.6, shape = 21, fill = "white",
             stroke = 0.6, colour = pal[["neutral_mid"]]) +
  geom_point(aes(x = rho_AD, colour = grp), size = 1.8) +
  geom_text(aes(x = rho_AD, label = sprintf("%+.3f", gain_rho), colour = grp),
            hjust = -0.28, size = 1.9, family = FONT, show.legend = FALSE) +
  scale_colour_manual(values = c("Mode-dependent" = unname(pal["music"]),
                                 "Other" = unname(pal["neutral_mid"])),
                      name = NULL) +
  scale_x_continuous(limits = c(0.40, 0.95), breaks = seq(0.4, 0.9, 0.1)) +
  labs(x = expression("Cross-validated Spearman "*rho), y = NULL,
       title = "Adding tonal descriptors") +
  theme(legend.position = "bottom", legend.key.size = unit(2.6, "mm"),
        legend.margin = margin(t = -1.5, unit = "mm"))

# ------------------------------------------------------------------ panel b --
# The diagonal is the claim: points above it are axes where 39 tonal
# descriptors beat 122 spectro-temporal ones.
p_b <- ggplot(ax, aes(rho_A, rho_D)) +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.35,
              linetype = "dashed", colour = pal["neutral_mid"]) +
  geom_point(aes(colour = mode_dependent), size = 1.9) +
  # Five of the eight axes cluster in the top-right; fixed offsets collide there.
  geom_text_repel(aes(label = axis, colour = mode_dependent),
                  size = 1.9, family = FONT, seed = 1,
                  min.segment.length = 0, segment.size = 0.2,
                  segment.colour = pal[["neutral_light"]],
                  box.padding = 0.22, point.padding = 0.12,
                  max.overlaps = Inf) +
  annotate("text", x = 0.50, y = 0.86, hjust = 0, size = 1.85, family = FONT,
           colour = pal["neutral_mid"], lineheight = 0.95,
           label = "above the line:\n39 tonal descriptors beat\n122 spectro-temporal ones") +
  scale_colour_manual(values = c(`TRUE` = unname(pal["music"]),
                                 `FALSE` = unname(pal["neutral_mid"])),
                      guide = "none") +
  scale_x_continuous(limits = c(0.42, 0.92), breaks = seq(0.5, 0.9, 0.2)) +
  scale_y_continuous(limits = c(0.42, 0.92), breaks = seq(0.5, 0.9, 0.2)) +
  coord_fixed() +
  labs(x = expression("Spectro-temporal only ("*rho*")"),
       y = expression("Tonal only ("*rho*")"),
       title = "Representation, not dimensionality")

# ------------------------------------------------------------------ panel c --
# Rank among all 161 descriptors, log scale: rank 2 versus rank 20 is the
# comparison that matters and it is invisible on a linear axis.
lab <- c(key_major_strength = "Major-key strength",
         mode_score = "Mode (major - minor)",
         key_clarity = "Key clarity",
         chroma_consonance = "Interval consonance",
         roughness_pl_mean = "Sensory roughness")

c_d <- rk %>%
  filter(!is.na(rank)) %>%
  mutate(feature = factor(feature, levels = names(lab)),
         axis = factor(axis, levels = c("happy", "tension"),
                       labels = c("happy", "tense")))

p_c <- ggplot(c_d, aes(rank, feature, colour = axis)) +
  geom_line(aes(group = feature), linewidth = 0.9, alpha = 0.3,
            colour = pal[["neutral_mid"]], lineend = "round") +
  geom_point(size = 2.0) +
  geom_text(aes(label = rank), vjust = -1.15, size = 1.85, family = FONT,
            show.legend = FALSE) +
  scale_colour_manual(values = c("happy" = unname(pal["music"]),
                                 "tense" = unname(pal["ambient"])), name = NULL) +
  scale_x_log10(breaks = c(1, 3, 10, 30, 100), limits = c(0.7, 200)) +
  scale_y_discrete(labels = lab, limits = rev(names(lab)),
                   expand = expansion(add = c(0.6, 0.9))) +
  labs(x = "Rank among all 161 descriptors (log scale)", y = NULL,
       title = "Mechanism check") +
  theme(legend.position = "bottom", legend.key.size = unit(2.6, "mm"),
        legend.margin = margin(t = -1.5, unit = "mm"),
        panel.grid.major.x = element_line(linewidth = 0.2,
                                          colour = pal[["neutral_light"]]))

# ------------------------------------------------------------------ assemble --
design <- "
AB
CC
"
fig <- p_a + p_b + p_c +
  plot_layout(design = design, widths = c(1.12, 1), heights = c(1, 0.78)) +
  plot_annotation(tag_levels = "a") & tag_theme()

save_ms(fig, file.path("..", "reports", "figures_r", "Fig4_tonality"),
        width_mm = W_DOUBLE, height_mm = 118)
