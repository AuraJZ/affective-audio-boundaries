# =============================================================================
# Figure 2 (hero) | What a feature-selection route recovers depends on what it
#                   tests: within-domain predictive power or generalisation.
#
# The four routes differ in statistical principle, not hyper-parameters:
#   post-hoc attribution + permutation FDR | cross-domain effect size |
#   additive-model shape functions         | model multiplicity (Rashomon)
#
# NOTE ON FRAMING. An earlier draft of this figure claimed all four routes
# converge on temporal variability. Panel b refutes that: the permutation-FDR
# route sits at 2/12 (below chance), while the three routes that probe
# generalisation or model multiplicity sit at 60-67%. The honest reading is a
# dissociation, not a consensus — routes that test within-domain significance
# recover spectral-level features, which is exactly what Fig. 1-2 show does not
# transfer. The figure is built to make that split visible rather than hide it.
#
# Layout: UpSet-style hero. A plain overlap count hides *which* routes agree;
# the UpSet matrix shows the recovery pattern per feature and the per-route set
# sizes at once.
# =============================================================================

source("_theme.R")

d <- read_src("fig_convergence")
chain_lv <- c("SHAP + permutation FDR", "Cross-domain direction",
              "EBM native importance", "Rashomon consensus")
chain_short <- c("SHAP+FDR", "Cross-domain", "EBM", "Rashomon")
names(chain_short) <- chain_lv

# Order features: most-supported first, then by class.
ord <- d %>%
  distinct(feature, n_chains, feature_class) %>%
  arrange(desc(n_chains), feature_class, feature)

hits <- d %>%
  filter(feature %in% ord$feature) %>%
  mutate(feature = factor(feature, levels = ord$feature),
         evidence_chain = factor(evidence_chain, levels = rev(chain_lv)))

full_grid <- expand.grid(feature = factor(ord$feature, levels = ord$feature),
                         evidence_chain = factor(chain_lv, levels = rev(chain_lv))) %>%
  left_join(hits %>% mutate(hit = TRUE) %>% select(feature, evidence_chain, hit),
            by = c("feature", "evidence_chain")) %>%
  mutate(hit = !is.na(hit)) %>%
  left_join(ord %>% select(feature, feature_class, n_chains), by = "feature")

# ------------------------------------------------- panel a: UpSet matrix -----
p_matrix <- ggplot(full_grid, aes(feature, evidence_chain)) +
  # background shading for readability, alternating by feature
  geom_tile(fill = "white", colour = NA) +
  geom_point(data = filter(full_grid, !hit), size = 1.3,
             colour = pal["neutral_light"]) +
  geom_point(data = filter(full_grid, hit), aes(colour = feature_class),
             size = 1.9) +
  # vertical connectors within a feature
  geom_line(data = filter(full_grid, hit),
            aes(group = feature, colour = feature_class), linewidth = 0.45) +
  scale_colour_manual(values = pal_class, name = NULL) +
  scale_y_discrete(labels = chain_short) +
  scale_x_discrete(expand = expansion(add = 0.6)) +
  labs(x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.line = element_blank(), axis.ticks = element_blank(),
        legend.position = "none",
        plot.margin = margin(1, 3, 3, 3))

# Top bar: how many routes recovered each feature.
p_top <- ggplot(ord, aes(factor(feature, levels = ord$feature), n_chains,
                         fill = feature_class)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = n_chains), vjust = -0.35, size = 1.9, family = FONT,
            colour = pal["neutral_dark"]) +
  scale_fill_manual(values = pal_class, name = NULL) +
  scale_y_continuous(limits = c(0, 4.6), breaks = c(0, 2, 4),
                     expand = c(0, 0)) +
  scale_x_discrete(expand = expansion(add = 0.6)) +
  labs(x = NULL, y = "Routes") +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        axis.line.x = element_blank(),
        legend.position = "top", legend.margin = margin(b = -3),
        plot.margin = margin(3, 3, 1, 3))

# Left bar: set size per route, split by feature class.
setsize <- d %>%
  count(evidence_chain, feature_class) %>%
  mutate(evidence_chain = factor(evidence_chain, levels = rev(chain_lv)))

p_left <- ggplot(setsize, aes(n, evidence_chain, fill = feature_class)) +
  geom_col(width = 0.62) +
  annotate("text", x = 12.4, y = 4.62, label = "Set size", hjust = 0,
           size = 2.0, family = FONT, colour = pal["neutral_mid"]) +
  scale_fill_manual(values = pal_class, guide = "none") +
  scale_x_reverse(expand = c(0, 0), limits = c(13, 0),
                  breaks = c(12, 8, 4, 0)) +
  scale_y_discrete(labels = NULL, expand = expansion(add = 0.62)) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = NULL) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        axis.line.y = element_blank(),
        plot.margin = margin(1, 1, 3, 3))

p_a <- (plot_spacer() + p_top + p_left + p_matrix) +
  plot_layout(ncol = 2, widths = c(0.30, 1), heights = c(0.42, 1))

# --------------------------------------------- panel b: class composition ----
share <- d %>%
  group_by(evidence_chain) %>%
  summarise(n_rate = sum(feature_class == "Temporal variability"),
            n_tot = n(), .groups = "drop") %>%
  mutate(frac = n_rate / n_tot,
         evidence_chain = factor(evidence_chain, levels = rev(chain_lv)))

p_b <- ggplot(share, aes(frac, evidence_chain)) +
  annotate("rect", xmin = 0, xmax = 0.5, ymin = -Inf, ymax = Inf,
           fill = pal[["neutral_light"]], alpha = 0.28) +
  geom_col(fill = pal[["success"]], width = 0.55) +
  geom_vline(xintercept = 0.5, linetype = "dashed", linewidth = 0.3,
             colour = pal["neutral_mid"]) +
  geom_text(aes(label = sprintf("%d/%d", n_rate, n_tot)),
            hjust = -0.25, size = 1.9, family = FONT,
            colour = pal["neutral_dark"]) +
  annotate("text", x = 0.49, y = 0.55, label = "chance", size = 1.8,
           colour = pal["neutral_mid"], hjust = 1, family = FONT) +
  annotate("text", x = 0.25, y = 4.42, label = "tests within-domain significance",
           size = 1.85, colour = pal["neutral_mid"], family = FONT, fontface = "italic") +
  annotate("text", x = 0.78, y = 2.42, label = "test generalisation or model multiplicity",
           size = 1.85, colour = pal["neutral_mid"], family = FONT, fontface = "italic") +
  # A bracket at x = 1.02 spanning the lower three rows was removed: it sat far
  # from the label it was meant to group and read as a stray mark. The two
  # italic annotations carry the grouping on their own.
  scale_x_continuous(limits = c(0, 1.12), expand = c(0, 0),
                     breaks = seq(0, 1, 0.25),
                     labels = percent_format(accuracy = 1)) +
  scale_y_discrete(labels = chain_short, expand = expansion(add = 0.62)) +
  coord_cartesian(clip = "off") +
  labs(x = "Temporal-variability share of recovered features", y = NULL)

# ------------------------------------------------------------------ assemble --
fig <- wrap_elements(p_a) / p_b +
  plot_layout(heights = c(1, 0.45)) +
  plot_annotation(tag_levels = list(c("a", "b"))) & tag_theme()

save_ms(fig, file.path("..", "reports", "figures_r", "Fig3_routes"),
        width_mm = W_DOUBLE, height_mm = 106)
