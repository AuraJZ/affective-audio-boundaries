# =============================================================================
# Figure 6 | The individual-level question is a design problem, not a null result.
#
#   a  MEASURED detection against observations per person, in real physiology:
#      75 listeners, three independent channels, an effect that is certainly
#      present (auditory evoked response, group p = 0.0002)
#   b  simulated power for the association strengths the paper is actually
#      about, with the band a laboratory session supplies
#   c  requirement against supply, simulated and measured on one axis
#
# Panel a is new and carries the figure. Everything downstream of it in the
# Discussion used to rest on a simulation; it now rests on a measurement.
# The number that converts "we did not detect it" into "this design cannot
# detect it" is no longer an assumption about covariance structure.
# =============================================================================

source("_theme.R")

# The hires grid (500 replicates per cell, n extended to 5,000) supersedes the
# original 60-replicate run. The two weakest effects were previously reportable
# only as "> 1,000"; they now resolve at 1,364 and 4,888, so the capping
# machinery below is no longer needed and would actively misreport them.
pw    <- read_src("design_power_grid_hires")
need  <- read_src("design_n_required_hires")
meas  <- read_src("measured_trialcount_curve")
mreq  <- read_src("measured_requirement")

SESSION <- 30      # this corpus
PUBLIC  <- 60      # comparable public corpora
NIGHT   <- 480     # 1-minute windows over an 8-hour recording

# Channel colours: EEG on the signal family, the two peripheral channels on
# neutral and accent. Heart rate gets the failure accent deliberately -- it is
# the channel a consumer device actually has, and it is the most expensive.
pal_chan <- c("Scalp EEG (N1)"   = unname(pal["music"]),
              "Pupil diameter"   = unname(pal["neutral_mid"]),
              "Heart rate"       = unname(pal["failure"]))
CH_ORDER <- names(pal_chan)

meas <- meas %>% mutate(channel = factor(channel, levels = CH_ORDER))

# ------------------------------------------------------------------ panel a --
lastm <- meas %>% group_by(channel) %>% slice_max(n, n = 1) %>% ungroup()
fp    <- meas %>% group_by(n) %>% summarise(false_pos = mean(false_pos),
                                            .groups = "drop")

p_a <- ggplot(meas, aes(n, detect, colour = channel)) +
  annotate("rect", xmin = 20, xmax = PUBLIC, ymin = -Inf, ymax = Inf,
           fill = pal[["failure"]], alpha = 0.06) +
  annotate("text", x = 34, y = 0.99, label = "laboratory\nsession",
           size = 1.85, family = FONT, lineheight = 0.95,
           colour = pal[["failure"]]) +
  geom_hline(yintercept = 0.8, linewidth = 0.35, linetype = "dashed",
             colour = pal["neutral_mid"]) +
  annotate("text", x = 10, y = 0.845, hjust = 0, label = "80% detection",
           size = 1.8, family = FONT, colour = pal["neutral_mid"]) +
  # false-positive floor: the curves rise, this does not
  geom_line(data = fp, aes(n, false_pos), inherit.aes = FALSE,
            linewidth = 0.4, linetype = "dotted",
            colour = pal["neutral_mid"]) +
  annotate("text", x = 62, y = 0.015, hjust = 0,
           label = "false positives (sign-flip null)", size = 1.75,
           family = FONT, colour = pal["neutral_mid"]) +
  geom_line(aes(y = analytic), linewidth = 0.9, alpha = 0.22,
            show.legend = FALSE) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 1.1) +
  geom_text(data = lastm, aes(label = channel), hjust = 1, vjust = -1.1,
            size = 1.9, family = FONT, show.legend = FALSE) +
  scale_colour_manual(values = pal_chan, guide = "none") +
  scale_x_log10(breaks = c(10, 20, 30, 50, 100, 200),
                limits = c(9, 260)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                     labels = percent) +
  labs(x = "Observations per person (log scale)",
       y = "Individuals in whom the effect is detected",
       title = "Measured, in real physiology")

# ------------------------------------------------------------------ panel b --
pw <- pw %>% mutate(r_lab = sprintf("r = %.2f", r_true))
lastp <- pw %>% group_by(r_lab) %>% slice_max(n_trials, n = 1) %>% ungroup()

p_b <- ggplot(pw, aes(n_trials, power, colour = r_lab)) +
  annotate("rect", xmin = 20, xmax = PUBLIC, ymin = -Inf, ymax = Inf,
           fill = pal[["failure"]], alpha = 0.06) +
  geom_hline(yintercept = 0.8, linewidth = 0.35, linetype = "dashed",
             colour = pal["neutral_mid"]) +
  annotate("text", x = 22, y = 0.845, hjust = 0, label = "80% power",
           size = 1.8, family = FONT, colour = pal["neutral_mid"]) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 1.1) +
  geom_text(data = lastp, aes(label = r_lab), hjust = 1.05, vjust = -0.9,
            size = 1.9,
            family = FONT, show.legend = FALSE) +
  scale_colour_manual(values = colorRampPalette(
    c(pal[["neutral_mid"]], pal[["music"]]))(4), guide = "none") +
  scale_x_log10(breaks = c(30, 120, 500, 2000, 5000),
                limits = c(20, 6000)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                     labels = percent) +
  labs(x = "Observations per person (log scale)",
       y = "Probability of detection",
       title = "Simulated, for the target association")

# ------------------------------------------------------------------ panel c --
# Requirements above the simulated range are drawn at the range edge with an
# open marker: the simulation bounds them below, it does not resolve them.
# No cap. Every requirement in the hires grid is resolved, so an open marker at
# a range edge would now be inventing a bound the simulation does not need.
nd <- need %>%
  mutate(capped = !is.finite(n_required_80pct),
         n_plot = n_required_80pct,
         lab = format(round(n_required_80pct), big.mark = ","),
         row = sprintf("r = %.2f", r_true),
         kind = "Simulated (target association)")
stopifnot(!any(nd$capped))

md <- mreq %>% filter(group == "All") %>%
  mutate(capped = FALSE, n_plot = n_required_80pct,
         lab = format(round(n_required_80pct), big.mark = ","),
         row = channel, kind = "Measured (evoked response)")

# Measured channels on top, simulated below, cheapest first within each block.
# ggplot puts the FIRST discrete level at the bottom, so the vector is built
# bottom-up. Ties are broken by r (two simulated rows are both capped at 1,000
# and would otherwise order arbitrarily).
nd <- nd %>% arrange(n_plot, desc(r_true))
md <- md %>% arrange(n_plot)
ord <- c(rev(nd$row), rev(md$row))
both <- bind_rows(md[, c("row", "kind", "n_plot", "capped", "lab")],
                  nd[, c("row", "kind", "n_plot", "capped", "lab")]) %>%
  mutate(row = factor(row, levels = ord))

supply <- data.frame(
  what = c("This corpus", "Public corpora", "One night's recording"),
  n = c(SESSION, PUBLIC, NIGHT))

p_c <- ggplot(both, aes(n_plot, row)) +
  geom_vline(data = supply, aes(xintercept = n), linewidth = 0.35,
             linetype = "dotted", colour = pal["neutral_mid"]) +
  geom_segment(aes(x = 9, xend = n_plot, yend = row),
               linewidth = 0.35, colour = pal["neutral_light"]) +
  geom_point(aes(shape = capped, colour = kind), size = 2.1,
             fill = "white", stroke = 0.7) +
  geom_text(aes(label = lab), hjust = -0.25, size = 1.9, family = FONT,
            colour = pal["neutral_dark"]) +
  # Horizontal, above the panel. Rotated 90 degrees these ran down through the
  # data rows and collided with the point labels; a reference line's caption
  # must not sit on top of the thing it is a reference for.
  geom_text(data = supply, aes(x = n, y = nrow(both) + 1.15, label = what),
            hjust = 0.5, vjust = 0, size = 1.8, family = FONT,
            colour = pal["neutral_mid"], inherit.aes = FALSE) +
  scale_shape_manual(values = c(`FALSE` = 19, `TRUE` = 21), guide = "none") +
  scale_colour_manual(values = c("Measured (evoked response)" =
                                   unname(pal["failure"]),
                                 "Simulated (target association)" =
                                   unname(pal["music"])),
                      name = NULL) +
  scale_x_log10(breaks = c(10, 30, 100, 300, 1000, 3000),
                limits = c(9, 9000)) +
  scale_y_discrete(expand = expansion(add = c(0.6, 2.4))) +
  coord_cartesian(clip = "off") +
  labs(x = "Observations per person required for 80% detection (log scale)",
       y = NULL, title = "Requirement against supply") +
  # Top-right: the bottom-right corner carries the "> 1,000" labels.
  theme(legend.position = c(0.99, 0.99), legend.justification = c(1, 1),
        legend.key.height = unit(6, "pt"))

# ------------------------------------------------------------------ assemble --

# ------------------------------------------------------------------ panel d --
# Panels a-c answer "how many observations to DETECT a within-person effect".
# An application asks something narrower and more useful: at what point does
# giving a listener their own slope beat giving them the population slope. That
# break-even is a different quantity and, unlike the detection requirement, it
# is not one number -- it varies by a factor of five across settings.
#
# The null column is plotted, not just subtracted. A between-person spread
# estimated from few observations per person is biased upward, and at 18
# observations the estimator returns 0.11 when the truth is zero; a reader who
# cannot see that cannot judge the physiological row.
tau <- read_src("canonical_tau_table")

DOM <- c("主观 · 声景（产品域）" = "Urban soundscapes,\nratings",
         "主观 · 音乐"          = "Music,\nratings",
         "生理 · 音乐"          = "Music,\nelectrodermal")

td <- tau %>%
  mutate(setting = DOM[domain]) %>%
  filter(!is.na(setting)) %>%
  group_by(setting) %>%
  summarise(n_per = median(n_per_unit),
            tau_raw = median(tau_raw), null = median(null),
            tau_c = median(tau),
            nstar = median(n_star[is.finite(n_star)]),
            usable = sum(usable_now), cells = dplyr::n(), .groups = "drop") %>%
  mutate(setting = factor(setting, levels = rev(DOM)))

p_d <- ggplot(td, aes(y = setting)) +
  geom_segment(aes(x = null, xend = tau_raw, yend = setting),
               linewidth = 0.9, colour = pal[["neutral_light"]]) +
  geom_point(aes(x = null), shape = 4, size = 2.0, stroke = 0.8,
             colour = pal[["failure"]]) +
  geom_point(aes(x = tau_raw), size = 2.2, colour = pal[["music"]]) +
  geom_text(aes(x = pmax(tau_raw, null) + 0.012,
                label = ifelse(is.finite(nstar),
                               sprintf("n* = %.0f   (%d/%d cells already past)",
                                       nstar, usable, cells),
                               sprintf("n* undefined   (%d/%d cells)",
                                       usable, cells))),
            hjust = 0, size = SZ_LEGEND / .pt,
            colour = pal[["neutral_dark"]]) +
  # The caption belongs beside the markers it names. Placed top-right it sat
  # above the longest bar and read as a label for the n* text instead.
  annotate("text", x = 0.006, y = 3.55, hjust = 0, size = SZ_LEGEND / .pt,
           colour = pal[["failure"]],
           label = "×  the estimator's own null") +
  scale_x_continuous(limits = c(0, 0.40), breaks = seq(0, .3, .1)) +
  coord_cartesian(clip = "off", ylim = c(0.6, 3.75)) +
  labs(title = "When per-person calibration starts to pay",
       x = "Between-person SD of the acoustic slope", y = NULL)

fig <- ((p_a | p_b) / p_c / p_d) +
  plot_layout(heights = c(1, 0.92, 0.62)) +
  plot_annotation(tag_levels = "a") & tag_theme()

stopifnot(
  abs(td$nstar[grepl("soundscape", td$setting)] - 67) < 4,
  abs(td$nstar[grepl("Music,
ratings", td$setting)] - 166) < 8,
  # The physiological row's raw estimate must not exceed its own null, or the
  # withdrawn tau = 0.106 has crept back in.
  td$tau_raw[grepl("electrodermal", td$setting)] <=
    td$null[grepl("electrodermal", td$setting)] + 0.03
)
message("  Fig6: per-domain break-even and null checks passed")

save_ms(fig, file.path("..", "reports", "figures_r", "Fig7_design"),
        width_mm = W_DOUBLE, height_mm = 196)
