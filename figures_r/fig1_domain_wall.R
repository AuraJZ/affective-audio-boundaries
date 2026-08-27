# =============================================================================
# Figure 1 (hero) | Acoustic arousal models transfer within a sound domain but
#                   not across domains.
#
# Layout: asymmetric hero panel (skill pattern 15). The hero is a slope graph —
# every algorithm walks from within-corpus, to same-domain transfer, to
# cross-domain transfer. The cliff between the second and third position *is*
# the finding, so it gets the largest visual footprint.
#
# Evidence chain
#   a  hero slope graph        -> the collapse, per algorithm, all at once
#   b  transfer matrix         -> which specific corpus pairs fail
#   c  directional asymmetry   -> the barrier is not equally dense in the two
#                                 directions, and its density in the outward
#                                 direction tracks how much atmospheric
#                                 material the source corpus contains
#   d  duration control        -> the collapse is not a clip-length artefact
#   e  leave-one-category-out  -> the model tolerates unseen content, so the
#                                 collapse is specific to the domain border
#
# Four corpora. Soundtracks (Eerola & Vuoskoski 2011) is independent of the
# other two music corpora in collection site, decade, rating protocol and
# material, so it tests whether within-music transfer merely reflects a shared
# annotation tradition.
# =============================================================================

source("_theme.R")

library(ggrepel)

d <- read_src("fig_domain_wall")
asym <- read_src("fig_wall_asymmetry")
corpus_lv <- c("deam", "pmemo", "soundtracks", "emo_mix")

stage_lv <- c("Within-corpus", "Same-domain\ntransfer", "Cross-domain\ntransfer")

long <- d %>%
  mutate(stage = case_when(
    transfer == "within" ~ stage_lv[1],
    domain_match == "same-domain" ~ stage_lv[2],
    TRUE ~ stage_lv[3]),
    stage = factor(stage, levels = stage_lv))

algo_mean <- long %>%
  group_by(algorithm, stage) %>%
  summarise(rho = mean(spearman), .groups = "drop")

# ------------------------------------------------------- panel a (hero) ------
# Band marking the region the models fall into once the domain changes.
p_a <- ggplot() +
  annotate("rect", xmin = 2.5, xmax = 3.5, ymin = -0.45, ymax = 0.9,
           fill = pal[["failure"]], alpha = 0.055) +
  geom_hline(yintercept = 0, linewidth = 0.3, linetype = "dashed",
             colour = pal["neutral_mid"]) +
  # individual corpus pairs, faint
  geom_point(data = long, aes(as.numeric(stage), spearman),
             position = position_jitter(width = 0.09, height = 0, seed = 1),
             size = 0.55, colour = pal["neutral_light"]) +
  # one line per algorithm
  geom_line(data = algo_mean,
            aes(as.numeric(stage), rho, group = algorithm),
            linewidth = 0.45, colour = pal["music"], alpha = 0.85) +
  geom_point(data = algo_mean, aes(as.numeric(stage), rho),
             size = 1.3, colour = pal["music"]) +
  # direct labels instead of a legend (skill pattern 16)
  geom_text_repel(data = filter(algo_mean, stage == stage_lv[1]),
                  aes(as.numeric(stage), rho, label = algorithm),
                  nudge_x = -0.42, direction = "y", hjust = 1, size = 2.0,
                  segment.size = 0.2, segment.colour = pal[["neutral_light"]],
                  min.segment.length = 0, family = FONT) +
  annotate("text", x = 3, y = 0.42, label = "transfer\ncollapses",
           size = 2.2, colour = pal[["failure"]], fontface = "bold",
           lineheight = 0.9, family = FONT) +
  annotate("segment", x = 3, xend = 3, y = 0.30, yend = 0.12,
           linewidth = 0.35, colour = pal[["failure"]],
           arrow = arrow(length = unit(2.2, "pt"), type = "closed")) +
  scale_x_continuous(breaks = 1:3, labels = stage_lv,
                     limits = c(0.15, 3.5), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-0.45, 0.9), breaks = seq(-0.4, 0.8, 0.2)) +
  labs(x = NULL, y = expression("Spearman "*rho),
       title = "Generalisation stages") +
  theme(axis.text.x = element_text(lineheight = 0.85))

# ------------------------------------------------------------------ panel b --
# Mean over the six algorithms, not a single one: panel c reports algorithm
# means, and a single-algorithm matrix would print a different number for the
# same cell (XGBoost puts Soundtracks -> ambient at 0.38 against a mean of
# 0.24).
mat <- d %>%
  group_by(source, target) %>%
  summarise(spearman = mean(spearman), .groups = "drop") %>%
  mutate(source = factor(source, levels = corpus_lv),
         target = factor(target, levels = corpus_lv))

p_b <- ggplot(mat, aes(target, source, fill = spearman)) +
  geom_tile(colour = "white", linewidth = 0.7) +
  geom_text(aes(label = sprintf("%.2f", spearman),
                colour = abs(spearman) > 0.55), size = 2.0, family = FONT) +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = pal[["neutral_dark"]]),
                      guide = "none") +
  scale_fill_gradient2(low = pal[["failure"]], mid = "white",
                       high = pal[["success"]], midpoint = 0,
                       limits = c(-0.35, 0.85), name = expression(rho),
                       breaks = c(-0.3, 0, 0.4, 0.8)) +
  scale_x_discrete(labels = corpus_short[corpus_lv], expand = c(0, 0)) +
  scale_y_discrete(labels = corpus_short[corpus_lv], expand = c(0, 0),
                   limits = rev(corpus_lv)) +
  coord_fixed() +
  labs(x = "Tested on", y = "Trained on", title = "Transfer matrix") +
  theme(legend.key.width = unit(4, "pt"), legend.key.height = unit(12, "pt"),
        axis.line = element_blank(), axis.ticks = element_blank(),
        # Four square tiles are narrower than the corpus names; angling the
        # labels is the only way to keep them from running together.
        axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1))

# ------------------------------------------------------- panel c: asymmetry --
# Outward (music -> ambient) and inward (ambient -> music) transfer for each
# music corpus. The inward direction is uniformly sealed; the outward direction
# is not, and rises with the amount of atmospheric material in the source.
asy <- asym %>%
  mutate(corpus = factor(corpus, levels = corpus[order(to_ambient)])) %>%
  pivot_longer(c(to_ambient, from_ambient), names_to = "dir", values_to = "rho") %>%
  mutate(dir = factor(dir, levels = c("from_ambient", "to_ambient"),
                      labels = c("Ambient → music", "Music → ambient")))

p_c <- ggplot(asy, aes(rho, corpus)) +
  annotate("rect", xmin = -Inf, xmax = Inf,
           ymin = -Inf, ymax = Inf, fill = NA) +
  geom_vline(xintercept = asym$same_domain_reference[1], linewidth = 0.4,
             linetype = "dashed", colour = pal["neutral_mid"]) +
  geom_vline(xintercept = 0, linewidth = 0.3, colour = pal["neutral_light"]) +
  geom_line(aes(group = corpus), linewidth = 0.9, alpha = 0.28,
            colour = pal[["neutral_mid"]], lineend = "round") +
  geom_point(aes(colour = dir), size = 1.8) +
  geom_text(data = filter(asy, dir == "Music → ambient"),
            aes(label = sprintf("%+.2f", rho)), hjust = -0.35, vjust = -0.5,
            size = 1.85, family = FONT, colour = pal[["ambient"]]) +
  annotate("text", x = asym$same_domain_reference[1], y = 3.62, hjust = 1.06,
           label = "same-domain mean", size = 1.8, family = FONT,
           colour = pal["neutral_mid"]) +
  scale_colour_manual(values = c("Music → ambient" = unname(pal["ambient"]),
                                 "Ambient → music" = unname(pal["music"])),
                      name = NULL) +
  scale_y_discrete(labels = corpus_short, expand = expansion(add = c(0.5, 0.9))) +
  scale_x_continuous(limits = c(-0.2, 0.72), breaks = seq(-0.2, 0.6, 0.2)) +
  labs(x = expression("Cross-domain "*rho), y = NULL,
       title = "Directional asymmetry") +
  theme(legend.position = "bottom",
        legend.key.size = unit(2.4, "mm"),
        legend.margin = margin(t = -1.5, unit = "mm"))

# ------------------------------------------------------------------ panel d --
ctl <- read_src("ed_wall_controls")

dur <- ctl %>%
  filter(control == "C1_duration") %>%
  mutate(len = ifelse(grepl("6s$", setting), "6 s", "30 s"),
         kind = factor(kind, levels = c("same_domain", "cross_domain"),
                       labels = c("Same domain", "Cross domain"))) %>%
  group_by(len, kind) %>%
  summarise(rho = mean(spearman), .groups = "drop") %>%
  pivot_wider(names_from = len, values_from = rho)

# Dumbbell: 30 s -> 6 s. Flat dumbbells mean duration changes nothing.
p_d <- ggplot(dur, aes(y = kind)) +
  geom_vline(xintercept = 0, linewidth = 0.3, linetype = "dashed",
             colour = pal["neutral_mid"]) +
  geom_segment(aes(x = `30 s`, xend = `6 s`, yend = kind, colour = kind),
               linewidth = 1.6, alpha = 0.35, lineend = "round") +
  geom_point(aes(x = `30 s`, colour = kind), size = 1.6) +
  geom_point(aes(x = `6 s`, colour = kind), size = 1.6, shape = 21,
             fill = "white", stroke = 0.6) +
  geom_text(aes(x = pmax(`30 s`, `6 s`), label = sprintf("%.2f → %.2f", `30 s`, `6 s`)),
            hjust = -0.14, size = 1.9, colour = pal["neutral_dark"], family = FONT) +
  scale_colour_manual(values = c("Same domain" = unname(pal["success"]),
                                 "Cross domain" = unname(pal["failure"])),
                      guide = "none") +
  scale_x_continuous(limits = c(-0.05, 1.15), breaks = seq(0, 0.8, 0.2)) +
  labs(x = expression("Transfer "*rho), y = NULL,
       title = "Duration control")

# ------------------------------------------------------------------ panel e --
loco <- ctl %>%
  filter(control == "C2_within_domain_LOCO") %>%
  mutate(setting = reorder(setting, spearman))

cross_mean <- ctl %>%
  filter(control == "C1_duration", kind == "cross_domain") %>%
  pull(spearman) %>% mean()

p_e <- ggplot(loco, aes(spearman, setting)) +
  annotate("rect", xmin = -0.02, xmax = cross_mean, ymin = -Inf, ymax = Inf,
           fill = pal[["failure"]], alpha = 0.07) +
  geom_segment(aes(x = 0, xend = spearman, yend = setting),
               linewidth = 0.35, colour = pal["neutral_light"]) +
  geom_point(size = 1.7, colour = pal["ambient"]) +
  annotate("text", x = cross_mean + 0.02, y = 6.4, label = "cross-domain level",
           size = 1.8, colour = pal["failure"], hjust = 0, family = FONT) +
  scale_x_continuous(limits = c(-0.02, 0.86), breaks = seq(0, 0.8, 0.2),
                     expand = c(0, 0)) +
  labs(x = expression("Held-out category "*rho), y = NULL,
       title = "Within-domain generalisation")

# ------------------------------------------------------------------ assemble --
# ------------------------------------------------------------------ panel f --
# The load-bearing panel of this figure, and the one the narrative turns on.
#
# The claim is not "the barrier is high" but "the barrier is in a particular
# place". So the panel shows, per representation, the SAME operation costed
# twice: swap the corpus inside the domain, and swap it across. Reading the two
# marks against each other is the whole point, which is why they share a row and
# are joined by a segment rather than being two separate bar charts.
#
# CLAP is drawn but set apart. Its pretraining data contain the source of the
# environmental corpus, and that overlap was declared before any of these
# numbers existed. Dropping it would hide a result; merging it with the rest
# would let a contaminated family carry a claim. It gets its own block.
fam <- read_src("ast_domain_wall")

FAM_LAB <- c(ast = "AST\n(AudioSet, supervised)",
             mert = "MERT\n(music, self-supervised)",
             w2v2 = "Wav2Vec2\n(speech, self-supervised)",
             clap = "CLAP\n(audio-text; contaminated)")

fam_wall <- fam %>%
  mutate(bucket = case_when(
    domain_match == "within-corpus" ~ "within",
    domain_match %in% c("within-domain", "same-domain") ~ "same",
    TRUE ~ "cross")) %>%
  group_by(family, bucket) %>%
  summarise(rho = median(rho), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = bucket, values_from = rho) %>%
  mutate(loss_same = 1 - same / within,
         loss_cross = 1 - cross / within,
         label = FAM_LAB[family],
         contaminated = family == "clap")

# The hand-specified descriptors are the reference the families are judged
# against, so they are computed from the same table the rest of the figure uses.
hand <- long %>%
  group_by(stage) %>%
  summarise(rho = median(spearman), .groups = "drop")
hand_w <- setNames(hand$rho, c("within", "same", "cross")[
  match(hand$stage, stage_lv)])
fam_wall <- bind_rows(
  tibble(family = "hand", within = hand_w[["within"]],
         same = hand_w[["same"]], cross = hand_w[["cross"]],
         loss_same = 1 - hand_w[["same"]] / hand_w[["within"]],
         loss_cross = 1 - hand_w[["cross"]] / hand_w[["within"]],
         label = "Hand-specified\n(161 descriptors)", contaminated = FALSE),
  fam_wall)

fam_wall <- fam_wall %>%
  mutate(label = factor(label, levels = rev(c(
    "Hand-specified\n(161 descriptors)", FAM_LAB[["ast"]], FAM_LAB[["mert"]],
    FAM_LAB[["w2v2"]], FAM_LAB[["clap"]]))))

# Key for panel f, drawn on a reserved row above the bars rather than floated
# over them. y = 5.9 sits above the topmost category (5) inside the expanded
# ylim; the two entries are spaced far enough apart that the first label cannot
# reach the second swatch at any plausible text size.
key <- tibble(
  x   = c(0.02, 0.42),
  y   = c(5.9, 5.9),
  lab = c("swap corpus, same domain", "swap corpus, across domains"),
  col = c(unname(pal[["neutral_mid"]]), unname(pal[["failure"]]))
)

p_f <- ggplot(fam_wall, aes(y = label)) +
  geom_segment(aes(x = loss_same, xend = loss_cross, yend = label,
                   colour = contaminated),
               linewidth = 0.9, alpha = 0.55, show.legend = FALSE) +
  geom_point(aes(x = loss_same), colour = pal[["neutral_mid"]], size = 2.1) +
  geom_point(aes(x = loss_cross, colour = contaminated), size = 2.1,
             show.legend = FALSE) +
  geom_text(aes(x = loss_cross,
                label = sprintf("%+.0f pp", (loss_cross - loss_same) * 100)),
            hjust = -0.28, size = SZ_LEGEND / .pt,
            colour = pal[["neutral_dark"]]) +
  scale_colour_manual(values = c(`FALSE` = unname(pal[["failure"]]),
                                 `TRUE` = unname(pal[["neutral_light"]]))) +
  scale_x_continuous(labels = percent_format(1), limits = c(0, 1.12),
                     breaks = seq(0, 1, .25)) +
  # 🔴 These two labels used to be free-floating text at y = 5.42, x = 0.20 and
  # x = 0.94. Both collided with the top row: the dumbbell for the hand-specified
  # descriptors runs from 0.20 to 0.94, so the labels were pinned to exactly the
  # two points they were naming and printed on top of them. No y-offset fixes
  # that -- the collision is in x, and the bar is as long as the panel.
  #
  # They are now a proper key on a reserved row above the data, anchored to the
  # left where no bar starts, with a swatch each instead of a position that
  # pretends to point at something.
  geom_point(data = key, aes(x = x, y = y), colour = key$col,
             size = 2.1, inherit.aes = FALSE) +
  geom_text(data = key, aes(x = x + 0.025, y = y, label = lab), hjust = 0,
            size = SZ_LEGEND / .pt, colour = pal[["neutral_dark"]],
            inherit.aes = FALSE) +
  coord_cartesian(clip = "off", ylim = c(0.5, 6.1)) +
  labs(title = "Every representation collapses at the same place, not at the same height",
       # The two colours are now named by the key inside the panel, so the
       # subtitle only has to explain the printed number.
       subtitle = "printed value, the gap between the two swaps",
       x = "Performance lost, as a fraction of within-corpus", y = NULL) +
  theme(plot.subtitle = element_text(size = SZ_LEGEND,
                                     colour = pal[["neutral_mid"]],
                                     margin = margin(b = 3)),
        axis.line.y = element_blank(), axis.ticks.y = element_blank())

# Asymmetric hero: the slope graph carries the claim and spans the upper two
# rows; the matrix and the asymmetry sit beside it because they qualify it, and
# the two controls run along the bottom because they only defend it.
# F spans the full width and comes LAST: with it in the third row the tags read
# a, b, c, f, d, e down the page.
design <- "
AB
AC
DE
FF
"
fig <- p_a + p_b + p_c + p_d + p_e + p_f +
  plot_layout(design = design, widths = c(1.2, 1),
              heights = c(1.0, 1.0, 0.72, 0.95)) +
  plot_annotation(tag_levels = "a") & tag_theme()

save_ms(fig, file.path("..", "reports", "figures_r", "Fig1_wall"),
        width_mm = W_DOUBLE, height_mm = 196)

# Guards. These numbers are quoted in the Results text and in Table 1, so a
# silent change in the upstream CSV must break the build rather than produce a
# figure that disagrees with the prose.
chk <- function(f, col, want) {
  got <- fam_wall[[col]][fam_wall$family == f]
  stopifnot(length(got) == 1, abs(got - want) < 0.02)
}
chk("hand", "loss_cross", 0.939); chk("ast", "loss_cross", 0.842)
chk("mert", "loss_cross", 0.949); chk("w2v2", "loss_cross", 0.900)
chk("clap", "loss_cross", 0.573)
stopifnot(all(fam_wall$loss_cross - fam_wall$loss_same > 0.35))
message("  Fig1: family-panel agreement checks passed")
