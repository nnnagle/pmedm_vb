# Figure E: agreement with the reference HMC, from the paper tables' CSVs.
#
# Replaces Tables 4 and 4t. Each method against the HMC reference at the same
# alpha, over the fitted (constraint) leaf cells of the four PUMAs, where the
# reference is exactly the target the methods approximate. Two columns of
# panels:
#   median difference  |method's median - reference median| in reference sds,
#                      over cells: is the method centred where the posterior is?
#                      0 is perfect;
#   width ratio        the method's 90% interval width over the reference's, over
#                      cells: is its spread right? 1 is right (dashed), below 1
#                      overconfident.
# Filled points are the median over cells; open points, joined to them, the
# tail: the 99th percentile of the median difference (the worst-centred cells)
# and the 10th of the width ratio (the most overconfident). One row of panels
# per level (block group, tract) and alpha; methods are rows within a panel,
# one point per experiment (colour). IPF multinomial has no alpha and is
# compared with each alpha's reference; MAP + Laplace is left out (Tables 4 and
# 4t have it), as in Figure C.
#
# Reads table4_*.csv and table4t_*.csv from the paper tables' folder
# (experiments/paper_tables.py --out, e.g. <root>/tables_compare). Writes to
# <out>:
#   fig_agreement.pdf, fig_agreement.png   the figure
#   fig_agreement_data.csv                 the values plotted
#
#   Rscript experiments/figures/fig_agreement.R <tables folder> <out folder>
#
# Or, from a Quarto chunk, with the CSVs in a folder "results":
#   source("results/fig_agreement.R")
#   plot_agreement(agreement_data("results"))

suppressPackageStartupMessages({
  library(ggplot2)
})

# Display labels, in the order the methods are drawn (top to bottom).
AGR_METHODS <- c(
  ipf_multinomial = "IPF, multinomial",
  vb_gaussian = "VB Gaussian",
  vb_skewed = "VB skewed",
  vb_gaussian_trunc = "VB Gaussian, truncated",
  hmc_short = "HMC short"
)
AGR_EXPERIMENTS <- c(
  exp01_baseline = "Baseline",
  exp02_nullspace = "Null space",
  exp03_rollup = "Roll-up"
)
# As in Figures B and D: categorical slots 1-3 of the reference palette.
AGR_EXPERIMENT_COLOURS <- c(Baseline = "#2a78d6", `Null space` = "#eb6834",
                            `Roll-up` = "#1baf7a")
# (table, level label)
AGR_LEVELS <- c(`4` = "Block group", `4t` = "Tract")
# (measure, its median key, its tail key, label)
AGR_MEASURES <- data.frame(
  measure = c("mdiff", "width"),
  centre = c("mdiff_p50", "width_p50"),
  tail = c("mdiff_p99", "width_p10"),
  label = c("Median difference (reference sds)", "Width ratio (90% interval)")
)
AGR_STATS <- c(centre = "Median over cells", tail = "Tail: p99 (difference), p10 (width)")

agreement_data <- function(tables_dir) {
  data <- do.call(rbind, lapply(names(AGR_LEVELS), function(table) {
    file <- list.files(tables_dir, pattern = paste0("^table", table, "_.*\\.csv$"),
                       full.names = TRUE)
    if (length(file) != 1) stop("expected one table", table, "_*.csv in ", tables_dir)
    frame <- read.csv(file, colClasses = c(table = "character"))
    frame$level <- AGR_LEVELS[[table]]
    frame
  }))
  data <- data[data$method %in% names(AGR_METHODS) & data$experiment %in% names(AGR_EXPERIMENTS) &
                 !is.na(data$value), ]
  long <- do.call(rbind, lapply(seq_len(nrow(AGR_MEASURES)), function(i) {
    m <- AGR_MEASURES[i, ]
    do.call(rbind, lapply(c("centre", "tail"), function(stat) {
      part <- data[data$key == m[[stat]], c("level", "alpha", "method", "experiment", "value")]
      if (nrow(part)) transform(part, measure = m$label, stat = AGR_STATS[[stat]]) else NULL
    }))
  }))
  long$panel <- paste0(gsub(" ", "~", long$level), "*','~~alpha == ",
                       format(long$alpha, drop0trailing = TRUE))
  long$method <- unname(AGR_METHODS[long$method])
  long$experiment <- unname(AGR_EXPERIMENTS[long$experiment])
  long[, c("panel", "level", "alpha", "measure", "method", "experiment", "stat", "value")]
}

plot_agreement <- function(data) {
  # Methods top to bottom, the experiments side by side within each.
  data$y <- match(data$method, rev(unname(AGR_METHODS))) +
    0.25 * (2 - match(data$experiment, unname(AGR_EXPERIMENTS)))
  data$experiment <- factor(data$experiment, levels = unname(AGR_EXPERIMENTS))
  data$stat <- factor(data$stat, levels = unname(AGR_STATS))
  data$measure <- factor(data$measure, levels = AGR_MEASURES$label)
  panels <- unique(data[order(data$level, data$alpha), "panel"])
  data$panel <- factor(data$panel, levels = panels)
  pairs <- merge(data[data$stat == AGR_STATS[["centre"]], ],
                 data[data$stat == AGR_STATS[["tail"]], c("panel", "measure", "method",
                                                         "experiment", "value")],
                 by = c("panel", "measure", "method", "experiment"), suffixes = c("", "_tail"))
  ones <- data.frame(measure = factor(AGR_MEASURES$label[2], levels = AGR_MEASURES$label))

  ggplot(data, aes(x = value, y = y, colour = experiment)) +
    geom_vline(data = ones, aes(xintercept = 1), linetype = "dashed", linewidth = 0.3,
               colour = "grey45") +
    geom_segment(data = pairs, aes(x = value, xend = value_tail, y = y, yend = y),
                 linewidth = 0.35, alpha = 0.7, show.legend = FALSE) +
    geom_point(aes(shape = stat), size = 1.5, fill = "white", stroke = 0.6) +
    facet_grid(panel ~ measure, scales = "free_x", labeller = labeller(panel = label_parsed)) +
    scale_x_log10() +
    scale_y_continuous(breaks = seq_along(AGR_METHODS), labels = rev(unname(AGR_METHODS)),
                       expand = expansion(add = 0.5)) +
    scale_colour_manual(values = AGR_EXPERIMENT_COLOURS, name = "Experiment", drop = FALSE) +
    scale_shape_manual(values = c(16, 21), name = NULL, drop = FALSE) +
    labs(x = "Against the HMC reference (log scale)", y = NULL) +
    theme_bw(base_size = 8) +
    theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_line(colour = "grey90", linewidth = 0.3),
          strip.background = element_blank(), strip.text = element_text(face = "bold"),
          legend.position = "bottom", legend.box = "vertical")
}

main <- function(args) {
  if (length(args) != 2) {
    stop("usage: Rscript fig_agreement.R <tables folder> <out folder>")
  }
  out <- args[2]
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  data <- agreement_data(args[1])
  write.csv(data, file.path(out, "fig_agreement_data.csv"), row.names = FALSE)
  figure <- plot_agreement(data)
  # Elsevier's full text width is 190 mm.
  ggsave(file.path(out, "fig_agreement.pdf"), figure, width = 190, height = 170, units = "mm",
         device = if (capabilities("cairo")) cairo_pdf else pdf)
  ggsave(file.path(out, "fig_agreement.png"), figure, width = 190, height = 170, units = "mm",
         dpi = 300)
  message("wrote ", out)
}

if (sys.nframe() == 0) main(commandArgs(trailingOnly = TRUE))
