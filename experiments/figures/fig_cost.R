# Figure C: cost against quality, from the paper tables' CSVs.
#
# x: total minutes per PUMA (log scale), Table 1's "Total" (for HMC, the VB
# Gaussian fit that whitens it plus sampling). One column of panels per alpha;
# raking has no alpha and appears in both. One row per cell set, then one for
# the draws:
#   interval score, the mean over the set's sampled cells of each cell's 90%
#     predictive interval width plus 20 times how far the published value falls
#     outside it, in published SEs (Gneiting & Raftery 2007): the accuracy of
#     the whole distribution, centre and spread. Lower is better. Four sets:
#     fitted block group and fitted tract cells (Tables 2 and 2t; the published
#     values were used in the fit, so these describe the fit) and every held-out
#     block group and tract cell (Tables 3 and 3t, "All"; cells the fit never
#     saw). On the held-out rows the dashed line (about 4.1) is the expected
#     score of a method that knew every true count;
#   the share of draws with no single (record, block group) cell above 1% of N
#     (one minus Table 5's "Draws with a cell > 0.01 N"): how often a draw is a
#     plausible population.
# Colour is the method, shape the experiment, and an arrow runs from each
# method's Baseline point to its Null space and to its Roll-up point, to show
# what each remedy buys. IPF is shown by its multinomial draws only: the IPF fit
# itself has the same cost and the same estimates. MAP + Laplace's scores are
# far above the others and are left off the score rows, which would otherwise
# squeeze every other method into a strip; each score panel notes its range.
#
# Reads table1_*.csv, table2_*.csv, table2t_*.csv, table3_*.csv, table3t_*.csv
# and table5_*.csv from the paper tables' folder (experiments/paper_tables.py
# --out, e.g. <root>/tables_compare), and writes to <out>:
#   fig_cost.pdf, fig_cost.png   the figure
#   fig_cost_data.csv            the values plotted
#
#   Rscript experiments/figures/fig_cost.R <tables folder> <out folder>
#
# Or, from a Quarto chunk, with the CSVs in a folder "results":
#   source("results/fig_cost.R")
#   plot_cost(cost_data("results"))

suppressPackageStartupMessages({
  library(ggplot2)
})

# Display labels, in legend order.
COST_METHODS <- c(
  ipf_multinomial = "IPF, multinomial",
  map_laplace = "MAP + Laplace",
  vb_gaussian = "VB Gaussian",
  vb_skewed = "VB skewed",
  hmc_short = "HMC short",
  hmc_ref = "HMC reference"
)
COST_EXPERIMENTS <- c(
  exp01_baseline = "Baseline",
  exp02_nullspace = "Null space",
  exp03_rollup = "Roll-up"
)
# Categorical slots 1-6 of the reference palette, checked for colour-vision
# deficiency separation; shape (the experiment) and the legend are the other cues.
COST_METHOD_COLOURS <- setNames(c("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"),
                           unname(COST_METHODS))
COST_EXPERIMENT_SHAPES <- c(Baseline = 16, `Null space` = 17, `Roll-up` = 15)
COST_WALL <- 0.01  # Table 5's share of N that makes a cell a wall (for the label)
# The expected interval score of a method that knew every true count: the
# published value is then off by a standard normal number of SEs.
COST_ORACLE_SCORE <- 2 * 1.645 +
  2 / 0.10 * 2 * (dnorm(1.645) - 1.645 * pnorm(1.645, lower.tail = FALSE))

# The rows of panels: (table, key, label), the score rows first.
COST_ROWS <- data.frame(
  table = c("2", "2t", "3", "3t", "5"),
  key = c("is", "is", "all_is", "all_is", "walls"),
  row = c("Fitted block group", "Fitted tract", "Held-out block group", "Held-out tract",
          "Plausible draws")
)

cost_data <- function(tables_dir) {
  read <- function(number) {
    file <- list.files(tables_dir, pattern = paste0("^table", number, "_.*\\.csv$"),
                       full.names = TRUE)
    if (length(file) != 1) stop("expected one table", number, "_*.csv in ", tables_dir)
    read.csv(file, colClasses = c(table = "character"))
  }
  keys <- c("experiment", "alpha", "method")
  minutes <- read("1")
  minutes <- minutes[minutes$key == "total", c(keys, "value")]
  names(minutes)[4] <- "minutes"
  values <- do.call(rbind, lapply(seq_len(nrow(COST_ROWS)), function(i) {
    frame <- read(COST_ROWS$table[i])
    part <- frame[frame$key == COST_ROWS$key[i], c(keys, "value")]
    if (COST_ROWS$key[i] == "walls") part$value <- 1 - part$value  # the plausible share
    part$row <- COST_ROWS$row[i]
    part
  }))
  data <- merge(values, minutes, by = keys)
  data <- data[data$method %in% names(COST_METHODS) &
                 data$experiment %in% names(COST_EXPERIMENTS) & !is.na(data$value), ]
  data$alpha_label <- paste("alpha ==", format(data$alpha, drop0trailing = TRUE))
  data$experiment <- unname(COST_EXPERIMENTS[data$experiment])
  data$method <- unname(COST_METHODS[data$method])
  data <- data[order(match(data$row, COST_ROWS$row), data$alpha, data$method,
                     match(data$experiment, COST_EXPERIMENTS)), ]
  data[, c("row", "alpha_label", "method", "experiment", "minutes", "value")]
}

plot_cost <- function(data) {
  data$method <- factor(data$method, levels = unname(COST_METHODS))
  data$experiment <- factor(data$experiment, levels = unname(COST_EXPERIMENTS))
  data$row <- factor(data$row, levels = COST_ROWS$row)
  # MAP + Laplace off the score rows; each score panel says where it is.
  laplace <- data$method == "MAP + Laplace" & data$row != "Plausible draws"
  off_scale <- aggregate(value ~ row + alpha_label, data = data[laplace, ], FUN = range)
  off_scale$label <- sprintf("MAP + Laplace off scale (%.1f to %.1f)",
                             off_scale$value[, 1], off_scale$value[, 2])
  off_scale$value <- NULL
  shown <- data[!laplace, ]
  # One arrow per remedy, from the method's Baseline point.
  base <- shown[shown$experiment == "Baseline", ]
  remedies <- merge(shown[shown$experiment != "Baseline", ], base,
                    by = c("row", "alpha_label", "method"), suffixes = c("", "_base"))
  held_out <- data.frame(row = factor(c("Held-out block group", "Held-out tract"),
                                      levels = COST_ROWS$row))
  # The plausible share is a proportion: show all of 0 to 1.
  bounds <- data.frame(row = factor("Plausible draws", levels = COST_ROWS$row),
                       minutes = min(data$minutes), value = c(0, 1))
  row_labels <- c(`Fitted block group` = "Interval score / SE:\nfitted block group",
                  `Fitted tract` = "Interval score / SE:\nfitted tract",
                  `Held-out block group` = "Interval score / SE:\nheld-out block group",
                  `Held-out tract` = "Interval score / SE:\nheld-out tract",
                  `Plausible draws` = paste0("Plausible draws: share\nwith no cell > ",
                                             100 * COST_WALL, "% of N"))

  ggplot(shown, aes(x = minutes, y = value, colour = method)) +
    geom_hline(data = held_out, aes(yintercept = COST_ORACLE_SCORE), linetype = "dashed",
               linewidth = 0.3, colour = "grey45") +
    geom_blank(data = bounds, aes(x = minutes, y = value), inherit.aes = FALSE) +
    geom_segment(data = remedies,
                 aes(x = minutes_base, y = value_base, xend = minutes, yend = value),
                 linewidth = 0.4, alpha = 0.6, show.legend = FALSE,
                 arrow = arrow(length = unit(1.4, "mm"), type = "closed")) +
    geom_point(aes(shape = experiment), size = 1.8) +
    geom_text(data = off_scale, aes(label = label), x = -Inf, y = Inf, hjust = -0.05,
              vjust = 1.6, size = 2.2, colour = COST_METHOD_COLOURS[["MAP + Laplace"]],
              inherit.aes = FALSE) +
    facet_grid(row ~ alpha_label, scales = "free_y", switch = "y",
               labeller = labeller(alpha_label = label_parsed, row = row_labels)) +
    scale_x_log10(breaks = c(1, 10, 100, 1000), labels = c("1", "10", "100", "1,000")) +
    # Room above the points for the off-scale note; on the share row (the only
    # one reaching below 0.5), no ticks outside 0 to 1.
    scale_y_continuous(expand = expansion(mult = c(0.06, 0.22)), breaks = function(limits) {
      breaks <- pretty(limits)
      if (limits[1] < 0.5) breaks[breaks > -1e-9 & breaks < 1 + 1e-9] else breaks
    }) +
    scale_colour_manual(values = COST_METHOD_COLOURS, name = "Method", drop = FALSE) +
    scale_shape_manual(values = COST_EXPERIMENT_SHAPES, name = "Experiment", drop = FALSE) +
    labs(x = "Total minutes per PUMA (log scale)", y = NULL) +
    theme_bw(base_size = 8) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(colour = "grey90", linewidth = 0.3),
          strip.background = element_blank(), strip.text = element_text(face = "bold"),
          strip.placement = "outside", legend.position = "right")
}

main <- function(args) {
  if (length(args) != 2) {
    stop("usage: Rscript fig_cost.R <tables folder> <out folder>")
  }
  out <- args[2]
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  data <- cost_data(args[1])
  write.csv(data, file.path(out, "fig_cost_data.csv"), row.names = FALSE)
  figure <- plot_cost(data)
  # Elsevier's full text width is 190 mm.
  ggsave(file.path(out, "fig_cost.pdf"), figure, width = 190, height = 210, units = "mm",
         device = if (capabilities("cairo")) cairo_pdf else pdf)
  ggsave(file.path(out, "fig_cost.png"), figure, width = 190, height = 210, units = "mm",
         dpi = 300)
  message("wrote ", out)
}

if (sys.nframe() == 0) main(commandArgs(trailingOnly = TRUE))
