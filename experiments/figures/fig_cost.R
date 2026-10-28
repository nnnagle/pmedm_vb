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
#     block group and tract cell (Tables 3 and 3t; cells the fit never
#     saw). On the held-out rows the dashed line (about 4.1) is the expected
#     score of a method that knew every true count;
#   the share of usable draws: draws in which every block group (over 100
#     published households) has D/H of at least Table 5's floor, 0.04 (one minus
#     Table 5's "Draws with D/H < 0.04"; D/H as in Figure B). The floor is below
#     the smallest D/H any HMC reference draw reached (0.054): a draw under it is
#     unlike anything the posterior was seen to make. The row's axis starts at
#     COST_SHARE_MIN, so that the differences near 1 show.
# Colour is the method, shape the experiment, and a line joins each method's
# Baseline point to its Null space and to its Roll-up point, to show what each
# remedy buys. IPF is shown by its multinomial draws only: the IPF fit
# itself has the same cost and the same estimates. MAP + Laplace is left out:
# its scores are many times the others' and its draws are never plausible
# (Tables 2-5 have it), and on these axes it would squeeze every other method
# into a strip.
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
  vb_gaussian = "VB Gaussian",
  vb_skewed = "VB skewed",
  vb_gaussian_trunc = "VB Gaussian, truncated",
  hmc_short = "HMC short",
  hmc_ref = "HMC reference"
)
COST_EXPERIMENTS <- c(
  exp01_baseline = "Baseline",
  exp02_nullspace = "Null space",
  exp03_rollup = "Roll-up"
)
# Categorical slots of the reference palette (2, orange, was MAP + Laplace's),
# checked for colour-vision deficiency separation; shape (the experiment) and the
# legend are the other cues.
COST_METHOD_COLOURS <- c(`IPF, multinomial` = "#2a78d6",
                         `VB Gaussian` = "#1baf7a", `VB skewed` = "#eda100",
                         `HMC short` = "#e87ba4", `HMC reference` = "#008300",
                         `VB Gaussian, truncated` = "#6250d6")
COST_EXPERIMENT_SHAPES <- c(Baseline = 16, `Null space` = 17, `Roll-up` = 15)
# Table 5's D/H floor (paper_tables.py --dh-floor), for the label; cost_data
# checks it against the CSV's.
COST_DH_FLOOR <- 0.04
# The bottom of the usable-draws row's axis (a share below it still shows: the
# axis then reaches down to it).
COST_SHARE_MIN <- 0.9
# The expected interval score of a method that knew every true count: the
# published value is then off by a standard normal number of SEs.
COST_ORACLE_SCORE <- 2 * 1.645 +
  2 / 0.10 * 2 * (dnorm(1.645) - 1.645 * pnorm(1.645, lower.tail = FALSE))

# The rows of panels: (table, key, label), the score rows first.
COST_ROWS <- data.frame(
  table = c("2", "2t", "3", "3t", "5"),
  key = c("is", "is", "all_is", "all_is", "dh_lt"),
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
    if (COST_ROWS$key[i] == "dh_lt") {
      label <- unique(frame$label[frame$key == "dh_lt"])
      if (!any(grepl(paste0("< ", format(COST_DH_FLOOR), "$"), label))) {
        stop("Table 5's D/H floor is not ", COST_DH_FLOOR, ": ", paste(label, collapse = "; "))
      }
    }
    part <- frame[frame$key == COST_ROWS$key[i], c(keys, "value")]
    if (COST_ROWS$key[i] == "dh_lt") part$value <- 1 - part$value  # the plausible share
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
  shown <- data
  # One line per remedy, from the method's Baseline point.
  base <- shown[shown$experiment == "Baseline", ]
  remedies <- merge(shown[shown$experiment != "Baseline", ], base,
                    by = c("row", "alpha_label", "method"), suffixes = c("", "_base"))
  held_out <- data.frame(row = factor(c("Held-out block group", "Held-out tract"),
                                      levels = COST_ROWS$row))
  # The usable share is a proportion: show COST_SHARE_MIN to 1.
  bounds <- data.frame(row = factor("Plausible draws", levels = COST_ROWS$row),
                       minutes = min(data$minutes), value = c(COST_SHARE_MIN, 1))
  # Short row labels, so that they fit a panel's height; the y-axis title says
  # what each row measures.
  row_labels <- c(`Fitted block group` = "Fitted\nblock group", `Fitted tract` = "Fitted\ntract",
                  `Held-out block group` = "Held-out\nblock group",
                  `Held-out tract` = "Held-out\ntract", `Plausible draws` = "Plausible\ndraws")

  ggplot(shown, aes(x = minutes, y = value, colour = method)) +
    geom_hline(data = held_out, aes(yintercept = COST_ORACLE_SCORE), linetype = "dashed",
               linewidth = 0.3, colour = "grey45") +
    geom_blank(data = bounds, aes(x = minutes, y = value), inherit.aes = FALSE) +
    geom_segment(data = remedies,
                 aes(x = minutes_base, y = value_base, xend = minutes, yend = value),
                 linewidth = 0.4, alpha = 0.6, show.legend = FALSE) +
    geom_point(aes(shape = experiment), size = 1.8) +
    facet_grid(row ~ alpha_label, scales = "free_y", switch = "y",
               labeller = labeller(alpha_label = label_parsed, row = row_labels)) +
    scale_x_log10(breaks = c(1, 10, 100, 1000), labels = c("1", "10", "100", "1,000")) +
    # On the share row (the only one ending near 1; the scores are over 3), no
    # ticks above 1.
    scale_y_continuous(expand = expansion(mult = 0.06), breaks = function(limits) {
      breaks <- pretty(limits)
      if (limits[2] < 2) breaks[breaks > -1e-9 & breaks < 1 + 1e-9] else breaks
    }) +
    scale_colour_manual(values = COST_METHOD_COLOURS, name = "Method", drop = FALSE) +
    scale_shape_manual(values = COST_EXPERIMENT_SHAPES, name = "Experiment", drop = FALSE) +
    labs(x = "Total minutes per PUMA (log scale)",
         y = paste0("Interval score / SE, lower is better (rows 1-4)\n",
                    "Share of draws with every block group's D/H \u2265 ", COST_DH_FLOOR,
                    " (row 5)")) +
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
