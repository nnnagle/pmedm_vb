# Figure C: cost against quality, from the paper tables' CSVs.
#
# x: total minutes per PUMA (log scale), Table 1's "Total" (for HMC, the VB
# Gaussian fit that whitens it plus sampling). One column of panels per alpha;
# raking has no alpha and appears in both. Two rows:
#   top     held-out interval score, the mean over every held-out
#           block-group cell of its 90% predictive interval's width plus 20
#           times how far the published value falls outside it, in published
#           SEs (Table 3's "All: interval score"; Gneiting & Raftery 2007): the
#           accuracy of the whole distribution, centre and spread, on cells the
#           fit never saw. Lower is better; the dashed line (about 4.1) is the
#           expected score of a method that knew every true count;
#   bottom  the share of draws with no single (record, block group) cell above
#           1% of N (one minus Table 5's "Draws with a cell > 0.01 N"): how
#           often a draw is a plausible population.
# Colour is the method, shape the experiment, and an arrow runs from each
# method's Baseline point to its Null space and to its Roll-up point, to show
# what each remedy buys. IPF is shown by its multinomial draws only: the IPF fit
# itself has the same cost and the same estimates. MAP + Laplace's held-out
# score is far above the others and is left off the top row, which would
# otherwise squeeze every other method into a strip; each panel notes its range.
#
# Reads table1_*.csv, table3_*.csv and table5_*.csv from the paper tables'
# folder (experiments/paper_tables.py --out, e.g. <root>/tables_compare), and
# writes to <out>:
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
  library(patchwork)
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
COST_WALL <- 0.01  # Table 5's share of N that makes a cell a wall
# The expected interval score of a method that knew every true count: the
# published value is then off by a standard normal number of SEs.
COST_ORACLE_SCORE <- 2 * 1.645 +
  2 / 0.10 * 2 * (dnorm(1.645) - 1.645 * pnorm(1.645, lower.tail = FALSE))

cost_data <- function(tables_dir) {
  read <- function(number) {
    file <- list.files(tables_dir, pattern = paste0("^table", number, "_.*\\.csv$"),
                       full.names = TRUE)
    if (length(file) != 1) stop("expected one table", number, "_*.csv in ", tables_dir)
    read.csv(file, colClasses = c(table = "character"))
  }
  pick <- function(frame, key, column) {
    part <- frame[frame$key == key, c("experiment", "alpha", "method", "value")]
    names(part)[4] <- column
    part
  }
  data <- Reduce(function(a, b) merge(a, b, by = c("experiment", "alpha", "method")),
                 list(pick(read("1"), "total", "minutes"),
                      pick(read("3"), "all_is", "heldout_score"),
                      pick(read("5"), "walls", "walls")))
  data <- data[data$method %in% names(COST_METHODS) &
                 data$experiment %in% names(COST_EXPERIMENTS), ]
  data$plausible <- 1 - data$walls
  data$alpha_label <- paste("alpha ==", format(data$alpha, drop0trailing = TRUE))
  data$experiment <- unname(COST_EXPERIMENTS[data$experiment])
  data$method <- unname(COST_METHODS[data$method])
  data <- data[order(data$alpha, data$method, match(data$experiment, COST_EXPERIMENTS)), ]
  data[, c("alpha_label", "method", "experiment", "minutes", "heldout_score", "plausible")]
}

plot_cost <- function(data) {
  data$method <- factor(data$method, levels = unname(COST_METHODS))
  data$experiment <- factor(data$experiment, levels = unname(COST_EXPERIMENTS))
  # One arrow per remedy, from the method's Baseline point.
  base <- data[data$experiment == "Baseline", ]
  remedies <- merge(data[data$experiment != "Baseline", ], base,
                    by = c("alpha_label", "method"), suffixes = c("", "_base"))

  panel <- function(y, part = data, arrows = remedies) {
    ggplot(part, aes(x = minutes, y = .data[[y]], colour = method)) +
      geom_segment(data = arrows,
                   aes(x = minutes_base, y = .data[[paste0(y, "_base")]],
                       xend = minutes, yend = .data[[y]]),
                   linewidth = 0.4, alpha = 0.6, show.legend = FALSE,
                   arrow = arrow(length = unit(1.4, "mm"), type = "closed")) +
      geom_point(aes(shape = experiment), size = 1.8) +
      facet_wrap(~alpha_label, nrow = 1, labeller = label_parsed) +
      scale_x_log10(limits = range(data$minutes), breaks = c(1, 10, 100, 1000),
                    labels = c("1", "10", "100", "1,000")) +  # the same in both rows
      scale_colour_manual(values = COST_METHOD_COLOURS, name = "Method", drop = FALSE) +
      scale_shape_manual(values = COST_EXPERIMENT_SHAPES, name = "Experiment", drop = FALSE) +
      theme_bw(base_size = 8) +
      theme(panel.grid.minor = element_blank(),
            panel.grid.major = element_line(colour = "grey90", linewidth = 0.3),
            strip.background = element_blank(), strip.text = element_text(face = "bold"),
            legend.position = "right")
  }

  laplace <- data$method == "MAP + Laplace"
  off_scale <- do.call(rbind, lapply(split(data[laplace, ], data$alpha_label[laplace]), function(d)
    data.frame(alpha_label = d$alpha_label[1],
               label = sprintf("MAP + Laplace off scale (%.1f to %.1f)",
                               min(d$heldout_score), max(d$heldout_score)))))
  top <- panel("heldout_score", data[!laplace, ],
               remedies[remedies$method != "MAP + Laplace", ]) +
    geom_text(data = off_scale, aes(label = label), x = -Inf, y = Inf, hjust = -0.05,
              vjust = 1.6, size = 2.3, colour = COST_METHOD_COLOURS[["MAP + Laplace"]],
              inherit.aes = FALSE) +
    geom_hline(yintercept = COST_ORACLE_SCORE, linetype = "dashed", linewidth = 0.3,
               colour = "grey45") +
    # The method legend comes from the bottom row, which has every method; this
    # row lacks MAP + Laplace, so its own would differ and be drawn twice.
    guides(colour = "none") +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +  # room for the note
    labs(x = NULL, y = "Held-out interval score / SE")
  bottom <- panel("plausible") +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
    labs(x = "Total minutes per PUMA (log scale)",
         y = paste0("Share of draws with no cell above ", 100 * COST_WALL, "% of N")) +
    theme(strip.text = element_blank())
  top / bottom + plot_layout(guides = "collect")
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
  ggsave(file.path(out, "fig_cost.pdf"), figure, width = 190, height = 120, units = "mm",
         device = if (capabilities("cairo")) cairo_pdf else pdf)
  ggsave(file.path(out, "fig_cost.png"), figure, width = 190, height = 120, units = "mm",
         dpi = 300)
  message("wrote ", out)
}

if (sys.nframe() == 0) main(commandArgs(trailingOnly = TRUE))
