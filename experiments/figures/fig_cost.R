# Figure C: cost against quality, from the paper tables' CSVs.
#
# x: total minutes per PUMA (log scale), Table 1's "Total" (for HMC, the VB
# Gaussian fit that whitens it plus sampling). One column of panels per alpha;
# raking has no alpha and appears in both. Two rows:
#   top     held-out |z| p90 (log scale), over every held-out block-group cell
#           (Table 3's "All"): the accuracy of the estimates;
#   bottom  the share of draws with no single (record, block group) cell above
#           1% of N (one minus Table 5's "Draws with a cell > 0.01 N"): how
#           often a draw is a plausible population.
# Colour is the method, shape the experiment, and an arrow runs from each
# method's Baseline point to its Null space and to its Roll-up point, to show
# what each remedy buys. IPF is shown by its multinomial draws only: the IPF fit
# itself has the same cost and the same estimates. MAP + Laplace's held-out
# |z| p90 (about 5 to 12) is left off the top row, which would otherwise squeeze
# every other method into a strip; each panel notes its range.
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
METHODS <- c(
  ipf_multinomial = "IPF, multinomial",
  map_laplace = "MAP + Laplace",
  vb_gaussian = "VB Gaussian",
  vb_skewed = "VB skewed",
  hmc_short = "HMC short",
  hmc_ref = "HMC reference"
)
EXPERIMENTS <- c(
  exp01_baseline = "Baseline",
  exp02_nullspace = "Null space",
  exp03_rollup = "Roll-up"
)
# Categorical slots 1-6 of the reference palette, checked for colour-vision
# deficiency separation; shape (the experiment) and the legend are the other cues.
METHOD_COLOURS <- setNames(c("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"),
                           unname(METHODS))
EXPERIMENT_SHAPES <- c(Baseline = 16, `Null space` = 17, `Roll-up` = 15)
WALL <- 0.01  # Table 5's share of N that makes a cell a wall

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
                      pick(read("3"), "all_absz_p90", "heldout_absz_p90"),
                      pick(read("5"), "walls", "walls")))
  data <- data[data$method %in% names(METHODS) & data$experiment %in% names(EXPERIMENTS), ]
  data$plausible <- 1 - data$walls
  data$alpha_label <- paste("alpha ==", format(data$alpha, drop0trailing = TRUE))
  data$experiment <- unname(EXPERIMENTS[data$experiment])
  data$method <- unname(METHODS[data$method])
  data <- data[order(data$alpha, data$method, match(data$experiment, EXPERIMENTS)), ]
  data[, c("alpha_label", "method", "experiment", "minutes", "heldout_absz_p90", "plausible")]
}

plot_cost <- function(data) {
  data$method <- factor(data$method, levels = unname(METHODS))
  data$experiment <- factor(data$experiment, levels = unname(EXPERIMENTS))
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
      scale_colour_manual(values = METHOD_COLOURS, name = "Method", drop = FALSE) +
      scale_shape_manual(values = EXPERIMENT_SHAPES, name = "Experiment", drop = FALSE) +
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
                               min(d$heldout_absz_p90), max(d$heldout_absz_p90)))))
  top <- panel("heldout_absz_p90", data[!laplace, ],
               remedies[remedies$method != "MAP + Laplace", ]) +
    geom_text(data = off_scale, aes(label = label), x = -Inf, y = -Inf, hjust = -0.05,
              vjust = -0.8, size = 2.3, colour = METHOD_COLOURS[["MAP + Laplace"]],
              inherit.aes = FALSE) +
    # sampling error alone would give |z| p90 = 1.645 on held-out cells
    geom_hline(yintercept = 1.645, linetype = "dashed", linewidth = 0.3, colour = "grey45") +
    scale_y_log10() +
    labs(x = NULL, y = "Held-out |z| p90 (log scale)")
  bottom <- panel("plausible") +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
    labs(x = "Total minutes per PUMA (log scale)",
         y = paste0("Share of draws with no cell above ", 100 * WALL, "% of N")) +
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
