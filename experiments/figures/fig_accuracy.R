# Figure D: small-area accuracy, from the paper tables' CSVs.
#
# Replaces Tables 2, 2t and 3. Methods are rows; each row carries one point per
# experiment (colour) and alpha (shape), dodged. Two rows of panels:
#   top     |z| p90 (log scale), the 90th percentile over cells of
#           |posterior mean - published| / SE;
#   bottom  the share of cells whose published value lies in the method's
#           posterior 90% interval.
# Columns of panels: fitted block-group cells (Table 2), fitted tract cells
# (Table 2t), and every held-out block-group cell together (Table 3's "All";
# the held-out tables one by one are a separate figure). Sampled cells only,
# as in the tables. HMC reference is left out (it matches HMC short to two
# decimals; their agreement is Figure E). Raking has no alpha and is drawn once.
#
# Reads table2_*.csv, table2t_*.csv and table3_*.csv from the paper tables'
# folder (experiments/paper_tables.py --out, e.g. <root>/tables_compare), and
# writes to <out>:
#   fig_accuracy.pdf, fig_accuracy.png   the figure
#   fig_accuracy_data.csv                the values plotted, so that the figure
#                                        can be redrawn without this repository
#
#   Rscript experiments/figures/fig_accuracy.R <tables folder> <out folder>
#
# Or, from a Quarto chunk, with the data file copied next to it:
#   source("fig_accuracy.R")
#   plot_accuracy(read.csv("fig_accuracy_data.csv"))

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

# Display labels, in the order the methods are drawn (top to bottom).
ACC_METHODS <- c(
  ipf = "IPF",
  ipf_multinomial = "IPF, multinomial",
  map_laplace = "MAP + Laplace",
  vb_gaussian = "VB Gaussian",
  vb_skewed = "VB skewed",
  hmc_short = "HMC short"
)
ACC_EXPERIMENTS <- c(
  exp01_baseline = "Baseline",
  exp02_nullspace = "Null space",
  exp03_rollup = "Roll-up"
)
# Categorical slots 1-3 of the reference palette, checked for colour-vision
# deficiency separation; shape carries alpha, so colour is not the only cue.
ACC_EXPERIMENT_COLOURS <- c(Baseline = "#2a78d6", `Null space` = "#eb6834", `Roll-up` = "#1baf7a")
ACC_ALPHA_SHAPES <- c(`alpha = 0.01` = 16, `alpha = 1` = 17, `raking (no alpha)` = 15)
ACC_ALPHA_LABELS <- c(expression(alpha == 0.01), expression(alpha == 1), "raking (no alpha)")

# (table, key) -> panel column and metric.
ACC_PANELS <- data.frame(
  table = c("2", "2", "2t", "2t", "3", "3"),
  key = c("post_cover", "absz_p90", "post_cover", "absz_p90", "all_cover", "all_absz_p90"),
  target = c("Fitted block group", "Fitted block group", "Fitted tract", "Fitted tract",
             "Held-out block group", "Held-out block group"),
  metric = rep(c("cover", "absz_p90"), 3)
)
ACC_TARGETS <- unique(ACC_PANELS$target)

accuracy_data <- function(tables_dir) {
  files <- list.files(tables_dir, pattern = "^table(2|2t|3)_.*\\.csv$", full.names = TRUE)
  if (length(files) != 3) {
    stop("expected table2_*.csv, table2t_*.csv and table3_*.csv in ", tables_dir)
  }
  raw <- do.call(rbind, lapply(files, read.csv, colClasses = c(table = "character")))
  data <- merge(raw, ACC_PANELS, by = c("table", "key"))
  data <- data[data$method %in% names(ACC_METHODS) & data$experiment %in% names(ACC_EXPERIMENTS), ]
  # Raking is fitted once and scored in both alpha cells: keep one copy.
  raking <- data$method %in% c("ipf", "ipf_multinomial")
  data <- data[!raking | abs(data$alpha - min(data$alpha)) < 1e-9, ]
  data$alpha_label <- ifelse(data$method %in% c("ipf", "ipf_multinomial"), "raking (no alpha)",
                             paste("alpha =", format(data$alpha, drop0trailing = TRUE)))
  data$experiment <- unname(ACC_EXPERIMENTS[data$experiment])
  data$method <- unname(ACC_METHODS[data$method])
  data <- data[!is.na(data$value), ]
  data[order(data$metric, data$target, data$method, data$experiment, data$alpha_label),
       c("metric", "target", "method", "experiment", "alpha_label", "value", "n")]
}

plot_accuracy <- function(data) {
  data$method <- factor(data$method, levels = rev(unname(ACC_METHODS)))  # first at the top
  data$experiment <- factor(data$experiment, levels = unname(ACC_EXPERIMENTS))
  data$alpha_label <- factor(data$alpha_label, levels = names(ACC_ALPHA_SHAPES))
  data$target <- factor(data$target, levels = ACC_TARGETS)
  data$group <- interaction(data$experiment, data$alpha_label)
  dodge <- position_dodge(width = 0.75)

  # Reference lines on the held-out panel only: there the published value is
  # independent of the fit, so sampling error alone gives |z| p90 = 1.645, and
  # 0.90 is a benchmark for coverage (below it is expected, as the interval is
  # for the true count). Fitted cells have no such benchmark.
  held_out <- factor(ACC_TARGETS[startsWith(ACC_TARGETS, "Held-out")], levels = ACC_TARGETS)
  panel <- function(part, reference) {
    ggplot(part, aes(x = method, y = value, colour = experiment, shape = alpha_label,
                     group = group)) +
      geom_hline(data = data.frame(target = held_out), aes(yintercept = reference),
                 linetype = "dashed", linewidth = 0.3, colour = "grey45") +
      geom_point(position = dodge, size = 1.6) +
      facet_wrap(~target, nrow = 1, drop = FALSE) +
      coord_flip() +
      scale_x_discrete(drop = FALSE) +  # the same rows in both panel rows
      scale_colour_manual(values = ACC_EXPERIMENT_COLOURS, name = "Experiment", drop = FALSE) +
      scale_shape_manual(values = ACC_ALPHA_SHAPES, labels = ACC_ALPHA_LABELS, name = NULL,
                         drop = FALSE) +
      labs(x = NULL) +
      theme_bw(base_size = 8) +
      theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
            panel.grid.major.x = element_line(colour = "grey90", linewidth = 0.3),
            strip.background = element_blank(), strip.text = element_text(face = "bold"),
            legend.position = "bottom")
  }

  top <- panel(data[data$metric == "absz_p90", ], 1.645) +
    scale_y_log10() +
    labs(y = "|z| p90 (log scale)")
  bottom <- panel(data[data$metric == "cover", ], 0.90) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
    labs(y = "Share of cells with the published value in the posterior 90% interval") +
    theme(strip.text = element_blank())
  top / bottom + plot_layout(guides = "collect") & theme(legend.position = "bottom")
}

main <- function(args) {
  if (length(args) != 2) {
    stop("usage: Rscript fig_accuracy.R <tables folder> <out folder>")
  }
  out <- args[2]
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  data <- accuracy_data(args[1])
  write.csv(data, file.path(out, "fig_accuracy_data.csv"), row.names = FALSE)
  figure <- plot_accuracy(data)
  # Elsevier's full text width is 190 mm.
  ggsave(file.path(out, "fig_accuracy.pdf"), figure, width = 190, height = 130, units = "mm",
         device = if (capabilities("cairo")) cairo_pdf else pdf)
  ggsave(file.path(out, "fig_accuracy.png"), figure, width = 190, height = 130, units = "mm",
         dpi = 300)
  message("wrote ", out)
}

if (sys.nframe() == 0) main(commandArgs(trailingOnly = TRUE))
