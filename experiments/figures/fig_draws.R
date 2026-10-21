# Figure B: usability of the draws, from paper_tables.py's draws_sample.csv.
#
# Replaces Table 5. For each draw (a whole synthetic population of one PUMA),
# each block group's effective number of household records,
# n_eff = 1 / sum s^2 over the records' shares s of its households (Kish's
# effective sample size; only block groups over 100 published households). Two
# rows of panels:
#   top     the draw's smallest n_eff over its block groups: its worst block
#           group;
#   bottom  the draw's median n_eff over its block groups: its typical one.
# Methods are rows (log scale across), one violin per experiment (colour),
# dodged, with each violin's median marked; one column of panels per alpha.
# Raking has no alpha and appears in both columns; the IPF fit is one
# population per PUMA, so it is drawn as its four points. The dashed line is
# Table 5's floor of 10. The exact posterior itself puts some block groups below
# it, so falling under it is not by itself a failure of the method.
#
# Reads draws_sample.csv from the paper tables' folder
# (experiments/paper_tables.py --out, e.g. <root>/tables_compare): up to 1,000
# random draws of each (experiment, method, alpha, PUMA). Writes to <out>:
#   fig_draws.pdf, fig_draws.png   the figure
#   fig_draws_data.csv             the values plotted
#
#   Rscript experiments/figures/fig_draws.R <tables folder> <out folder>
#
# Or, from a Quarto chunk, with the CSV in a folder "results":
#   source("results/fig_draws.R")
#   plot_draws(draws_data("results"))

suppressPackageStartupMessages({
  library(ggplot2)
})

# Display labels, in the order the methods are drawn (top to bottom).
DRAWS_METHODS <- c(
  ipf = "IPF",
  ipf_multinomial = "IPF, multinomial",
  map_laplace = "MAP + Laplace",
  vb_gaussian = "VB Gaussian",
  vb_skewed = "VB skewed",
  hmc_short = "HMC short",
  hmc_ref = "HMC reference"
)
DRAWS_EXPERIMENTS <- c(
  exp01_baseline = "Baseline",
  exp02_nullspace = "Null space",
  exp03_rollup = "Roll-up"
)
# As in Figure D: categorical slots 1-3 of the reference palette, checked for
# colour-vision deficiency separation.
DRAWS_EXPERIMENT_COLOURS <- c(Baseline = "#2a78d6", `Null space` = "#eb6834",
                              `Roll-up` = "#1baf7a")
DRAWS_RAKING <- c("ipf", "ipf_multinomial")
DRAWS_FLOOR <- 10  # Table 5's n_eff floor
DRAWS_METRICS <- c(min_bg_n_eff = "Smallest n_eff over block groups",
                   median_bg_n_eff = "Median n_eff over block groups")

draws_data <- function(tables_dir) {
  file <- file.path(tables_dir, "draws_sample.csv")
  if (!file.exists(file)) stop("no draws_sample.csv in ", tables_dir)
  raw <- read.csv(file)
  raw <- raw[raw$method %in% names(DRAWS_METHODS) &
               raw$experiment %in% names(DRAWS_EXPERIMENTS), ]
  # Raking is scored in each alpha cell from the same fit: one copy, shown in
  # every alpha column.
  alphas <- sort(unique(raw$alpha[!raw$method %in% DRAWS_RAKING]))
  raking <- raw$method %in% DRAWS_RAKING
  first <- raw[raking & abs(raw$alpha - min(raw$alpha[raking])) < 1e-9, ]
  raw <- rbind(raw[!raking, ],
               do.call(rbind, lapply(alphas, function(a) transform(first, alpha = a))))
  long <- do.call(rbind, lapply(names(DRAWS_METRICS), function(column) {
    data.frame(metric = DRAWS_METRICS[[column]], method = raw$method,
               experiment = raw$experiment, alpha = raw$alpha, puma = raw$puma,
               value = raw[[column]])
  }))
  long <- long[!is.na(long$value), ]
  long$alpha_label <- paste("alpha ==", format(long$alpha, drop0trailing = TRUE))
  long$method <- unname(DRAWS_METHODS[long$method])
  long$experiment <- unname(DRAWS_EXPERIMENTS[long$experiment])
  long[, c("metric", "alpha_label", "method", "experiment", "puma", "value")]
}

plot_draws <- function(data) {
  data$method <- factor(data$method, levels = rev(unname(DRAWS_METHODS)))  # first at the top
  data$experiment <- factor(data$experiment, levels = unname(DRAWS_EXPERIMENTS))
  data$metric <- factor(data$metric, levels = unname(DRAWS_METRICS))
  dodge <- position_dodge(width = 0.85)
  single <- data$method == "IPF"  # one population per PUMA: points, not a violin
  floor_line <- data.frame(metric = factor(DRAWS_METRICS[["min_bg_n_eff"]],
                                           levels = unname(DRAWS_METRICS)))

  ggplot(data, aes(x = method, y = value, fill = experiment)) +
    geom_hline(data = floor_line, aes(yintercept = DRAWS_FLOOR), linetype = "dashed",
               linewidth = 0.3, colour = "grey45") +
    geom_violin(data = data[!single, ], position = dodge, scale = "width", width = 0.8,
                linewidth = 0.2, colour = "grey30", alpha = 0.75) +
    stat_summary(data = data[!single, ], aes(group = experiment), fun = median,
                 geom = "point", position = dodge, shape = 21, size = 1.1, fill = "white",
                 colour = "black", stroke = 0.3) +
    geom_point(data = data[single, ], aes(colour = experiment), position = dodge, size = 1.1,
               show.legend = FALSE) +
    facet_grid(metric ~ alpha_label, labeller = labeller(alpha_label = label_parsed)) +
    coord_flip() +
    scale_x_discrete(drop = FALSE) +
    scale_y_log10(breaks = c(1, 2, 5, 10, 20, 50, 100, 200, 500)) +
    scale_fill_manual(values = DRAWS_EXPERIMENT_COLOURS, name = "Experiment", drop = FALSE) +
    scale_colour_manual(values = DRAWS_EXPERIMENT_COLOURS, guide = "none") +
    labs(x = NULL, y = "Effective number of household records in a block group (log scale)") +
    theme_bw(base_size = 8) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
          panel.grid.major.x = element_line(colour = "grey90", linewidth = 0.3),
          strip.background = element_blank(), strip.text = element_text(face = "bold"),
          legend.position = "bottom")
}

main <- function(args) {
  if (length(args) != 2) {
    stop("usage: Rscript fig_draws.R <tables folder> <out folder>")
  }
  out <- args[2]
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  data <- draws_data(args[1])
  write.csv(data, file.path(out, "fig_draws_data.csv"), row.names = FALSE)
  figure <- plot_draws(data)
  # Elsevier's full text width is 190 mm.
  ggsave(file.path(out, "fig_draws.pdf"), figure, width = 190, height = 150, units = "mm",
         device = if (capabilities("cairo")) cairo_pdf else pdf)
  ggsave(file.path(out, "fig_draws.png"), figure, width = 190, height = 150, units = "mm",
         dpi = 300)
  message("wrote ", out)
}

if (sys.nframe() == 0) main(commandArgs(trailingOnly = TRUE))
