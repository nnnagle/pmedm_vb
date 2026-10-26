# Figure B: usability of the draws, from paper_tables.py's draws_sample.csv.
#
# Replaces Table 5. For each draw (a whole synthetic population of one PUMA),
# each block group's distinct household records per household, D/H: D is the
# expected number of distinct records among its H published occupied
# households drawn by the records' shares s, D = sum 1 - (1 - s)^H (only block
# groups over 100 published households). 1 is every household a different
# record; small is a few records copied many times. Two rows of panels:
#   top     the draw's smallest D/H over its block groups: its worst block
#           group;
#   bottom  the draw's median D/H over its block groups: its typical one.
# Methods are rows (log scale across), one violin per experiment (colour),
# dodged, with each violin's median marked; one column of panels per alpha.
# Raking has no alpha and appears in both columns; the IPF fit is one
# population per PUMA, so it is drawn as its four points. The dashed line is
# Table 5's floor of 0.04, below the smallest D/H any HMC draw reached in the
# paper_v3 runs (0.054): below it, a draw is unlike anything the posterior was
# seen to make. (draws_sample.csv also has n_eff.)
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

# Display labels, in the order the methods are drawn (top to bottom); a method
# left out here is left out of the figure (MAP + Laplace and VB skewed are in
# Table 5).
DRAWS_METHODS <- c(
  ipf = "IPF",
  ipf_multinomial = "IPF, multinomial",
  vb_gaussian = "VB Gaussian",
  vb_gaussian_trunc = "VB Gaussian, truncated",
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
DRAWS_FLOOR <- 0.04  # Table 5's D/H floor (paper_tables.py --dh-floor)
DRAWS_METRICS <- c(min_bg_distinct_per_hh = "Smallest D/H over block groups",
                   median_bg_distinct_per_hh = "Median D/H over block groups")

draws_data <- function(tables_dir) {
  file <- file.path(tables_dir, "draws_sample.csv")
  if (!file.exists(file)) stop("no draws_sample.csv in ", tables_dir)
  raw <- read.csv(file)
  missing <- setdiff(names(DRAWS_METRICS), names(raw))
  if (length(missing)) stop(file, " has no ", paste(missing, collapse = ", "),
                            ": rerun paper_tables.py")
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
  floor_line <- data.frame(metric = factor(DRAWS_METRICS[["min_bg_distinct_per_hh"]],
                                           levels = unname(DRAWS_METRICS)))

  # Bandwidth: Sheather & Jones (1991), which follows clusters such as IPF's
  # one tight mode per PUMA that the default (Silverman's rule, set by the whole
  # spread) blurs together. Groups mostly at one value (to 3 significant
  # digits), such as MAP + Laplace's draws of one record per block group, keep the
  # default: there SJ fails, or collapses to a spike.
  key <- interaction(data$metric, data$alpha_label, data$method, data$experiment, drop = TRUE)
  sj <- vapply(split(log10(data$value), key), function(v) {
    max(table(signif(v, 3))) <= length(v) / 2 &&
      !inherits(try(bw.SJ(v), silent = TRUE), "try-error")
  }, logical(1))
  by_sj <- !single & sj[as.character(key)]
  violin <- function(part, bw) {
    geom_violin(data = part, position = dodge, scale = "width", width = 0.8, bw = bw,
                linewidth = 0.2, colour = "grey30", alpha = 0.75)
  }

  ggplot(data, aes(x = method, y = value, fill = experiment)) +
    geom_hline(data = floor_line, aes(yintercept = DRAWS_FLOOR), linetype = "dashed",
               linewidth = 0.3, colour = "grey45") +
    violin(data[by_sj, ], "SJ") +
    violin(data[!single & !by_sj, ], "nrd0") +
    stat_summary(data = data[!single, ], aes(group = experiment), fun = median,
                 geom = "point", position = dodge, shape = 21, size = 1.1, fill = "white",
                 colour = "black", stroke = 0.3) +
    geom_point(data = data[single, ], aes(colour = experiment), position = dodge, size = 1.1,
               show.legend = FALSE) +
    facet_grid(metric ~ alpha_label, labeller = labeller(alpha_label = label_parsed)) +
    coord_flip() +
    scale_x_discrete(drop = FALSE) +
    scale_y_log10(breaks = c(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1),
                  labels = c("0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1")) +
    scale_fill_manual(values = DRAWS_EXPERIMENT_COLOURS, name = "Experiment", drop = FALSE) +
    scale_colour_manual(values = DRAWS_EXPERIMENT_COLOURS, guide = "none") +
    labs(x = NULL, y = "Distinct household records per household in a block group, D/H (log scale)") +
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
