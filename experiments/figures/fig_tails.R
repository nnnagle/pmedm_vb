# Figure: the HMC reference draws' errors against the model's Sigma, from
# draw_sigma.py's draw_sigma_hist.csv.
#
# Each draw's error on the model's rows, e = N X'p - Y, in the model's own
# standard deviations, z = e / sqrt(diag Sigma) (the published variance plus the
# multiplier prior's tau^2). Pooled over the draws, the rows and the PUMAs of
# each experiment, the share of |z| beyond x, against x, both on log scales:
# where the bulk of the posterior lies (left) and how far its tails reach
# (right). The dashed black curve is a standard normal's, 2 Phi(-x): what the
# errors would follow if they matched Sigma's diagonal. One panel per level of
# the model's rows (tract, block group, the PUMA rows); solid lines for rows
# with a nonzero published value, dashed for published zeros; colour for the
# experiment. Each curve stops at the smallest share its values can show,
# about one over their number.
#
# Reads draw_sigma_hist.csv (experiments/draw_sigma.py, written beside its
# --out). Writes to <out>:
#   fig_tails.pdf, fig_tails.png   the figure
#   fig_tails_data.csv             the values plotted
#
#   Rscript experiments/figures/fig_tails.R <folder with draw_sigma_hist.csv> <out folder>
#
# Or, from a Quarto chunk, with the CSV in a folder "results":
#   source("results/fig_tails.R")
#   plot_tails(tails_data("results"))

suppressPackageStartupMessages({
  library(ggplot2)
})

TAILS_EXPERIMENTS <- c(
  exp01_baseline = "Baseline",
  exp02_nullspace = "Null space",
  exp03_rollup = "Roll-up"
)
# As in Figures B, D and E: categorical slots 1-3 of the reference palette.
TAILS_EXPERIMENT_COLOURS <- c(Baseline = "#2a78d6", `Null space` = "#eb6834",
                              `Roll-up` = "#1baf7a")
TAILS_LEVELS <- c("block group" = "Block group rows", tract = "Tract rows", PUMA = "PUMA rows")
TAILS_PUBLISHED <- c(nonzero = "Published nonzero", zero = "Published zero")
TAILS_SCALING <- "model"  # "published": against the published variances alone
TAILS_ALPHA <- 0.01
TAILS_X <- c(0.1, 1000)   # the x range drawn

tails_data <- function(dir) {
  file <- file.path(dir, "draw_sigma_hist.csv")
  if (!file.exists(file)) stop("no draw_sigma_hist.csv in ", dir)
  raw <- read.csv(file)
  raw <- raw[raw$scaling == TAILS_SCALING & abs(raw$alpha - TAILS_ALPHA) < 1e-9 &
               raw$experiment %in% names(TAILS_EXPERIMENTS), ]
  if (!nrow(raw)) stop("no rows for scaling ", TAILS_SCALING, " at alpha ", TAILS_ALPHA)
  # Pool the PUMAs, then the share beyond each bin's lower edge.
  pooled <- aggregate(count ~ experiment + level + published + bin + lo, data = raw, FUN = sum)
  pooled <- pooled[order(pooled$experiment, pooled$level, pooled$published, -pooled$bin), ]
  group <- interaction(pooled$experiment, pooled$level, pooled$published, drop = TRUE)
  pooled$beyond <- ave(pooled$count, group, FUN = cumsum)
  pooled$values <- ave(pooled$count, group, FUN = sum)
  pooled$share <- pooled$beyond / pooled$values
  pooled <- pooled[pooled$lo > 0 & pooled$share > 0, ]
  pooled$experiment <- unname(TAILS_EXPERIMENTS[pooled$experiment])
  pooled$level <- unname(TAILS_LEVELS[pooled$level])
  pooled$published <- unname(TAILS_PUBLISHED[pooled$published])
  pooled <- pooled[order(pooled$experiment, pooled$level, pooled$published, pooled$lo), ]
  data.frame(experiment = pooled$experiment, level = pooled$level, published = pooled$published,
             x = pooled$lo, share = pooled$share, values = pooled$values)
}

plot_tails <- function(data) {
  data$experiment <- factor(data$experiment, levels = unname(TAILS_EXPERIMENTS))
  data$level <- factor(data$level, levels = unname(TAILS_LEVELS))
  data$published <- factor(data$published, levels = unname(TAILS_PUBLISHED))
  data <- data[data$x >= TAILS_X[1] & data$x <= TAILS_X[2], ]
  x <- 10^seq(log10(TAILS_X[1]), log10(TAILS_X[2]), length.out = 400)
  floor_share <- 0.5 / max(data$values)
  normal <- data.frame(x = x, share = 2 * pnorm(-x))
  normal <- normal[normal$share >= floor_share, ]

  ggplot(data, aes(x = x, y = share, colour = experiment, linetype = published)) +
    geom_line(data = normal, aes(x = x, y = share), inherit.aes = FALSE, colour = "black",
              linetype = "dashed", linewidth = 0.4) +
    geom_step(linewidth = 0.5, direction = "hv") +
    facet_wrap(~level, nrow = 1) +
    scale_x_log10(breaks = c(0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000),
                  labels = c("0.1", "0.3", "1", "3", "10", "30", "100", "300", "1,000")) +
    scale_y_log10(breaks = 10^seq(0, -10, by = -2),
                  labels = c("1", expression(10^-2), expression(10^-4), expression(10^-6),
                             expression(10^-8), expression(10^-10)),
                  limits = c(floor_share, 1)) +
    scale_colour_manual(values = TAILS_EXPERIMENT_COLOURS, name = "Experiment", drop = FALSE) +
    scale_linetype_manual(values = c("solid", "22"), name = NULL, drop = FALSE) +
    labs(x = "|error| in the model's standard deviations, |z| (log scale)",
         y = "Share of errors beyond |z| (log scale)") +
    theme_bw(base_size = 8) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(colour = "grey90", linewidth = 0.3),
          strip.background = element_blank(), strip.text = element_text(face = "bold"),
          legend.position = "bottom")
}

main <- function(args) {
  if (length(args) != 2) {
    stop("usage: Rscript fig_tails.R <folder with draw_sigma_hist.csv> <out folder>")
  }
  out <- args[2]
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  data <- tails_data(args[1])
  write.csv(data, file.path(out, "fig_tails_data.csv"), row.names = FALSE)
  figure <- plot_tails(data)
  # Elsevier's full text width is 190 mm.
  ggsave(file.path(out, "fig_tails.pdf"), figure, width = 190, height = 85, units = "mm",
         device = if (capabilities("cairo")) cairo_pdf else pdf)
  ggsave(file.path(out, "fig_tails.png"), figure, width = 190, height = 85, units = "mm",
         dpi = 300)
  message("wrote ", out)
}

if (sys.nframe() == 0) main(commandArgs(trailingOnly = TRUE))
