# Figure S: the simulation against known truth (experiments/sim_score.py).
#
# Truths are populations of one PUMA drawn from the baseline's reference HMC
# and from its IPF multinomial draws; each was surveyed (1 in 8, Bernoulli,
# within block groups) and refitted to a smaller constraint set. Held-out
# tables, nonzero truths, one level (block group by default):
#   top     the truth's rank among each method's 16 draws, as the share of
#           cells per rank (0: the truth below every draw; 16: above every
#           draw). Flat at 1/17 (dashed) if the draws and the truth come from
#           one distribution; a U, too narrow; a slope, biased. One row per
#           kind of truth, one column per sampled method;
#   bottom  accuracy: the median |estimate - truth| (IPF's fit; the other
#           methods' draws' mean) and the mean CRPS of the draws (for IPF,
#           its absolute error), in units of the cell's SE; lower is better.
#           Colour is the kind of truth.
#
# Reads ranks.csv and summary.csv from the scores folder
# (<root>/<sim>/simulation/output/scores), and writes to <out>:
#   fig_sim.pdf, fig_sim.png             the figure
#   fig_sim_ranks.csv, fig_sim_scores.csv  the values plotted
#
#   Rscript experiments/figures/fig_sim.R <scores folder> <out folder> [level]
#
# Or, from a Quarto chunk, with sim.zip unzipped into the project:
#   source("sim/fig_sim.R")
#   plot_sim(sim_data("sim"))

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

# Display labels, in drawing order; colours as in Figure C, IPF's fit in grey.
SIM_METHODS <- c(ipf = "IPF", ipf_multinomial = "IPF, multinomial",
                 vb_gaussian = "VB Gaussian", hmc_short = "HMC short")
SIM_METHOD_COLOURS <- c(IPF = "#5b5b5b", `IPF, multinomial` = "#2a78d6",
                        `VB Gaussian` = "#1baf7a", `HMC short` = "#e87ba4")
SIM_KINDS <- c(hmc_ref = "Truth: HMC reference draw",
               ipf_multinomial = "Truth: IPF multinomial draw")
SIM_KIND_COLOURS <- c(`Truth: HMC reference draw` = "#008300",
                      `Truth: IPF multinomial draw` = "#eb6834")
SIM_MEASURES <- c(abs_error_p50 = "Median |error| / SE", crps = "Mean CRPS / SE")

nonzero <- function(x) as.character(x) %in% c("FALSE", "False", "false")

sim_data <- function(scores_dir, role = "heldout") {
  ranks <- read.csv(file.path(scores_dir, "ranks.csv"))
  ranks <- ranks[ranks$role == role & nonzero(ranks$truth_zero) &
                   ranks$method %in% names(SIM_METHODS), ]
  # Every rank 0..m in every panel, an empty bin as 0.
  m <- round(1 / ranks$nominal[1]) - 1
  grid <- unique(ranks[, c("kind", "level", "method", "nominal")])
  grid <- merge(grid, data.frame(rank = 0:m))
  ranks <- merge(grid, ranks[, c("kind", "level", "method", "rank", "share")],
                 all.x = TRUE)
  ranks$share[is.na(ranks$share)] <- 0
  summary <- read.csv(file.path(scores_dir, "summary.csv"))
  summary <- summary[summary$role == role & nonzero(summary$truth_zero), ]
  scores <- do.call(rbind, lapply(names(SIM_MEASURES), function(measure) {
    do.call(rbind, lapply(names(SIM_METHODS), function(method) {
      column <- paste0(method, "_", measure)
      if (!column %in% names(summary)) return(NULL)
      data.frame(kind = summary$kind, level = summary$level, method = method,
                 measure = measure, value = summary[[column]], cells = summary$cells)
    }))
  }))
  list(ranks = ranks, scores = scores)
}

plot_sim <- function(data, level = "block group") {
  ranks <- data$ranks[data$ranks$level == level, ]
  scores <- data$scores[data$scores$level == level, ]
  ranks$method <- factor(SIM_METHODS[ranks$method], levels = SIM_METHODS[-1])
  ranks$kind <- factor(SIM_KINDS[ranks$kind], levels = SIM_KINDS)
  scores$method <- factor(SIM_METHODS[scores$method], levels = rev(SIM_METHODS))
  scores$kind <- factor(SIM_KINDS[scores$kind], levels = SIM_KINDS)
  scores$measure <- factor(SIM_MEASURES[scores$measure], levels = SIM_MEASURES)
  m <- max(ranks$rank)

  top <- ggplot(ranks, aes(x = rank, y = share, fill = method)) +
    geom_col(width = 0.85) +
    geom_hline(aes(yintercept = nominal), linetype = "dashed", linewidth = 0.3,
               colour = "grey30") +
    facet_grid(kind ~ method) +
    scale_x_continuous(breaks = c(0, m / 2, m), expand = expansion(add = 0.6)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
    scale_fill_manual(values = SIM_METHOD_COLOURS, guide = "none") +
    labs(x = paste0("Rank of the true count among the method's ", m, " draws"),
         y = "Share of cells") +
    theme_bw(base_size = 8) +
    theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
          panel.grid.major.y = element_line(colour = "grey90", linewidth = 0.3),
          strip.background = element_blank(), strip.text = element_text(face = "bold"))

  bottom <- ggplot(scores, aes(x = value, y = method, colour = kind)) +
    geom_point(size = 1.8, position = position_dodge(width = 0.5)) +
    facet_wrap(~measure, nrow = 1, scales = "free_x") +
    scale_x_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.08))) +
    scale_colour_manual(values = SIM_KIND_COLOURS, name = NULL) +
    labs(x = "Lower is better", y = NULL) +
    theme_bw(base_size = 8) +
    theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_line(colour = "grey90", linewidth = 0.3),
          strip.background = element_blank(), strip.text = element_text(face = "bold"),
          legend.position = "bottom")

  top / bottom + plot_layout(heights = c(2, 1))
}

main <- function(args) {
  if (!length(args) %in% c(2, 3)) {
    stop("usage: Rscript fig_sim.R <scores folder> <out folder> [level]")
  }
  out <- args[2]
  level <- if (length(args) == 3) args[3] else "block group"
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  data <- sim_data(args[1])
  write.csv(data$ranks, file.path(out, "fig_sim_ranks.csv"), row.names = FALSE)
  write.csv(data$scores, file.path(out, "fig_sim_scores.csv"), row.names = FALSE)
  figure <- plot_sim(data, level)
  # Elsevier's full text width is 190 mm.
  ggsave(file.path(out, "fig_sim.pdf"), figure, width = 190, height = 150, units = "mm",
         device = if (capabilities("cairo")) cairo_pdf else pdf)
  ggsave(file.path(out, "fig_sim.png"), figure, width = 190, height = 150, units = "mm",
         dpi = 300)
  message("wrote ", out)
}

if (sys.nframe() == 0) main(commandArgs(trailingOnly = TRUE))
