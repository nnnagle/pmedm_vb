# Figure A: what a draw looks like, from draw_examples.py's draw_examples.csv.
#
# Three block groups of one PUMA (columns): a typical one (the median HMC
# reference n_eff), the IPF fit's least spread one and VB Gaussian's least
# spread one. One row per method. Each bar is one draw, a whole synthetic
# population: the block group's households stacked by the household record they
# copy, largest first. The block group's five records with the largest mean
# share over the shown draws (averaged over the methods) are coloured, the same
# record the same colour in every bar of the column; every other record is grey,
# with records under 1% of the households together in the light grey at the
# top (EXAMPLES_SMALL). The number above each bar is its n_eff = 1 / sum s^2 (as Figure B). IPF
# is one fit: its one bar is repeated.
#
# Reads draw_examples.csv from the paper tables' folder (experiments/
# draw_examples.py --out, e.g. <root>/tables_compare). Writes to <out>:
#   fig_examples.pdf, fig_examples.png   the figure
#   fig_examples_data.csv                the values plotted
#
#   Rscript experiments/figures/fig_examples.R <tables folder> <out folder>
#
# Or, from a Quarto chunk, with the CSV in a folder "results":
#   source("results/fig_examples.R")
#   plot_examples(examples_data("results"))

suppressPackageStartupMessages({
  library(ggplot2)
})

# Display labels, in the order the methods are drawn (top to bottom); drop a
# method here to leave it out.
EXAMPLES_METHODS <- c(
  ipf = "IPF",
  ipf_multinomial = "IPF, multinomial",
  vb_gaussian = "VB Gaussian",
  hmc_ref = "HMC reference"
)
EXAMPLES_CASES <- c(
  typical = "Typical block group",
  ipf = "IPF's least spread",
  vb = "VB Gaussian's least spread"
)
# Categorical slots 1-5 of the reference palette, checked for colour-vision
# deficiency separation; here they name records, not methods.
EXAMPLES_COLOURS <- c(`1st` = "#2a78d6", `2nd` = "#eb6834", `3rd` = "#1baf7a",
                      `4th` = "#eda100", `5th` = "#e87ba4",
                      `Other records` = "grey62", `Records under 1% each` = "grey88")
# Records under this share of the block group's households are drawn together.
EXAMPLES_SMALL <- 0.01

examples_data <- function(tables_dir) {
  file <- file.path(tables_dir, "draw_examples.csv")
  if (!file.exists(file)) stop("no draw_examples.csv in ", tables_dir)
  raw <- read.csv(file, colClasses = c(block_group = "character"))
  raw <- raw[raw$method %in% names(EXAMPLES_METHODS) & raw$case %in% names(EXAMPLES_CASES), ]
  # IPF is one fit: repeat its bar as many times as the other methods have draws.
  shown <- max(raw$draw)
  single <- raw$method == "ipf"
  raw <- rbind(raw[!single, ],
               do.call(rbind, lapply(seq_len(shown), function(d) transform(raw[single, ], draw = d))))
  # The small records (and the CSV's own remainder, record -1) as one block per bar.
  small <- raw$record < 0 | raw$share < EXAMPLES_SMALL
  keys <- c("case", "block_group", "households", "method", "draw", "n_eff")
  rest <- aggregate(raw$share[small], raw[small, keys], sum)
  names(rest)[ncol(rest)] <- "share"
  raw <- raw[!small, c(keys, "record", "highlight", "share")]
  raw$group <- ifelse(is.na(raw$highlight), "Other records",
                      names(EXAMPLES_COLOURS)[pmax(1, raw$highlight)])
  raw <- rbind(raw[, c(keys, "record", "group", "share")],
               transform(rest, record = -1, group = "Records under 1% each"))
  # Stack each bar largest first, the small records on top.
  raw <- raw[order(raw$case, raw$method, raw$draw, raw$record < 0, -raw$share), ]
  bar <- interaction(raw$case, raw$method, raw$draw, drop = TRUE)
  raw$top <- ave(raw$share, bar, FUN = cumsum)
  raw$bottom <- raw$top - raw$share
  raw$case_label <- sprintf("%s\n%s, %s households", EXAMPLES_CASES[raw$case], raw$block_group,
                            formatC(round(raw$households), format = "d", big.mark = ","))
  raw$method <- unname(EXAMPLES_METHODS[raw$method])
  raw[, c("case", "case_label", "method", "draw", "record", "group", "share", "bottom", "top",
          "n_eff")]
}

plot_examples <- function(data) {
  data$method <- factor(data$method, levels = unname(EXAMPLES_METHODS))
  data$case_label <- factor(data$case_label,
                            levels = unique(data$case_label[order(match(data$case,
                                                                        names(EXAMPLES_CASES)))]))
  data$group <- factor(data$group, levels = names(EXAMPLES_COLOURS))
  rest <- data$group == "Records under 1% each"
  bars <- unique(data[, c("case_label", "method", "draw", "n_eff")])
  bars$label <- ifelse(bars$n_eff < 10, sprintf("%.1f", bars$n_eff), sprintf("%.0f", bars$n_eff))

  ggplot(data, aes(xmin = draw - 0.42, xmax = draw + 0.42, ymin = bottom, ymax = top,
                   fill = group)) +
    geom_rect(data = data[rest, ], colour = NA) +
    geom_rect(data = data[!rest, ], colour = "white", linewidth = 0.1) +
    geom_text(data = bars, aes(x = draw, y = 1.02, label = label), vjust = 0, size = 1.8,
              colour = "grey30", inherit.aes = FALSE) +
    facet_grid(method ~ case_label) +
    scale_x_continuous(breaks = NULL, expand = expansion(add = 0.3)) +
    scale_y_continuous(breaks = c(0, 0.5, 1), labels = c("0", "50%", "100%"),
                       expand = expansion(mult = c(0, 0.14))) +
    scale_fill_manual(values = EXAMPLES_COLOURS, drop = FALSE,
                      name = "Household records, by mean share\nin the block group") +
    labs(x = "Draws (IPF: its one fit, repeated); above each bar its n_eff",
         y = "Share of the block group's households") +
    theme_bw(base_size = 8) +
    theme(panel.grid = element_blank(), strip.background = element_blank(),
          strip.text = element_text(face = "bold"), legend.position = "bottom")
}

main <- function(args) {
  if (length(args) != 2) {
    stop("usage: Rscript fig_examples.R <tables folder> <out folder>")
  }
  out <- args[2]
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  data <- examples_data(args[1])
  write.csv(data, file.path(out, "fig_examples_data.csv"), row.names = FALSE)
  figure <- plot_examples(data)
  # Elsevier's full text width is 190 mm.
  ggsave(file.path(out, "fig_examples.pdf"), figure, width = 190, height = 140, units = "mm",
         device = if (capabilities("cairo")) cairo_pdf else pdf)
  ggsave(file.path(out, "fig_examples.png"), figure, width = 190, height = 140, units = "mm",
         dpi = 300)
  message("wrote ", out)
}

if (sys.nframe() == 0) main(commandArgs(trailingOnly = TRUE))
