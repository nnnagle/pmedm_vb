# Figure A: what a draw looks like, from draw_examples.py's draw_examples.csv.
#
# Four block groups of one PUMA (columns): a typical one (the median HMC
# reference D/H), the least spread one of the HMC reference (smallest median
# D/H over its draws) and of the IPF fit, and the one where VB Gaussian's least
# diverse of 4,000 draws is least diverse: there its VB Gaussian row starts with
# that draw, outlined, then random draws; every other bar is a random draw. One
# row per method. Each bar is one draw, a whole synthetic population: the block
# group's households stacked by the household record they copy, largest first.
# Colour follows the record, the same record the same colour in every bar of
# the column: the IPF fit's five largest records in the block group in
# ColorBrewer's YlOrRd (its largest darkest); in blue, records outside IPF's
# top 20 that are in the top 10 of another method by its mean share over its
# draws (IPF multinomial, VB Gaussian over 4,000 draws, the HMC reference over
# 4,000) -- what a method favours that IPF does not; every other record grey,
# with records under 1% of the households together in the light grey at the
# top (EXAMPLES_SMALL). The number above each bar is its D/H,
# distinct household records per household (as Figure B).
# IPF is one fit: its one bar is repeated.
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
  hmc = "HMC reference's least spread",
  ipf = "IPF's least spread",
  vb_worst = "VB Gaussian's worst draw"
)
# IPF's five largest records: ColorBrewer's YlOrRd, five classes, its largest
# the darkest (RColorBrewer::brewer.pal(5, "YlOrRd"), reversed); a method's top
# records that IPF ranks low: the reference palette's blue.
EXAMPLES_COLOURS <- c(`IPF 1st` = "#BD0026", `IPF 2nd` = "#F03B20", `IPF 3rd` = "#FD8D3C",
                      `IPF 4th` = "#FECC5C", `IPF 5th` = "#FFFFB2",
                      `Another method's top 10, not IPF's top 20` = "#2a78d6",
                      `Other records` = "grey62", `Records under 1% each` = "grey88")
# The cuts the colours use: IPF's top EXAMPLES_IPF_TOP; blue for a method's top
# EXAMPLES_METHOD_TOP outside IPF's top EXAMPLES_IPF_EXCLUDE (draw_examples.py
# writes ranks up to --ipf-ranks 20 and --method-top 10).
EXAMPLES_IPF_TOP <- 5
EXAMPLES_IPF_EXCLUDE <- 20
EXAMPLES_METHOD_TOP <- 10
# Records under this share of the block group's households are drawn together.
EXAMPLES_SMALL <- 0.01

examples_data <- function(tables_dir) {
  file <- file.path(tables_dir, "draw_examples.csv")
  if (!file.exists(file)) stop("no draw_examples.csv in ", tables_dir)
  raw <- read.csv(file, colClasses = c(block_group = "character"))
  raw <- raw[raw$method %in% names(EXAMPLES_METHODS) & raw$case %in% names(EXAMPLES_CASES), ]
  raw$selected <- if (is.null(raw$selected)) FALSE else as.logical(raw$selected)
  # IPF is one fit: repeat its bar as many times as the other methods have draws.
  shown <- max(raw$draw)
  single <- raw$method == "ipf"
  raw <- rbind(raw[!single, ],
               do.call(rbind, lapply(seq_len(shown), function(d) transform(raw[single, ], draw = d))))
  # The small records (and the CSV's own remainder, record -1) as one block per bar.
  small <- raw$record < 0 | raw$share < EXAMPLES_SMALL
  keys <- c("case", "block_group", "households", "method", "draw", "dh", "selected")
  rest <- aggregate(raw$share[small], raw[small, keys], sum)
  names(rest)[ncol(rest)] <- "share"
  raw <- raw[!small, c(keys, "record", "ipf_rank", "method_rank", "share")]
  ipf_top <- !is.na(raw$ipf_rank) & raw$ipf_rank <= EXAMPLES_IPF_TOP
  method_only <- (is.na(raw$ipf_rank) | raw$ipf_rank > EXAMPLES_IPF_EXCLUDE) &
    !is.na(raw$method_rank) & raw$method_rank <= EXAMPLES_METHOD_TOP
  raw$group <- ifelse(ipf_top, names(EXAMPLES_COLOURS)[pmax(1, raw$ipf_rank)],
                      ifelse(method_only, "Another method's top 10, not IPF's top 20",
                             "Other records"))
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
          "dh", "selected")]
}

plot_examples <- function(data) {
  data$method <- factor(data$method, levels = unname(EXAMPLES_METHODS))
  data$case_label <- factor(data$case_label,
                            levels = unique(data$case_label[order(match(data$case,
                                                                        names(EXAMPLES_CASES)))]))
  data$group <- factor(data$group, levels = names(EXAMPLES_COLOURS))
  rest <- data$group == "Records under 1% each"
  bars <- unique(data[, c("case_label", "method", "draw", "dh", "selected")])
  bars$label <- sub("^0", "", ifelse(bars$dh < 0.1, sprintf("%.3f", bars$dh),
                                     sprintf("%.2f", bars$dh)))
  picked <- bars[bars$selected, ]

  ggplot(data, aes(xmin = draw - 0.42, xmax = draw + 0.42, ymin = bottom, ymax = top,
                   fill = group)) +
    geom_rect(data = data[rest, ], colour = NA) +
    geom_rect(data = data[!rest, ], colour = "white", linewidth = 0.1) +
    geom_text(data = bars[!bars$selected, ], aes(x = draw, y = 1.02, label = label), vjust = 0,
              size = 1.5, colour = "grey30", inherit.aes = FALSE) +
    # VB Gaussian's least diverse draw, chosen rather than random: outlined.
    geom_rect(data = picked, aes(xmin = draw - 0.5, xmax = draw + 0.5, ymin = 0, ymax = 1),
              fill = NA, colour = "black", linewidth = 0.5, inherit.aes = FALSE) +
    geom_text(data = picked, aes(x = draw, y = 1.02, label = label), vjust = 0, size = 1.5,
              fontface = "bold", inherit.aes = FALSE) +
    facet_grid(method ~ case_label) +
    scale_x_continuous(breaks = NULL, expand = expansion(add = 0.3)) +
    scale_y_continuous(breaks = c(0, 0.5, 1), labels = c("0", "50%", "100%"),
                       expand = expansion(mult = c(0, 0.14))) +
    scale_fill_manual(values = EXAMPLES_COLOURS, drop = FALSE, name = "Household records") +
    guides(fill = guide_legend(nrow = 2)) +
    labs(x = paste("Draws (IPF: its one fit, repeated); above each bar its D/H; outlined:",
                   "VB Gaussian's least diverse of 4,000 draws"),
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
