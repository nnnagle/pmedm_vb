# Numbers quoted in the manuscript, read when it renders from results/ and sim/
# (results.zip and sim.zip unzipped beside it), so that every number in the
# text follows the tables. Each lookup must find exactly one value, or it
# stops the render and says which.
#
#   source("results/numbers.R")
#   `r v("2", "absz_p90", "hmc_ref", 0.01)`          one value, 2 decimals
#   `r vrange("4", "width_p50", VB, c(0.01, 1))`     "0.92 to 1.05"
#   `r pct("5", "walls", "vb_gaussian", 0.01)`       a share as a percentage
#
# Paper tables (table<n>_*.csv, experiments/paper_tables.py): rows are (table,
# key, experiment, alpha, method) with ``value`` (and ``low``/``high`` for
# k-hat); IPF has no alpha, so its rows repeat over the alphas and any alpha,
# or none, finds them. Settings (settings.csv) are each experiment's recorded
# spec as ``<experiment>.<path>`` and the tables' options; the census tables'
# counts come from table_inventory.py's CSVs, study_area.csv and parameters.csv;
# Figure A's examples from draw_examples.csv; the simulation's from sim/.

NUMBERS_DIRS <- c(results = "results", sim = "sim")
.numbers_cache <- new.env()

# The methods the text names as groups.
VB <- c("vb_gaussian", "vb_skewed")
VB_ALL <- c("vb_gaussian", "vb_skewed", "vb_gaussian_trunc")
POSTERIOR <- c("vb_gaussian", "vb_skewed", "vb_gaussian_trunc", "hmc_short", "hmc_ref")
ALPHAS <- c(0.01, 1)

num_read <- function(where, name) {
  key <- paste(where, name)
  if (!exists(key, envir = .numbers_cache, inherits = FALSE)) {
    path <- file.path(NUMBERS_DIRS[[where]], name)
    if (!file.exists(path)) stop("numbers.R: ", path, " is missing")
    assign(key, read.csv(path, check.names = FALSE, stringsAsFactors = FALSE),
           envir = .numbers_cache)
  }
  get(key, envir = .numbers_cache)
}

# -- formatting -----------------------------------------------------------------

fmt <- function(x, digits = 2) {
  formatC(round(x, digits), format = "f", digits = digits, big.mark = ",")
}

# "a to b", or one value if they print alike.
range_text <- function(x, digits = 2, sep = " to ") {
  lo <- fmt(min(x), digits)
  hi <- fmt(max(x), digits)
  if (lo == hi) lo else paste0(lo, sep, hi)
}

# Small whole numbers as words, as the text writes them.
WORDS <- c("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
           "eleven", "twelve")
words <- function(n, capital = FALSE) {
  n <- round(n)
  text <- if (n >= 1 && n <= length(WORDS)) WORDS[n] else fmt(n, 0)
  if (capital) paste0(toupper(substring(text, 1, 1)), substring(text, 2)) else text
}

# -- the paper tables -------------------------------------------------------------

paper_table <- function(table) {
  files <- list.files(NUMBERS_DIRS[["results"]],
                      pattern = paste0("^table", table, "_.*\\.csv$"))
  if (length(files) != 1) stop("numbers.R: expected one table", table, "_*.csv in ",
                               NUMBERS_DIRS[["results"]], ", found ", length(files))
  num_read("results", files)
}

# One statistic: the value of (table, key, method, alpha, experiment).
stat <- function(table, key, method, alpha = NULL, experiment = "exp01_baseline",
                 field = "value") {
  t <- paper_table(table)
  rows <- t[t$key == key & t$method == method & t$experiment == experiment, ]
  if (!is.null(alpha)) rows <- rows[abs(rows$alpha - alpha) < 1e-9, ]
  values <- unique(rows[[field]])
  if (length(values) != 1 || is.na(values)) {
    stop(sprintf("numbers.R: table %s, key %s, method %s, alpha %s, %s, %s: %d values",
                 table, key, method, format(alpha), experiment, field, length(values)))
  }
  values
}

# Every combination of methods, alphas and experiments.
stats <- function(table, key, methods, alphas = NULL, experiments = "exp01_baseline",
                  field = "value") {
  grid <- expand.grid(method = methods, alpha = if (is.null(alphas)) NA else alphas,
                      experiment = experiments, stringsAsFactors = FALSE)
  mapply(function(m, a, e) stat(table, key, m, if (is.na(a)) NULL else a, e, field),
         grid$method, grid$alpha, grid$experiment)
}

v <- function(table, key, method, alpha = NULL, experiment = "exp01_baseline", digits = 2,
              field = "value", scale = 1) {
  fmt(scale * stat(table, key, method, alpha, experiment, field), digits)
}

pct <- function(table, key, method, alpha = NULL, experiment = "exp01_baseline", digits = 0) {
  fmt(100 * stat(table, key, method, alpha, experiment), digits)
}

vrange <- function(table, key, methods, alphas = NULL, experiments = "exp01_baseline",
                   digits = 2, field = "value", scale = 1, sep = " to ") {
  range_text(scale * stats(table, key, methods, alphas, experiments, field), digits, sep)
}

pctrange <- function(table, key, methods, alphas = NULL, experiments = "exp01_baseline",
                     digits = 0, sep = " to ") {
  range_text(100 * stats(table, key, methods, alphas, experiments), digits, sep)
}

# k-hat over fits, at every alpha: the smallest low to the largest high.
khat_range <- function(methods, alphas = ALPHAS, experiment = "exp01_baseline", digits = 0) {
  lo <- min(stats("4", "khat", methods, alphas, experiment, "low"))
  hi <- max(stats("4", "khat", methods, alphas, experiment, "high"))
  range_text(c(lo, hi), digits)
}

# The ratio of two statistics, e.g. how many times as long.
ratio <- function(a, b, digits = 1) fmt(a / b, digits)

# -- settings -----------------------------------------------------------------------

setting <- function(key, numeric = TRUE) {
  s <- num_read("results", "settings.csv")
  value <- s$value[s$key == key]
  if (length(value) != 1) stop("numbers.R: setting ", key, ": ", length(value), " values")
  if (numeric) as.numeric(value) else value
}

# An experiment's alphas, as numbers.
alphas <- function(experiment = "exp01_baseline") {
  as.numeric(strsplit(setting(paste0(experiment, ".alphas"), numeric = FALSE), ";")[[1]])
}

# A roll-up's thresholds, "25h10n": persons 25, households 10.
rollup_threshold <- function(which = c("person", "household"), experiment = "exp03_rollup") {
  which <- match.arg(which)
  spec <- setting(paste0(experiment, ".model.rollup"), numeric = FALSE)
  parts <- regmatches(spec, regexec("^([0-9.]+)h([0-9.]+)", spec))[[1]]
  if (length(parts) != 3) stop("numbers.R: roll-up ", spec, " has no thresholds")
  as.numeric(parts[if (which == "person") 2 else 3])
}

# The ACS years, "2020–2024".
acs_years <- function(experiment = "exp01_baseline") {
  year <- setting(paste0(experiment, ".area.year"))
  paste0(year - setting(paste0(experiment, ".area.span")) + 1, "–", year)
}

# -- the census tables and the study area ----------------------------------------------

# "1,234 (56)" -> c(1234, 56).
parse_count <- function(text) {
  as.numeric(gsub(",", "", regmatches(text, gregexpr("[0-9,]+", text))[[1]]))
}

# Constraint cells (what = "cells") or zeros, at a level ("block group", "tract",
# "puma"; NULL for every level), for a scheme ("published", "pruned",
# "roll-up"), of a table or the Total.
cells <- function(level = NULL, scheme = "published", what = c("cells", "zeros"),
                  table = "Total") {
  what <- match.arg(what)
  t <- num_read("results", "table_cells.csv")
  group <- t$Table
  for (i in seq_along(group)) if (group[i] == "" && i > 1) group[i] <- group[i - 1]
  labels <- c(puma = "PUMA", tract = "Tract", `block group` = "Block Group")
  rows <- group == table & (is.null(level) | t$Level %in% labels[level])
  column <- grep(paste0("^", scheme), names(t), value = TRUE)
  if (length(column) != 1 || !any(rows)) stop("numbers.R: cells ", table, " ", scheme)
  sum(vapply(t[rows, column], function(x) parse_count(x)[if (what == "cells") 1 else 2], 0))
}

# Each PUMA's coordinates (or identified ones) in an experiment, parameters.csv.
coords <- function(experiment = "Baseline", which = c("coordinates", "identified")) {
  which <- match.arg(which)
  t <- num_read("results", "parameters.csv")
  column <- grep(paste0("^", experiment, ":"), names(t), value = TRUE)
  rows <- t[["PUMA ID"]] != "Total"
  vapply(t[rows, column], function(x) parse_count(x)[if (which == "coordinates") 1 else 2], 0)
}

# A column of study_area.csv, for a PUMA or the Total; "mean" over the PUMAs.
study <- function(column, puma = "Total") {
  t <- num_read("results", "study_area.csv")
  if (puma == "mean") return(mean(t[[column]][t$puma != "Total"]))
  t[[column]][t$puma == puma]
}

# The number of census tables by role ("constraint", "held out"), and with
# block-group estimates if bg = TRUE, from table_inventory.csv.
n_tables <- function(role, bg = NULL) {
  t <- num_read("results", "table_inventory.csv")
  rows <- startsWith(t$role, role)
  if (!is.null(bg)) rows <- rows & grepl("BG", t$levels) == bg
  sum(rows)
}

# Figure A's examples (draw_examples.csv): each draw's D/H in a case, by method.
example_dh <- function(case, method, selected = NULL) {
  t <- num_read("results", "draw_examples.csv")
  rows <- t[t$case == case & t$method == method, ]
  if (!is.null(selected)) rows <- rows[as.logical(rows$selected) == selected, ]
  draws <- rows[!duplicated(rows$draw), ]
  if (!nrow(draws)) stop("numbers.R: no ", method, " draws in example ", case)
  draws$dh
}

example_households <- function(case) {
  t <- num_read("results", "draw_examples.csv")
  unique(t$households[t$case == case])
}

# The interval score of a method that knew every true count (Gneiting & Raftery
# 2007): the published value is off by a standard normal number of SEs.
oracle_interval_score <- function(level = 0.90) {
  z <- qnorm(1 - (1 - level) / 2)
  2 * z + 2 / (1 - level) * 2 * (dnorm(z) - z * pnorm(z, lower.tail = FALSE))
}

# -- the simulation (sim/) -------------------------------------------------------------

sim_setting <- function(key, numeric = TRUE) {
  s <- num_read("sim", "settings.csv")
  value <- s$value[s$key == key]
  if (length(value) != 1) stop("numbers.R: sim setting ", key, ": ", length(value), " values")
  if (numeric) as.numeric(value) else value
}

sim_draws <- function() sim_setting("sim.draws")

# The share of cells whose truth has one of ``ranks`` among a method's draws
# (ranks.csv); "outer" ranks are 0 and the number of draws.
sim_rank_share <- function(kind, method, ranks = c(0, sim_draws()), level = "block group",
                           role = "heldout", truth_zero = "all") {
  t <- num_read("sim", "ranks.csv")
  rows <- t$kind == kind & t$method == method & t$level == level & t$role == role &
    tolower(as.character(t$truth_zero)) == tolower(truth_zero)
  if (!any(rows)) stop("numbers.R: no ranks of ", method, " for ", kind, " ", role)
  sum(t$share[rows & t$rank %in% ranks])
}

sim_outer_pct <- function(kinds, method, ...) {
  range_text(100 * vapply(kinds, function(k) sim_rank_share(k, method, ...), 0), 0)
}

# A column of summary.csv (e.g. "hmc_short_int_crps"), nonzero truths by default.
sim_stat <- function(kind, column, level = "block group", role = "heldout",
                     truth_zero = FALSE) {
  t <- num_read("sim", "summary.csv")
  rows <- t$kind == kind & t$level == level & t$role == role &
    toupper(as.character(t$truth_zero)) == toupper(as.character(truth_zero))
  value <- t[[column]][rows]
  if (length(value) != 1) stop("numbers.R: sim ", column, " for ", kind, ": ", length(value))
  value
}

# Minutes per fit, over the synthetic PUMAs (timing.csv).
sim_minutes <- function(column = "hmc_seconds", fun = median) {
  fun(num_read("sim", "timing.csv")[[column]], na.rm = TRUE) / 60
}

# The lines of a role at a level (cells.csv): the constraint lines fitted, or
# the dropped lines.
sim_lines <- function(level, role = "constraint") {
  t <- num_read("sim", "cells.csv")
  sum(t$level == level & t$role == role)
}

# A sampling fraction as "1/8".
fraction_text <- function(f) paste0("1/", round(1 / f))

# The number of methods an experiment's cost table has.
n_methods <- function(experiment = "exp01_baseline") {
  t <- paper_table("1")
  length(unique(t$method[t$experiment == experiment]))
}

# A table's zero cells as a percentage of its cells, at a level and scheme.
zero_pct <- function(table = "Total", level = "block group", scheme = "published", digits = 0) {
  fmt(100 * cells(level, scheme, "zeros", table) / cells(level, scheme, "cells", table), digits)
}
