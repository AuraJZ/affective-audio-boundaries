# =============================================================================
# Build every manuscript figure from source, in one pass.
#
# There was no single entry point before this; figures were rendered one script
# at a time, which is how a figure can quietly fall out of step with the text
# after an upstream CSV changes. Each figure script ends in its own stopifnot()
# block asserting the numbers the manuscript quotes from it, so running them all
# is also the check that figures and prose still agree.
#
#     Rscript build_all.R           # everything
#     Rscript build_all.R fig1 fig7 # a subset, by name fragment
#
# A failing assertion stops that figure and is reported at the end. The build
# does NOT abort on the first failure: when several figures are stale at once,
# seeing all of them in one run is worth more than stopping early.
# =============================================================================

# Run from figures_r/. Every figure script uses paths relative to it
# (source("_theme.R"), ../reports/source_data), so rather than guess the script's
# own location -- sys.frame()$ofile does not exist under Rscript -- we assert the
# working directory and fail loudly if it is wrong.
if (!file.exists("_theme.R")) {
  stop("run this from figures_r/ (no _theme.R in ", getwd(), ")")
}

SCRIPTS <- c(
  "fig1_domain_wall.R",   # Fig 1: zero-shot transfer, four representations
  "fig2_convergence.R",   # Fig S9: attribution routes dissociate
  "fig_price.R",          # Fig 2: what labels and alignment recover
  "fig3_tonality.R",      # Fig 3: a targeted representation change
  "fig4_intervention.R",  # Fig 4: model output under audio edits
  "fig5_calibration.R",   # Fig 5: controls and stimulus-level reliability
  "fig6_design.R",        # Fig 6: requirement, supply, per-setting break-even
  "extended_data.R",      # Figs S1-S7
  "ed8_invariance.R"      # Fig S8
)

# fig_boundaries.R is deliberately NOT built. It drew the four-boundary summary
# that encoded the budget/information dichotomy, and the dichotomy is withdrawn
# (see reports/TAC_revision_audit.md). The script is kept so the withdrawal can
# be audited against what it actually drew, but the figure is not in the paper
# and rebuilding it would put a stale PDF back in reports/figures_r/.

args <- commandArgs(trailingOnly = TRUE)
todo <- if (length(args)) {
  SCRIPTS[vapply(SCRIPTS, function(s) any(vapply(args, grepl, logical(1),
                                                 x = s, fixed = TRUE)),
                 logical(1))]
} else SCRIPTS

if (!length(todo)) stop("no figure script matched: ", paste(args, collapse = " "))

t0 <- Sys.time()
status <- character(0)

for (s in todo) {
  message("\n", strrep("=", 70), "\n", s)
  # Each script sources _theme.R and defines its own objects; a fresh
  # environment per script keeps one figure's variables from leaking into the
  # next, which matters because several share names like `d`, `long`, `fig`.
  env <- new.env(parent = globalenv())
  ok <- tryCatch({
    sys.source(s, envir = env)
    TRUE
  }, error = function(e) {
    message("  FAILED: ", conditionMessage(e))
    FALSE
  })
  status[s] <- if (ok) "ok" else "FAILED"
  rm(env); invisible(gc(verbose = FALSE))
}

el <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")))
message("\n", strrep("=", 70))
for (s in names(status)) {
  message(sprintf("  %-24s %s", s, status[[s]]))
}
bad <- names(status)[status == "FAILED"]
message(sprintf("\n%d/%d built in %ds", sum(status == "ok"), length(status), el))
if (length(bad)) {
  message("\nfailed: ", paste(bad, collapse = ", "))
  message("A failure here is usually an assertion, not a plotting error: the ",
          "figure's numbers no longer match what the manuscript quotes.")
  quit(status = 1)
}
