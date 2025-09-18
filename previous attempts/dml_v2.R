# ============================================================
# DML model-selection metrics + estimation (logs-only)
#
# This script extends the original DML pipeline by computing formal
# model-selection metrics for the nuisance learners and then running
# cross-fitted PLR estimators.  It also includes functions to
# generate parallel-trends plots and a forest plot summarizing
# treatment effects.
#
# Inputs: final_final.csv in the working directory.  The dataset
# must include log-transformed variables ln_MIVA, ln_DCM, ln_GFCFM,
# ln_MFDI and the log covariates ln_PPP_I, ln_GB_I, ln_DCM,
# ln_GFCFM, ln_MFDI, plus Year and country_num.
#
# Outputs (written to the working directory):
#   - dml_metrics.csv : Cross-validated R2/AUC and composite score per
#     learner and target.
#   - dml_estimates.csv : DML estimates, standard errors, p-values,
#     confidence intervals, and cross-fit standard deviations for
#     each learner and target.
#   - selected_models.csv : One row per target indicating the
#     learner with the highest composite score (with tie-breaks)
#     and the corresponding effect estimate.
#   - plots/parallel_<target>.png : parallel-trends plots for each log
#     outcome (treated vs control over time).
#   - plots/forest_effects.png : forest plot summarising the
#     treatment effect estimates across all models and targets.

require_or_install <- function(pkgs) {
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
    library(p, character.only = TRUE)
  }
}

# --- Step 1: Load required packages ---
require_or_install(c(
  "data.table", "dplyr", "readr", "ggplot2",
  "DoubleML", "mlr3", "mlr3learners", "mlr3tuning", "mlr3measures",
  "paradox", "ranger", "glmnet"
))

# --- Step 2: Set seed for reproducibility ---
set.seed(123)

# --- Step 3: Load data and construct treatment indicators ---
df <- readr::read_csv("final_final.csv", show_col_types = FALSE) %>%
  dplyr::mutate(
    T    = as.integer(Year >= 2019),
    Post = as.integer((PPP_T > 0 | GB_T > 0) & !is.na(PPP_T) & !is.na(GB_T)),
    DID  = T * Post
  )

# --- Step 4: Define covariates and targets ---
log_covars <- intersect(c("ln_PPP_I", "ln_GB_I", "ln_DCM",
                          "ln_GFCFM", "ln_MFDI"), names(df))
struct     <- intersect(c("Year", "country_num"), names(df))
targets    <- intersect(c("ln_MIVA", "ln_DCM", "ln_GFCFM", "ln_MFDI"), names(df))

# --- Step 5: Helper to create AutoTuner objects for each learner ---
#make_autotuner <- function(task, kind) {
  # For Random Forest regressors and classifiers we tune mtry,
  # min.node.size and max.depth.  For Lasso and Ridge we rely on
  # glmnet cross-validation (no hyperparameters other than alpha).
  if (kind == "rf_reg") {
    learner <- lrn("regr.ranger", num.trees = 500, respect.unordered.factors = "order")
    ps <- paradox::ps(
      mtry = paradox::p_int(lower = 1, upper = max(1L, floor(ncol(task$backend$data()) / 3))),
      min.node.size = paradox::p_int(lower = 1, upper = 20),
      max.depth = paradox::p_int(lower = 0, upper = 20)
    )
    AutoTuner$new(
      learner = learner,
      resampling = rsmp("cv", folds = 5),
      measure = mlr3measures::rsq,
      search_space = ps,
      terminator = trm("evals", n_evals = 30),
      tuner = tnr("grid_search")
    )
  } else if (kind == "rf_cls") {
    learner <- lrn("classif.ranger", num.trees = 500, predict_type = "prob",
                   respect.unordered.factors = "order")
    ps <- paradox::ps(
      mtry = paradox::p_int(lower = 1, upper = max(1L, floor(ncol(task$backend$data()) / 3))),
      min.node.size = paradox::p_int(lower = 1, upper = 20),
      max.depth = paradox::p_int(lower = 0, upper = 20)
    )
    AutoTuner$new(
      learner = learner,
      resampling = rsmp("cv", folds = 5),
      measure = mlr3measures::auc,
      search_space = ps,
      terminator = trm("evals", n_evals = 30),
      tuner = tnr("grid_search")
    )
  } else if (kind == "lasso_reg") {
    lrn("regr.cv_glmnet", alpha = 1)
  } else if (kind == "lasso_cls") {
    lrn("classif.cv_glmnet", alpha = 1, predict_type = "prob")
  } else if (kind == "ridge_reg") {
    lrn("regr.cv_glmnet", alpha = 0)
  } else if (kind == "ridge_cls") {
    lrn("classif.cv_glmnet", alpha = 0, predict_type = "prob")
  } else {
    stop("Unknown learner kind: ", kind)
  }
}
make_autotuner <- function(kind, nfeat){
    mtry_upper <- max(1L, floor(nfeat/3))
  if (kind == "rf_reg"){
    lrn <- lrn("regr.ranger", num.trees = 500, respect.unordered.factors = "order")
    ps  <- paradox::ps(
      mtry          = paradox::p_int(lower = 1L, upper = mtry_upper),
      min.node.size = paradox::p_int(lower = 1L, upper = 20L),
      max.depth     = paradox::p_int(lower = 0L,  upper = 20L)
    )
    return(AutoTuner$new(
      learner     = lrn,
      resampling  = rsmp("cv", folds = 5),
      measure     = mlr3measures::rsq,
      search_space= ps,
      terminator  = trm("evals", n_evals = 30),
      tuner       = tnr("grid_search")
    ))
  }
  
  if (kind == "rf_cls"){
    lrn <- lrn("classif.ranger", num.trees = 500, predict_type = "prob",
               respect.unordered.factors = "order")
    ps  <- paradox::ps(
      mtry          = paradox::p_int(lower = 1L, upper = mtry_upper),
      min.node.size = paradox::p_int(lower = 1L, upper = 20L),
      max.depth     = paradox::p_int(lower = 0L,  upper = 20L)
    )
    return(AutoTuner$new(
      learner     = lrn,
      resampling  = rsmp("cv", folds = 5),
      measure     = mlr3measures::auc,
      search_space= ps,
      terminator  = trm("evals", n_evals = 30),
      tuner       = tnr("grid_search")
    ))
  }
  
  if (kind == "lasso_reg") return(lrn("regr.cv_glmnet", alpha = 1))
  if (kind == "lasso_cls") return(lrn("classif.cv_glmnet", alpha = 1, predict_type = "prob"))
  if (kind == "ridge_reg") return(lrn("regr.cv_glmnet", alpha = 0))
  if (kind == "ridge_cls") return(lrn("classif.cv_glmnet", alpha = 0, predict_type = "prob"))
  stop("Unknown kind")
}

# --- Step 6: Prepare containers for results ---
cv_metrics_list <- list()
dml_estimates_list <- list()
selected_models_list <- list()



#_----------------------------------------------------------
#---------------------------------------------------------------

# --- Step 7: Main loop over targets ---
for (y in targets) {
  # Exclude the current target from covariates to avoid leakage
  x_vars <- c(setdiff(log_covars, y), struct)
  cols   <- unique(c(y, "DID", x_vars))
  dat    <- df[, cols]
  dat    <- dat[stats::complete.cases(dat), , drop = FALSE]
  if (nrow(dat) < 30) next  # skip small samples

  # Construct regression and classification tasks
  task_y <- TaskRegr$new(id = paste0("y_", y), backend = dat[, c(y, x_vars), drop = FALSE], target = y)
  task_d <- TaskClassif$new(id = paste0("d_", y), backend = data.frame(DID = as.factor(dat$DID), dat[, x_vars, drop = FALSE]), target = "DID", positive = "1")

  # Build and train AutoTuners / learners
  #at_rf_y <- make_autotuner(task_y, "rf_reg"); at_rf_d <- make_autotuner(task_d, "rf_cls")
  #la_y    <- make_autotuner(task_y, "lasso_reg"); la_d <- make_autotuner(task_d, "lasso_cls")
  #ri_y    <- make_autotuner(task_y, "ridge_reg"); ri_d <- make_autotuner(task_d, "ridge_cls")
  
  nfeat <- length(x_vars)
  at_rf_y <- make_autotuner("rf_reg",  nfeat); at_rf_d <- make_autotuner("rf_cls",  nfeat)
  la_y    <- make_autotuner("lasso_reg", nfeat); la_d <- make_autotuner("lasso_cls", nfeat)
  ri_y    <- make_autotuner("ridge_reg", nfeat); ri_d <- make_autotuner("ridge_cls", nfeat)
  
  at_rf_y$train(task_y); at_rf_d$train(task_d)
  la_y$train(task_y); la_d$train(task_d)
  ri_y$train(task_y); ri_d$train(task_d)

  # Extract cross-validated performance
  get_mean <- function(archive, col) {
    if (col %in% names(archive$data)) {
      return(mean(archive$data[[col]], na.rm = TRUE))
    }
    return(NA_real_)
  }
  m_rf <- data.frame(
    Target = y, Learner = "RF",
    R2_Y  = get_mean(at_rf_y$archive, "rsq"),
    AUC_D = get_mean(at_rf_d$archive, "auc"),
    stringsAsFactors = FALSE
  )
  m_la <- data.frame(
    Target = y, Learner = "Lasso",
    R2_Y  = get_mean(la_y$archive, "rsq"),
    AUC_D = get_mean(la_d$archive, "auc"),
    stringsAsFactors = FALSE
  )
  m_ri <- data.frame(
    Target = y, Learner = "Ridge",
    R2_Y  = get_mean(ri_y$archive, "rsq"),
    AUC_D = get_mean(ri_d$archive, "auc"),
    stringsAsFactors = FALSE
  )
  mets <- rbind(m_rf, m_la, m_ri)
  # Clip metrics into [0,1] for safety and compute composite score
  clip01 <- function(z) pmin(1, pmax(0, z))
  mets$R2_Y  <- clip01(mets$R2_Y)
  mets$AUC_D <- clip01(mets$AUC_D)
  mets$Composite <- rowMeans(mets[, c("R2_Y", "AUC_D")], na.rm = TRUE)
  cv_metrics_list[[length(cv_metrics_list) + 1]] <- mets
#---------------------------
  # Helper to compute cross-fitted DML estimates with repeated cross-fitting
  one_dml <- function(ml_y, ml_d, label, n_folds = 5, n_rep = 20) {
    # Build DoubleMLData
    dt <- as.data.table(cbind(dat[, c(y, "DID"), drop = FALSE], dat[, x_vars, drop = FALSE]))
    dml_data <- DoubleMLData$new(data = dt, y_col = y, d_cols = "DID", x_cols = x_vars)
    coefs <- ses <- numeric(n_rep)
    for (i in seq_len(n_rep)) {
      plr <- DoubleMLPLR$new(data = dml_data,
                              ml_l = ml_y$clone(deep = TRUE),
                              ml_m = ml_d$clone(deep = TRUE),
                              n_folds = n_folds,
                              score = "partialling out")
      plr$fit()
      coefs[i] <- as.numeric(plr$coef)
      ses[i]   <- as.numeric(plr$se)
    }
    est <- mean(coefs, na.rm = TRUE)
    se  <- mean(ses,   na.rm = TRUE)
    z   <- est / se
    pval <- 2 * (1 - pnorm(abs(z)))
    data.frame(
      Target      = y,
      Method      = paste0("DML (", label, ")"),
      Coefficient = est,
      Std_Error   = se,
      P_value     = pval,
      CI_Lower    = est - 1.96 * se,
      CI_Upper    = est + 1.96 * se,
      CF_SD       = sd(coefs, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  # Compute DML estimates for each learner
  dml_rf <- one_dml(at_rf_y, at_rf_d, "Random Forest (tuned)")
  dml_la <- one_dml(la_y,     la_d,   "Lasso (tuned)")
  dml_ri <- one_dml(ri_y,     ri_d,   "Ridge (tuned)")
  dml_estimates_list[[length(dml_estimates_list) + 1]] <- rbind(dml_rf, dml_la, dml_ri)
  # Select the learner with highest composite score (tie-break by CF_SD then Std_Error)
  pick <- mets[order(-mets$Composite),][1,]
  chosen <- subset(
    rbind(dml_rf, dml_la, dml_ri),
    grepl(paste0("DML (", pick$Learner), Method, fixed = TRUE)
  )
  chosen <- chosen[order(chosen$CF_SD, chosen$Std_Error),][1, ]
  selected_models_list[[length(selected_models_list) + 1]] <- cbind(pick, chosen[, c("Coefficient","Std_Error","P_value","CI_Lower","CI_Upper","CF_SD")])
}


#------------------------------------------------------------------------
# --- Step 8: Save results to CSV ---
metrics_tbl   <- do.call(rbind, cv_metrics_list)
estimates_tbl <- do.call(rbind, dml_estimates_list)
selected_tbl  <- do.call(rbind, selected_models_list)
readr::write_csv(metrics_tbl,   "dml_metrics.csv")
readr::write_csv(estimates_tbl, "dml_estimates.csv")
readr::write_csv(selected_tbl,  "selected_models.csv")

# --- Step 9: Produce diagnostic plots ---
require_or_install(c("ggplot2", "dplyr"))

# Create directory for plots
if (!dir.exists("plots")) dir.create("plots")

# Parallel-trends plots
if ("treated" %in% names(df)) {
  plot_parallel <- function(yvar) {
    agg <- df %>%
      dplyr::select(Year, treated, !!yvar) %>%
      dplyr::filter(!is.na(.data[[yvar]])) %>%
      dplyr::group_by(Year, treated) %>%
      dplyr::summarise(mean_y = mean(.data[[yvar]], na.rm = TRUE), .groups = "drop")
    ggplot(agg, aes(x = Year, y = mean_y, color = factor(treated), group = factor(treated))) +
      geom_line(size = 1) + geom_point() +
      scale_color_manual(values = c("#444444", "#1f77b4"), labels = c("Control", "Treated")) +
      labs(title = paste("Parallel Trends:", yvar), x = NULL, y = "Average log outcome", color = NULL) +
      theme_minimal(base_size = 12)
  }
  for (y in targets) {
    g <- plot_parallel(y)
    ggsave(filename = file.path("plots", paste0("parallel_", y, ".png")), plot = g, width = 7.5, height = 4.5, dpi = 300)
  }
}

# Forest plot of all estimates (from dml_estimates.csv or existing results) ---
est_for_plot <- if (file.exists("dml_estimates.csv")) {
  readr::read_csv("dml_estimates.csv", show_col_types = FALSE)
} else if (file.exists("did_dml_fixed_results.csv")) {
  readr::read_csv("did_dml_fixed_results.csv", show_col_types = FALSE)
} else {
  NULL
}
if (!is.null(est_for_plot)) {
  est_for_plot$label <- paste0(est_for_plot$Target, " | ", est_for_plot$Method)
  g_forest <- ggplot(est_for_plot, aes(x = reorder(label, Coefficient), y = Coefficient)) +
    geom_point() +
    geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), width = 0.15) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    coord_flip() + theme_minimal(base_size = 12) +
    labs(title = "Treatment Effects (log outcomes)", x = NULL, y = "Effect (log points)")
  ggsave("plots/forest_effects.png", g_forest, width = 8.5, height = 7.5, dpi = 300)
}