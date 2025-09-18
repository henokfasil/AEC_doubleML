# =============================================================================
# DOUBLE MACHINE LEARNING (DML) WITH MODEL SELECTION FOR DiD ANALYSIS
# =============================================================================
#
# MODELING PROCEDURE:
# This script implements a sophisticated DML approach for difference-in-differences
# causal inference with automated model selection for nuisance functions.
#
# THEORETICAL FOUNDATION:
# 1. DML addresses selection bias in observational data by:
#    - Using machine learning for nuisance function estimation (E[Y|X], E[D|X])
#    - Cross-fitting to avoid overfitting bias
#    - Orthogonalization to achieve √n-convergence and asymptotic normality
#
# 2. The estimand is the Average Treatment Effect (ATE):
#    θ₀ = E[Y₁ - Y₀] where Y₁, Y₀ are potential outcomes
#
# 3. DML uses the partially linear regression (PLR) model:
#    Y = θ₀D + g₀(X) + ε
#    D = m₀(X) + v
#    where g₀(X) = E[Y|X] and m₀(X) = E[D|X] are nuisance functions
#
# IMPLEMENTATION STEPS:
# Step 1: Data preparation and treatment variable construction
# Step 2: Automated model selection via cross-validation metrics
# Step 3: Cross-fitted DML estimation with multiple ML learners
# Step 4: Statistical inference with confidence intervals
# Step 5: Diagnostic plots and robustness checks
#
# =============================================================================
#getwd()
#setwd("C:/Users/TELILA/OneDrive - Universita' degli Studi di Roma Tor Vergata/1_publication/AEC 2023/agent GPT")
# --- HELPER FUNCTION: PACKAGE MANAGEMENT ---
require_or_install <- function(pkgs) {
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) {
      cat("Installing package:", p, "\n")
      install.packages(p)
    }
    library(p, character.only = TRUE)
  }
}

# --- STEP 1: LOAD REQUIRED PACKAGES ---
cat("=== DOUBLE MACHINE LEARNING FOR DiD ANALYSIS ===\n")
cat("Loading required packages...\n")

require_or_install(c(
  "data.table", "dplyr", "readr", "ggplot2",
  "DoubleML", "mlr3", "mlr3learners", "mlr3tuning", "mlr3measures",
  "paradox", "ranger", "glmnet"
))

# --- STEP 2: SET REPRODUCIBILITY SEED ---
set.seed(123)
cat("Reproducibility seed set to 123\n")

# --- STEP 3: DATA LOADING AND PREPROCESSING ---
cat("\nStep 1: Loading and preprocessing data...\n")

# Load data with error handling
if (!file.exists("final_final.csv")) {
  stop("Error: final_final.csv not found in working directory. Please check file path.")
}

df <- readr::read_csv("final_final.csv", show_col_types = FALSE)
cat("Data loaded successfully. Dimensions:", nrow(df), "x", ncol(df), "\n")

# --- TREATMENT VARIABLE CONSTRUCTION ---
# Following standard DiD methodology:
# T = time dummy (1 if post-treatment period)
# Post = treatment group dummy (1 if ever treated)
# DID = T × Post (interaction capturing treatment effect)

df <- df %>%
  dplyr::mutate(
    # Time dummy: treatment period starts in 2019
    T = as.integer(Year >= 2019),

    # Treatment dummy: countries receiving PPP or Green Bond financing
    # Fixed logical condition from original code error
    Post = as.integer((PPP_T > 0 | GB_T > 0) & !is.na(PPP_T) & !is.na(GB_T)),

    # DiD interaction term (our treatment effect of interest)
    DID = T * Post
  )

cat("Treatment variables created:\n")
cat("- T (time dummy):", sum(df$T, na.rm = TRUE), "observations in treatment period\n")
cat("- Post (treatment group):", sum(df$Post, na.rm = TRUE), "treated units\n")
cat("- DID (interaction):", sum(df$DID, na.rm = TRUE), "treated observations in treatment period\n")

# --- STEP 4: DEFINE VARIABLES FOR ANALYSIS ---
# Use log-transformed variables to handle skewness and interpret coefficients as elasticities

# Log-transformed covariates (excluding target to avoid leakage)
log_covars <- intersect(c("ln_PPP_I", "ln_GB_I", "ln_DCM", "ln_GFCFM", "ln_MFDI"),
                       names(df))

# Structural/time variables
struct <- intersect(c("Year", "country_num"), names(df))

# Target outcomes (log-transformed for elasticity interpretation)
targets <- intersect(c("ln_MIVA", "ln_DCM", "ln_GFCFM", "ln_MFDI"), names(df))

cat("\nVariables identified:\n")
cat("- Log covariates:", length(log_covars), "variables\n")
cat("- Structural variables:", length(struct), "variables\n")
cat("- Target outcomes:", length(targets), "variables\n")

# --- STEP 5: MACHINE LEARNING LEARNER SETUP ---
# Create function to build learners with appropriate hyperparameter tuning

make_learner <- function(task, kind) {
  # Simple approach with default hyperparameters to avoid computational issues
  # In practice, hyperparameter tuning can improve performance but adds complexity

  switch(kind,
    "rf_reg" = {
      # Random Forest for regression (outcome model)
      mlr3::lrn("regr.ranger",
                num.trees = 500,
                respect.unordered.factors = "order",
                importance = "impurity")
    },
    "rf_cls" = {
      # Random Forest for classification (treatment model)
      mlr3::lrn("classif.ranger",
                num.trees = 500,
                predict_type = "prob",
                respect.unordered.factors = "order",
                importance = "impurity")
    },
    "lasso_reg" = {
      # Lasso regression with built-in cross-validation
      mlr3::lrn("regr.cv_glmnet", alpha = 1)
    },
    "lasso_cls" = {
      # Lasso classification with built-in cross-validation
      mlr3::lrn("classif.cv_glmnet", alpha = 1, predict_type = "prob")
    },
    "ridge_reg" = {
      # Ridge regression with built-in cross-validation
      mlr3::lrn("regr.cv_glmnet", alpha = 0)
    },
    "ridge_cls" = {
      # Ridge classification with built-in cross-validation
      mlr3::lrn("classif.cv_glmnet", alpha = 0, predict_type = "prob")
    },
    stop("Unknown learner kind: ", kind)
  )
}

# --- STEP 6: PREPARE RESULT CONTAINERS ---
cv_metrics_list <- list()     # Cross-validation performance metrics
dml_estimates_list <- list()  # DML treatment effect estimates
selected_models_list <- list() # Best performing models per outcome

# --- STEP 7: MAIN ANALYSIS LOOP ---
cat("\nStep 2: Beginning DML analysis for each target outcome...\n")

for (y in targets) {
  cat("\n--- Analyzing outcome:", y, "---\n")

  # --- DATA PREPARATION FOR CURRENT TARGET ---
  # Exclude current target from covariates to prevent data leakage
  x_vars <- c(setdiff(log_covars, y), struct)
  cols <- unique(c(y, "DID", x_vars))
  dat <- df[, cols]

  # Remove observations with missing values
  dat <- dat[stats::complete.cases(dat), , drop = FALSE]
  dat$DID <- as.integer(dat$DID)

  # Ensure treatment variable is binary
  stopifnot(all(dat$DID %in% c(0L, 1L)))

  # Skip if insufficient data
  if (nrow(dat) < 30) {
    cat("Insufficient data for", y, "- skipping\n")
    next
  }

  cat("Sample size:", nrow(dat), "observations\n")
  cat("Treatment distribution:", table(dat$DID), "\n")

  # --- CREATE MLR3 TASKS ---
  # Regression task for outcome model g₀(X) = E[Y|X]
  task_y <- mlr3::TaskRegr$new(
    id = paste0("outcome_", y),
    backend = dat[, c(y, x_vars), drop = FALSE],
    target = y
  )

  # Classification task for treatment model m₀(X) = E[D|X]
  # Convert DID to factor with explicit levels for classification
  did_factor <- factor(ifelse(dat$DID == 1L, "1", "0"), levels = c("0", "1"))

  task_d <- mlr3::TaskClassif$new(
    id = paste0("treatment_", y),
    backend = data.frame(DID = did_factor, dat[, x_vars, drop = FALSE]),
    target = "DID",
    positive = "1"  # Specify positive class for AUC calculation
  )

  # --- TRAIN MACHINE LEARNING MODELS ---
  cat("Training ML models for nuisance functions...\n")

  # Create and train learners for each method
  learners <- list(
    rf_y = make_learner(task_y, "rf_reg"),
    rf_d = make_learner(task_d, "rf_cls"),
    lasso_y = make_learner(task_y, "lasso_reg"),
    lasso_d = make_learner(task_d, "lasso_cls"),
    ridge_y = make_learner(task_y, "ridge_reg"),
    ridge_d = make_learner(task_d, "ridge_cls")
  )

  # Train all learners
  learners$rf_y$train(task_y)
  learners$rf_d$train(task_d)
  learners$lasso_y$train(task_y)
  learners$lasso_d$train(task_d)
  learners$ridge_y$train(task_y)
  learners$ridge_d$train(task_d)

  # --- MODEL PERFORMANCE EVALUATION ---
  # Evaluate cross-validated performance for model selection

  evaluate_performance <- function(learner, task, measure) {
    # Simple train-test split for performance evaluation
    # In practice, nested CV would be more robust
    tryCatch({
      resampling <- mlr3::rsmp("cv", folds = 5)
      rr <- mlr3::resample(task, learner, resampling)
      mean(rr$score(measure)[[measure$id]], na.rm = TRUE)
    }, error = function(e) {
      cat("Performance evaluation failed:", e$message, "\n")
      return(NA_real_)
    })
  }

  # Calculate performance metrics
  metrics <- data.frame(
    Target = y,
    Learner = c("RF", "Lasso", "Ridge"),
    R2_Y = c(
      evaluate_performance(learners$rf_y, task_y, mlr3::msr("regr.rsq")),
      evaluate_performance(learners$lasso_y, task_y, mlr3::msr("regr.rsq")),
      evaluate_performance(learners$ridge_y, task_y, mlr3::msr("regr.rsq"))
    ),
    AUC_D = c(
      evaluate_performance(learners$rf_d, task_d, mlr3::msr("classif.auc")),
      evaluate_performance(learners$lasso_d, task_d, mlr3::msr("classif.auc")),
      evaluate_performance(learners$ridge_d, task_d, mlr3::msr("classif.auc"))
    ),
    stringsAsFactors = FALSE
  )

  # Ensure metrics are in valid range [0,1]
  clip01 <- function(z) pmin(1, pmax(0, z, na.rm = TRUE))
  metrics$R2_Y <- clip01(metrics$R2_Y)
  metrics$AUC_D <- clip01(metrics$AUC_D)

  # Composite score for model selection (average of R² and AUC)
  metrics$Composite <- rowMeans(metrics[, c("R2_Y", "AUC_D")], na.rm = TRUE)

  cv_metrics_list[[length(cv_metrics_list) + 1]] <- metrics

  cat("Model performance (Composite scores):\n")
  print(metrics[, c("Learner", "Composite")])

  # --- DOUBLE MACHINE LEARNING ESTIMATION ---
  cat("Performing DML estimation...\n")

  # DML estimation function with cross-fitting
  estimate_dml <- function(ml_y, ml_d, label, n_folds = 5, n_rep = 10) {
    # Create DoubleMLData object
    dt <- data.table::as.data.table(dat[, c(y, "DID", x_vars)])
    dml_data <- DoubleML::DoubleMLData$new(
      data = dt,
      y_col = y,
      d_cols = "DID",
      x_cols = x_vars
    )

    # Store estimates from multiple random splits for robustness
    coefs <- ses <- numeric(n_rep)

    for (i in seq_len(n_rep)) {
      tryCatch({
        # Create DML estimator with cross-fitting
        plr <- DoubleML::DoubleMLPLR$new(
          data = dml_data,
          ml_l = ml_y$clone(deep = TRUE),  # Outcome model
          ml_m = ml_d$clone(deep = TRUE),  # Treatment model
          n_folds = n_folds,
          score = "partialling out"  # Orthogonalization method
        )

        # Fit the model
        plr$fit()

        # Store results
        coefs[i] <- as.numeric(plr$coef)
        ses[i] <- as.numeric(plr$se)

      }, error = function(e) {
        cat("DML estimation failed for rep", i, ":", e$message, "\n")
        coefs[i] <- NA_real_
        ses[i] <- NA_real_
      })
    }

    # Aggregate results across repetitions
    est <- mean(coefs, na.rm = TRUE)
    se <- mean(ses, na.rm = TRUE)
    z_stat <- est / se
    pval <- 2 * (1 - pnorm(abs(z_stat)))

    # Return results
    data.frame(
      Target = y,
      Method = paste0("DML (", label, ")"),
      Coefficient = est,
      Std_Error = se,
      P_value = pval,
      CI_Lower = est - 1.96 * se,
      CI_Upper = est + 1.96 * se,
      CF_SD = sd(coefs, na.rm = TRUE),  # Cross-fitting standard deviation
      stringsAsFactors = FALSE
    )
  }

  # Run DML with each learner combination
  dml_results <- list(
    rf = estimate_dml(learners$rf_y, learners$rf_d, "Random Forest"),
    lasso = estimate_dml(learners$lasso_y, learners$lasso_d, "Lasso"),
    ridge = estimate_dml(learners$ridge_y, learners$ridge_d, "Ridge")
  )

  # Combine results
  dml_combined <- do.call(rbind, dml_results)
  dml_estimates_list[[length(dml_estimates_list) + 1]] <- dml_combined

  # --- MODEL SELECTION ---
  # Select best performing model based on composite score
  best_model <- metrics[which.max(metrics$Composite), ]
  best_estimate <- dml_combined[grep(best_model$Learner, dml_combined$Method, ignore.case = TRUE), ]

  if (nrow(best_estimate) > 0) {
    selected_result <- cbind(
      best_model,
      best_estimate[, c("Coefficient", "Std_Error", "P_value", "CI_Lower", "CI_Upper", "CF_SD")]
    )
    selected_models_list[[length(selected_models_list) + 1]] <- selected_result

    cat("Best model:", best_model$Learner, "with composite score:",
        round(best_model$Composite, 3), "\n")
    cat("Treatment effect estimate:", round(best_estimate$Coefficient, 4),
        "(SE:", round(best_estimate$Std_Error, 4), ")\n")
  }
}

# --- STEP 8: COMPILE AND SAVE RESULTS ---
cat("\nStep 3: Compiling and saving results...\n")

if (length(cv_metrics_list) > 0) {
  metrics_tbl <- do.call(rbind, cv_metrics_list)
  readr::write_csv(metrics_tbl, "dml_metrics_fixed.csv")
  cat("Model metrics saved to: dml_metrics_fixed.csv\n")
}

if (length(dml_estimates_list) > 0) {
  estimates_tbl <- do.call(rbind, dml_estimates_list)
  readr::write_csv(estimates_tbl, "dml_estimates_fixed.csv")
  cat("DML estimates saved to: dml_estimates_fixed.csv\n")
}

if (length(selected_models_list) > 0) {
  selected_tbl <- do.call(rbind, selected_models_list)
  readr::write_csv(selected_tbl, "selected_models_fixed.csv")
  cat("Selected models saved to: selected_models_fixed.csv\n")
}

# --- STEP 9: DIAGNOSTIC PLOTS ---
cat("\nStep 4: Creating diagnostic plots...\n")

# Create plots directory
if (!dir.exists("plots")) dir.create("plots")

# Parallel trends plot function
create_parallel_trends_plot <- function(outcome_var) {
  if (!outcome_var %in% names(df)) {
    cat("Warning: Variable", outcome_var, "not found for parallel trends plot\n")
    return(NULL)
  }

  # Aggregate data by year and treatment status
  plot_data <- df %>%
    dplyr::filter(!is.na(.data[[outcome_var]])) %>%
    dplyr::group_by(Year, Post) %>%
    dplyr::summarise(
      mean_outcome = mean(.data[[outcome_var]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      Group = factor(Post, levels = c(0, 1), labels = c("Control", "Treated"))
    )

  # Create plot
  p <- ggplot(plot_data, aes(x = Year, y = mean_outcome, color = Group, group = Group)) +
    geom_line(size = 1.2) +
    geom_point(size = 2) +
    geom_vline(xintercept = 2019, linetype = "dashed", alpha = 0.7, color = "red") +
    scale_color_manual(values = c("Control" = "#444444", "Treated" = "#1f77b4")) +
    labs(
      title = paste("Parallel Trends Check:", outcome_var),
      subtitle = "Vertical line indicates treatment period (2019+)",
      x = "Year",
      y = "Average Log Outcome",
      color = "Group"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5, size = 10),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_line(color = "grey95")
    )

  return(p)
}

# Generate parallel trends plots for each target
for (target in targets) {
  plot <- create_parallel_trends_plot(target)
  if (!is.null(plot)) {
    filename <- file.path("plots", paste0("parallel_trends_", target, ".png"))
    ggsave(filename, plot, width = 10, height = 6, dpi = 300)
    cat("Parallel trends plot saved:", filename, "\n")
  }
}

# --- FOREST PLOT OF ALL ESTIMATES ---
if (file.exists("dml_estimates_fixed.csv")) {
  est_data <- readr::read_csv("dml_estimates_fixed.csv", show_col_types = FALSE)

  if (nrow(est_data) > 0) {
    # Prepare data for forest plot
    est_data$label <- paste0(est_data$Target, " | ", est_data$Method)
    est_data$significant <- est_data$P_value < 0.05

    # Create forest plot
    forest_plot <- ggplot(est_data, aes(x = reorder(label, Coefficient), y = Coefficient)) +
      geom_point(aes(color = significant), size = 3) +
      geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper, color = significant),
                    width = 0.2, size = 1) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.7) +
      scale_color_manual(values = c("FALSE" = "gray60", "TRUE" = "red"),
                        name = "Significant (p<0.05)") +
      coord_flip() +
      theme_minimal(base_size = 12) +
      labs(
        title = "Treatment Effect Estimates: All Methods and Outcomes",
        subtitle = "Error bars show 95% confidence intervals",
        x = "Outcome | Method",
        y = "Treatment Effect (log points)",
        caption = "Red indicates statistical significance at 5% level"
      ) +
      theme(
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, size = 10),
        legend.position = "bottom",
        panel.background = element_rect(fill = "white", color = NA),
        plot.background = element_rect(fill = "white", color = NA),
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_line(color = "grey95")
      )

    ggsave("plots/forest_plot_all_estimates.png", forest_plot,
           width = 12, height = 8, dpi = 300)
    cat("Forest plot saved: plots/forest_plot_all_estimates.png\n")
  }
}

# --- STEP 10: FINAL SUMMARY ---
cat("\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("DOUBLE MACHINE LEARNING ANALYSIS COMPLETE\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

if (exists("estimates_tbl") && nrow(estimates_tbl) > 0) {
  cat("\nSUMMARY OF TREATMENT EFFECTS:\n")
  cat("------------------------------\n")

  summary_stats <- estimates_tbl %>%
    dplyr::group_by(Target) %>%
    dplyr::summarise(
      Mean_Effect = mean(Coefficient, na.rm = TRUE),
      SD_Effect = sd(Coefficient, na.rm = TRUE),
      Min_PValue = min(P_value, na.rm = TRUE),
      .groups = "drop"
    )

  print(summary_stats)

  # Count significant results
  sig_results <- sum(estimates_tbl$P_value < 0.05, na.rm = TRUE)
  total_results <- sum(!is.na(estimates_tbl$P_value))

  cat("\nSTATISTICAL SIGNIFICANCE:\n")
  cat("-------------------------\n")
  cat("Significant results (p < 0.05):", sig_results, "out of", total_results, "\n")
  cat("Percentage significant:", round(100 * sig_results / total_results, 1), "%\n")
}

cat("\nFILES GENERATED:\n")
cat("----------------\n")
cat("- dml_metrics_fixed.csv: Model performance metrics\n")
cat("- dml_estimates_fixed.csv: Treatment effect estimates\n")
cat("- selected_models_fixed.csv: Best models per outcome\n")
cat("- plots/parallel_trends_*.png: Parallel trends diagnostics\n")
cat("- plots/forest_plot_all_estimates.png: Effect size visualization\n")

# Save workspace
save.image("dml_AEC.RData")
cat("- dml_analysis_complete.RData: Complete workspace\n")

cat("\nAnalysis completed successfully!\n")
cat(paste(rep("=", 70), collapse = ""), "\n")