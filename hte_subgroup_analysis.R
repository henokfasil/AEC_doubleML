# =============================================================================
# HETEROGENEOUS TREATMENT EFFECTS (HTE) ANALYSIS
# SUBGROUP ANALYSIS WITH DOUBLE MACHINE LEARNING
# =============================================================================
#
# This script estimates heterogeneous treatment effects (HTE) using:
# 1. Subgroup analysis by baseline characteristics
# 2. DML estimation for each subgroup
# 3. Comparison of treatment effects across subgroups
#
# Subgroups based on:
# - Baseline manufacturing capacity (ln_MIVA)
# - Treatment intensity (PPP_T + GB_T)
# - Economic development level (if available)
# - Temporal period (early vs late period)
#
# =============================================================================

sink("HTE_analysis_output_baseline_controls.txt")
setwd("C:/Users/henok/OneDrive - Universita' degli Studi di Roma Tor Vergata/1_publication/AEC 2023/final")

# Create HTE results folder structure with baseline controls
if (!dir.exists("hte_results_baseline_controls")) dir.create("hte_results_baseline_controls")
if (!dir.exists("hte_results_baseline_controls/plots")) dir.create("hte_results_baseline_controls/plots", recursive = TRUE)
if (!dir.exists("hte_results_baseline_controls/subgroup_estimates")) dir.create("hte_results_baseline_controls/subgroup_estimates", recursive = TRUE)

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
cat("=== HETEROGENEOUS TREATMENT EFFECTS (HTE) ANALYSIS ===\n")
cat("Loading required packages...\n")

require_or_install(c(
  "data.table", "dplyr", "readr", "ggplot2", "broom", "tidyr",
  "DoubleML", "mlr3", "mlr3learners", "ranger", "glmnet"
))

# --- STEP 2: SET REPRODUCIBILITY SEED ---
set.seed(123)
cat("Reproducibility seed set to 123\n")

# --- STEP 3: DATA LOADING ---
cat("\nLoading data...\n")

if (!file.exists("final_final.csv")) {
  stop("Error: final_final.csv not found in working directory.")
}

df <- readr::read_csv("final_final.csv", show_col_types = FALSE)
cat("Data loaded successfully. Dimensions:", nrow(df), "x", ncol(df), "\n")

# --- STEP 4: DEFINE VARIABLES ---
# UPDATED CONTROL VARIABLES:
# REMOVED: ln_PPP_I, ln_GB_I, country_num
# ADDED: GDP_per_capita_2014, Population_2014, Inflation_2014, Trade_2014
# KEPT: Year

baseline_controls <- intersect(c("GDP_per_capita_2014", "Population_2014", "Inflation_2014", "Trade_2014"), names(df))
struct <- intersect(c("Year"), names(df))
targets <- intersect(c("ln_MIVA", "ln_DCM", "ln_GFCFM", "ln_MFDI"), names(df))

cat("\nVariables identified:\n")
cat("- Baseline controls (2014):", length(baseline_controls), "variables:", paste(baseline_controls, collapse = ", "), "\n")
cat("- Structural variables:", length(struct), "variables:", paste(struct, collapse = ", "), "\n")
cat("- Outcomes:", paste(targets, collapse = ", "), "\n")

# =============================================================================
# STEP 5: CREATE SUBGROUPS BASED ON BASELINE CHARACTERISTICS
# =============================================================================

cat("\n=== CREATING SUBGROUPS FOR HTE ANALYSIS ===\n")

# --- 5.1: BASELINE MANUFACTURING CAPACITY ---
cat("\n--- Subgroup 1: Baseline Manufacturing Capacity ---\n")

# Calculate baseline values (first year observation for each country)
baseline_data <- df %>%
  arrange(Country, Year) %>%
  group_by(Country) %>%
  slice(1) %>%
  select(Country, baseline_ln_MIVA = ln_MIVA,
         baseline_DCM = ln_DCM, baseline_GFCFM = ln_GFCFM) %>%
  ungroup()

# Add baseline values to main dataset
df <- df %>%
  left_join(baseline_data, by = "Country")

# Create binary subgroup indicator based on median baseline MIVA
median_baseline_miva <- median(baseline_data$baseline_ln_MIVA, na.rm = TRUE)
df$high_baseline_capacity <- ifelse(df$baseline_ln_MIVA > median_baseline_miva, 1, 0)

cat("Median baseline ln_MIVA:", round(median_baseline_miva, 3), "\n")
cat("High capacity countries:", sum(df$high_baseline_capacity == 1 & df$treated == 1) /
    sum(df$treated == 1) * 100, "%\n")
cat("Low capacity countries:", sum(df$high_baseline_capacity == 0 & df$treated == 1) /
    sum(df$treated == 1) * 100, "%\n")

# --- 5.2: TREATMENT INTENSITY ---
cat("\n--- Subgroup 2: Treatment Intensity ---\n")

# Calculate average treatment intensity for each country
treatment_intensity <- df %>%
  filter(treated == 1) %>%
  group_by(Country) %>%
  summarise(
    avg_PPP_T = mean(PPP_T, na.rm = TRUE),
    avg_GB_T = mean(GB_T, na.rm = TRUE),
    total_intensity = avg_PPP_T + avg_GB_T,
    .groups = 'drop'
  )

# Add to main dataset
df <- df %>%
  left_join(treatment_intensity %>% select(Country, total_intensity), by = "Country")

# For control countries, set intensity to 0
df$total_intensity[is.na(df$total_intensity)] <- 0

# Create binary subgroup indicator based on median intensity (among treated)
median_intensity <- median(treatment_intensity$total_intensity, na.rm = TRUE)
df$high_intensity <- ifelse(df$total_intensity > median_intensity & df$treated == 1, 1, 0)

cat("Median treatment intensity:", round(median_intensity, 3), "\n")
cat("High intensity treated countries:", sum(df$high_intensity == 1, na.rm = TRUE), "\n")
cat("Low intensity treated countries:", sum(df$treated == 1 & df$high_intensity == 0, na.rm = TRUE), "\n")

# --- 5.3: TEMPORAL PERIOD ---
cat("\n--- Subgroup 3: Temporal Period ---\n")

# Split time period at median year
median_year <- median(df$Year, na.rm = TRUE)
df$late_period <- ifelse(df$Year > median_year, 1, 0)

cat("Early period:", min(df$Year), "-", median_year, "\n")
cat("Late period:", median_year + 1, "-", max(df$Year), "\n")

# --- 5.4: BASELINE CREDIT ACCESS ---
cat("\n--- Subgroup 4: Baseline Credit Access (ln_DCM) ---\n")

median_baseline_dcm <- median(baseline_data$baseline_DCM, na.rm = TRUE)
df$high_baseline_credit <- ifelse(df$baseline_DCM > median_baseline_dcm, 1, 0)

cat("Median baseline ln_DCM:", round(median_baseline_dcm, 3), "\n")

# =============================================================================
# STEP 6: SUBGROUP SUMMARY STATISTICS
# =============================================================================

cat("\n=== SUBGROUP SUMMARY STATISTICS ===\n")

# Create summary table
subgroup_summary <- df %>%
  filter(treated == 1) %>%
  group_by(high_baseline_capacity) %>%
  summarise(
    N_obs = n(),
    N_countries = n_distinct(Country),
    Mean_ln_MIVA = mean(ln_MIVA, na.rm = TRUE),
    Mean_PPP_T = mean(PPP_T, na.rm = TRUE),
    Mean_GB_T = mean(GB_T, na.rm = TRUE),
    .groups = 'drop'
  )

cat("Treated countries by baseline capacity:\n")
print(subgroup_summary)

# Save summary
write.csv(subgroup_summary, "hte_results_baseline_controls/subgroup_summary_statistics.csv", row.names = FALSE)

# =============================================================================
# STEP 7: DML HELPER FUNCTIONS
# =============================================================================

cat("\n=== SETTING UP DML ESTIMATION FUNCTIONS ===\n")

# Function to create Random Forest learner (simplified, no tuning for speed)
make_rf_learner <- function(task, is_classification = FALSE) {
  if (is_classification) {
    mlr3::lrn("classif.ranger",
              predict_type = "prob",
              num.trees = 500,
              mtry = max(1, floor(sqrt(length(task$feature_names)))),
              min.node.size = 5,
              importance = "impurity")
  } else {
    mlr3::lrn("regr.ranger",
              num.trees = 500,
              mtry = max(1, floor(sqrt(length(task$feature_names)))),
              min.node.size = 5,
              importance = "impurity")
  }
}

# Function to estimate DML for a subgroup
estimate_dml_subgroup <- function(data, outcome, covariates, subgroup_name, n_folds = 5, n_rep = 10) {

  cat("\n--- Estimating DML for:", subgroup_name, "| Outcome:", outcome, "---\n")
  cat("Sample size:", nrow(data), "\n")

  # Check sample size
  if (nrow(data) < 30) {
    cat("Warning: Insufficient sample size. Skipping.\n")
    return(NULL)
  }

  # Check treatment variation
  treated_count <- sum(data$treated == 1)
  control_count <- sum(data$treated == 0)

  cat("Treated:", treated_count, "| Control:", control_count, "\n")

  if (treated_count < 10 || control_count < 10) {
    cat("Warning: Insufficient treatment variation. Skipping.\n")
    return(NULL)
  }

  # Prepare data
  cols <- unique(c(outcome, "treated", covariates))
  dat <- data[, cols]
  dat <- dat[stats::complete.cases(dat), , drop = FALSE]
  dat$treated <- as.integer(dat$treated)

  if (nrow(dat) < 30) {
    cat("Warning: Insufficient complete cases. Skipping.\n")
    return(NULL)
  }

  # Standardize continuous variables (exclude Year and treated)
  # All baseline controls (GDP_per_capita_2014, Population_2014, Inflation_2014, Trade_2014) are continuous
  continuous_vars <- setdiff(covariates, c("Year", "treated"))
  for (var in continuous_vars) {
    if (var %in% names(dat)) {
      var_mean <- mean(dat[[var]], na.rm = TRUE)
      var_sd <- sd(dat[[var]], na.rm = TRUE)
      if (var_sd > 0) {
        dat[[var]] <- (dat[[var]] - var_mean) / var_sd
      }
    }
  }

  # Create MLR3 tasks
  task_y <- mlr3::TaskRegr$new(
    id = paste0("outcome_", outcome, "_", subgroup_name),
    backend = dat[, c(outcome, covariates), drop = FALSE],
    target = outcome
  )

  treated_factor <- factor(ifelse(dat$treated == 1L, "1", "0"), levels = c("0", "1"))
  task_d <- mlr3::TaskClassif$new(
    id = paste0("treatment_", outcome, "_", subgroup_name),
    backend = data.frame(treated = treated_factor, dat[, covariates, drop = FALSE]),
    target = "treated",
    positive = "1"
  )

  # Create learners
  ml_y <- make_rf_learner(task_y, is_classification = FALSE)
  ml_d <- make_rf_learner(task_d, is_classification = TRUE)

  # Train learners
  cat("Training ML models...\n")
  ml_y$train(task_y)
  ml_d$train(task_d)

  # DML estimation with multiple repetitions
  dt <- data.table::as.data.table(dat[, c(outcome, "treated", covariates)])
  dml_data <- DoubleML::DoubleMLData$new(
    data = dt,
    y_col = outcome,
    d_cols = "treated",
    x_cols = covariates
  )

  coefs <- ses <- numeric(n_rep)

  for (i in seq_len(n_rep)) {
    tryCatch({
      plr <- DoubleML::DoubleMLPLR$new(
        data = dml_data,
        ml_l = ml_y$clone(deep = TRUE),
        ml_m = ml_d$clone(deep = TRUE),
        n_folds = n_folds,
        score = "partialling out"
      )

      plr$fit()

      coefs[i] <- as.numeric(plr$coef)
      ses[i] <- as.numeric(plr$se)

    }, error = function(e) {
      cat("DML rep", i, "failed:", e$message, "\n")
      coefs[i] <- ses[i] <- NA_real_
    })
  }

  # Aggregate results
  valid_coefs <- !is.na(coefs) & !is.nan(coefs) & is.finite(coefs)

  if (sum(valid_coefs) < n_rep / 2) {
    cat("Warning: Less than 50% of DML repetitions succeeded\n")
    return(NULL)
  }

  est <- mean(coefs, na.rm = TRUE)
  se <- mean(ses, na.rm = TRUE)

  if (is.nan(est) || is.infinite(est)) est <- NA_real_
  if (is.nan(se) || is.infinite(se) || se <= 0) se <- NA_real_

  z_stat <- if (!is.na(est) && !is.na(se) && se > 0) est / se else NA_real_
  pval <- if (!is.na(z_stat)) 2 * (1 - pnorm(abs(z_stat))) else NA_real_

  result <- data.frame(
    Subgroup = subgroup_name,
    Outcome = outcome,
    N_obs = nrow(dat),
    N_treated = treated_count,
    N_control = control_count,
    Coefficient = est,
    Std_Error = se,
    Z_Statistic = z_stat,
    P_value = pval,
    CI_Lower = est - 1.96 * se,
    CI_Upper = est + 1.96 * se,
    CF_SD = sd(coefs, na.rm = TRUE),
    Significant = ifelse(pval < 0.05, "Yes", "No"),
    stringsAsFactors = FALSE
  )

  cat("Coefficient:", round(est, 4), "| SE:", round(se, 4),
      "| P-value:", round(pval, 4), "\n")

  return(result)
}

# =============================================================================
# STEP 8: RUN SUBGROUP ANALYSIS FOR ALL OUTCOMES
# =============================================================================

cat("\n=== RUNNING SUBGROUP DML ANALYSIS ===\n")

all_hte_results <- list()
result_counter <- 1

# Define subgroup definitions
subgroup_definitions <- list(
  list(name = "High_Baseline_Capacity", var = "high_baseline_capacity", value = 1),
  list(name = "Low_Baseline_Capacity", var = "high_baseline_capacity", value = 0),
  list(name = "High_Treatment_Intensity", var = "high_intensity", value = 1),
  list(name = "Low_Treatment_Intensity", var = "high_intensity", value = 0, filter_treated = TRUE),
  list(name = "Late_Period", var = "late_period", value = 1),
  list(name = "Early_Period", var = "late_period", value = 0),
  list(name = "High_Baseline_Credit", var = "high_baseline_credit", value = 1),
  list(name = "Low_Baseline_Credit", var = "high_baseline_credit", value = 0)
)

# Loop through each outcome
for (y in targets) {

  cat("\n", paste(rep("=", 70), collapse = ""), "\n")
  cat("OUTCOME:", y, "\n")
  cat(paste(rep("=", 70), collapse = ""), "\n")

  # Define covariates: baseline controls + Year (no outcome variables as controls)
  x_vars <- c(baseline_controls, struct)

  # Loop through each subgroup definition
  for (subgroup_def in subgroup_definitions) {

    # Filter data for this subgroup
    if (!is.null(subgroup_def$filter_treated) && subgroup_def$filter_treated) {
      # For low intensity, only look at treated countries
      subgroup_data <- df %>%
        filter(treated == 1) %>%
        filter(.data[[subgroup_def$var]] == subgroup_def$value)
    } else {
      subgroup_data <- df %>%
        filter(.data[[subgroup_def$var]] == subgroup_def$value)
    }

    # Run DML for this subgroup
    result <- estimate_dml_subgroup(
      data = subgroup_data,
      outcome = y,
      covariates = x_vars,
      subgroup_name = subgroup_def$name,
      n_folds = 5,
      n_rep = 10
    )

    if (!is.null(result)) {
      all_hte_results[[result_counter]] <- result
      result_counter <- result_counter + 1
    }
  }
}

# =============================================================================
# STEP 9: COMPILE AND SAVE HTE RESULTS
# =============================================================================

cat("\n=== COMPILING HTE RESULTS ===\n")

if (length(all_hte_results) > 0) {

  hte_estimates <- do.call(rbind, all_hte_results)

  # Save all estimates
  write.csv(hte_estimates, "hte_results_baseline_controls/hte_subgroup_estimates.csv", row.names = FALSE)
  cat("HTE estimates saved to: hte_results_baseline_controls/hte_subgroup_estimates.csv\n")

  # Print summary
  cat("\n=== HTE ESTIMATION SUMMARY ===\n")
  cat("Total subgroup-outcome combinations analyzed:", nrow(hte_estimates), "\n")
  cat("Significant results (p < 0.05):", sum(hte_estimates$P_value < 0.05, na.rm = TRUE), "\n")

  # Summary by outcome
  outcome_summary <- hte_estimates %>%
    group_by(Outcome) %>%
    summarise(
      N_subgroups = n(),
      Mean_Effect = mean(Coefficient, na.rm = TRUE),
      SD_Effect = sd(Coefficient, na.rm = TRUE),
      Min_Effect = min(Coefficient, na.rm = TRUE),
      Max_Effect = max(Coefficient, na.rm = TRUE),
      Range = Max_Effect - Min_Effect,
      Significant_Results = sum(P_value < 0.05, na.rm = TRUE),
      .groups = 'drop'
    )

  cat("\nSummary by outcome:\n")
  print(outcome_summary)

  write.csv(outcome_summary, "hte_results_baseline_controls/hte_outcome_summary.csv", row.names = FALSE)

  # =============================================================================
  # STEP 10: HETEROGENEITY TESTS
  # =============================================================================

  cat("\n=== TESTING FOR HETEROGENEITY ===\n")

  heterogeneity_tests <- list()

  for (y in targets) {

    cat("\nOutcome:", y, "\n")

    # Test 1: High vs Low Baseline Capacity
    high_cap <- hte_estimates %>%
      filter(Outcome == y, Subgroup == "High_Baseline_Capacity")
    low_cap <- hte_estimates %>%
      filter(Outcome == y, Subgroup == "Low_Baseline_Capacity")

    if (nrow(high_cap) > 0 && nrow(low_cap) > 0) {
      diff_capacity <- high_cap$Coefficient - low_cap$Coefficient
      se_diff_capacity <- sqrt(high_cap$Std_Error^2 + low_cap$Std_Error^2)
      z_diff_capacity <- diff_capacity / se_diff_capacity
      p_diff_capacity <- 2 * (1 - pnorm(abs(z_diff_capacity)))

      cat("High vs Low Baseline Capacity:\n")
      cat("  Difference:", round(diff_capacity, 4), "\n")
      cat("  P-value:", round(p_diff_capacity, 4), "\n")

      heterogeneity_tests[[length(heterogeneity_tests) + 1]] <- data.frame(
        Outcome = y,
        Comparison = "High_vs_Low_Baseline_Capacity",
        Coef_Group1 = high_cap$Coefficient,
        Coef_Group2 = low_cap$Coefficient,
        Difference = diff_capacity,
        SE_Difference = se_diff_capacity,
        Z_Statistic = z_diff_capacity,
        P_value = p_diff_capacity,
        Significant = ifelse(p_diff_capacity < 0.05, "Yes", "No"),
        stringsAsFactors = FALSE
      )
    }

    # Test 2: High vs Low Treatment Intensity
    high_int <- hte_estimates %>%
      filter(Outcome == y, Subgroup == "High_Treatment_Intensity")
    low_int <- hte_estimates %>%
      filter(Outcome == y, Subgroup == "Low_Treatment_Intensity")

    if (nrow(high_int) > 0 && nrow(low_int) > 0) {
      diff_intensity <- high_int$Coefficient - low_int$Coefficient
      se_diff_intensity <- sqrt(high_int$Std_Error^2 + low_int$Std_Error^2)
      z_diff_intensity <- diff_intensity / se_diff_intensity
      p_diff_intensity <- 2 * (1 - pnorm(abs(z_diff_intensity)))

      cat("High vs Low Treatment Intensity:\n")
      cat("  Difference:", round(diff_intensity, 4), "\n")
      cat("  P-value:", round(p_diff_intensity, 4), "\n")

      heterogeneity_tests[[length(heterogeneity_tests) + 1]] <- data.frame(
        Outcome = y,
        Comparison = "High_vs_Low_Treatment_Intensity",
        Coef_Group1 = high_int$Coefficient,
        Coef_Group2 = low_int$Coefficient,
        Difference = diff_intensity,
        SE_Difference = se_diff_intensity,
        Z_Statistic = z_diff_intensity,
        P_value = p_diff_intensity,
        Significant = ifelse(p_diff_intensity < 0.05, "Yes", "No"),
        stringsAsFactors = FALSE
      )
    }

    # Test 3: Late vs Early Period
    late <- hte_estimates %>%
      filter(Outcome == y, Subgroup == "Late_Period")
    early <- hte_estimates %>%
      filter(Outcome == y, Subgroup == "Early_Period")

    if (nrow(late) > 0 && nrow(early) > 0) {
      diff_period <- late$Coefficient - early$Coefficient
      se_diff_period <- sqrt(late$Std_Error^2 + early$Std_Error^2)
      z_diff_period <- diff_period / se_diff_period
      p_diff_period <- 2 * (1 - pnorm(abs(z_diff_period)))

      cat("Late vs Early Period:\n")
      cat("  Difference:", round(diff_period, 4), "\n")
      cat("  P-value:", round(p_diff_period, 4), "\n")

      heterogeneity_tests[[length(heterogeneity_tests) + 1]] <- data.frame(
        Outcome = y,
        Comparison = "Late_vs_Early_Period",
        Coef_Group1 = late$Coefficient,
        Coef_Group2 = early$Coefficient,
        Difference = diff_period,
        SE_Difference = se_diff_period,
        Z_Statistic = z_diff_period,
        P_value = p_diff_period,
        Significant = ifelse(p_diff_period < 0.05, "Yes", "No"),
        stringsAsFactors = FALSE
      )
    }
  }

  # Compile heterogeneity tests
  if (length(heterogeneity_tests) > 0) {
    heterogeneity_df <- do.call(rbind, heterogeneity_tests)
    write.csv(heterogeneity_df, "hte_results_baseline_controls/heterogeneity_tests.csv", row.names = FALSE)

    cat("\n=== HETEROGENEITY TEST SUMMARY ===\n")
    cat("Significant heterogeneity detected (p < 0.05):",
        sum(heterogeneity_df$P_value < 0.05, na.rm = TRUE), "out of",
        nrow(heterogeneity_df), "tests\n")
  }

  # =============================================================================
  # STEP 11: CREATE HTE VISUALIZATIONS
  # =============================================================================

  cat("\n=== CREATING HTE VISUALIZATIONS ===\n")

  # 1. Forest plot comparing subgroups
  for (y in targets) {

    outcome_data <- hte_estimates %>% filter(Outcome == y)

    if (nrow(outcome_data) > 0) {

      # Create subgroup labels
      outcome_data$Subgroup_Label <- factor(
        outcome_data$Subgroup,
        levels = rev(unique(outcome_data$Subgroup))
      )

      forest_plot <- ggplot(outcome_data, aes(x = Coefficient, y = Subgroup_Label)) +
        geom_point(aes(color = Significant), size = 3) +
        geom_errorbarh(aes(xmin = CI_Lower, xmax = CI_Upper, color = Significant),
                       height = 0.2, size = 1) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "black", alpha = 0.7) +
        scale_color_manual(values = c("Yes" = "#2E86AB", "No" = "#A23B72")) +
        labs(
          title = paste("Heterogeneous Treatment Effects:", gsub("ln_", "", y)),
          subtitle = "DML estimates by subgroup with 95% confidence intervals",
          x = "Treatment Effect (log points)",
          y = "Subgroup",
          color = "Significant\n(p < 0.05)"
        ) +
        theme_minimal() +
        theme(
          plot.background = element_rect(fill = "white", color = NA),
          panel.background = element_rect(fill = "white", color = NA),
          plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, size = 10),
          legend.position = "bottom"
        )

      ggsave(
        paste0("hte_results_baseline_controls/plots/forest_plot_", y, ".png"),
        forest_plot,
        width = 10,
        height = 8,
        dpi = 300,
        bg = "white"
      )
    }
  }

  # 2. Comparison plots for key subgroup pairs
  comparison_data <- hte_estimates %>%
    filter(Subgroup %in% c("High_Baseline_Capacity", "Low_Baseline_Capacity",
                           "High_Treatment_Intensity", "Low_Treatment_Intensity",
                           "Late_Period", "Early_Period"))

  if (nrow(comparison_data) > 0) {

    comparison_plot <- ggplot(comparison_data,
                              aes(x = Outcome, y = Coefficient, fill = Subgroup)) +
      geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
      geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper),
                    position = position_dodge(0.9), width = 0.2) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
      facet_wrap(~Subgroup, ncol = 2) +
      labs(
        title = "Treatment Effects Across Subgroups",
        subtitle = "Comparison of HTE estimates with 95% CI",
        x = "Outcome Variable",
        y = "Treatment Effect (log points)"
      ) +
      scale_fill_brewer(palette = "Set3") +
      theme_minimal() +
      theme(
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none"
      )

    ggsave(
      "hte_results_baseline_controls/plots/subgroup_comparison.png",
      comparison_plot,
      width = 12,
      height = 10,
      dpi = 300,
      bg = "white"
    )
  }

  # 3. Heterogeneity magnitude plot
  if (exists("heterogeneity_df")) {

    het_plot <- ggplot(heterogeneity_df,
                       aes(x = Outcome, y = abs(Difference), fill = Significant)) +
      geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
      facet_wrap(~Comparison, ncol = 1) +
      labs(
        title = "Magnitude of Treatment Effect Heterogeneity",
        subtitle = "Absolute difference between subgroups",
        x = "Outcome Variable",
        y = "Absolute Difference in Treatment Effects",
        fill = "Statistically\nSignificant"
      ) +
      scale_fill_manual(values = c("Yes" = "#2E86AB", "No" = "#A23B72")) +
      theme_minimal() +
      theme(
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom"
      )

    ggsave(
      "hte_results_baseline_controls/plots/heterogeneity_magnitude.png",
      het_plot,
      width = 10,
      height = 10,
      dpi = 300,
      bg = "white"
    )
  }

  cat("HTE visualizations saved to hte_results_baseline_controls/plots/\n")

} else {
  cat("No HTE results to compile.\n")
}

# =============================================================================
# FINAL SUMMARY
# =============================================================================

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("HTE SUBGROUP ANALYSIS COMPLETE\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

cat("\nFILES GENERATED:\n")
cat("- hte_results_baseline_controls/hte_subgroup_estimates.csv: All subgroup DML estimates\n")
cat("- hte_results_baseline_controls/hte_outcome_summary.csv: Summary statistics by outcome\n")
cat("- hte_results_baseline_controls/heterogeneity_tests.csv: Statistical tests for heterogeneity\n")
cat("- hte_results_baseline_controls/subgroup_summary_statistics.csv: Descriptive stats by subgroup\n")
cat("\nVISUALIZATIONS:\n")
cat("- hte_results_baseline_controls/plots/forest_plot_[outcome].png: Forest plots for each outcome\n")
cat("- hte_results_baseline_controls/plots/subgroup_comparison.png: Comparison across subgroups\n")
cat("- hte_results_baseline_controls/plots/heterogeneity_magnitude.png: Magnitude of heterogeneity\n")

cat("\nAnalysis complete!\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

sink()
