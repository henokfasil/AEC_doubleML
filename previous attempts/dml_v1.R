# ============================================================
# DML + DiD (logs-only) — fixed version that produced latest CSV
#
# This script reproduces the latest results (did_dml_fixed_results.csv).
# Key changes vs. earlier draft:
#  - Logs-only outcomes & covariates
#  - For each target Y, exclude that log variable from X to avoid leakage
#  - Classical DiD (HC1 robust)
#  - DML with cross-fitting using three nuisance learners: RF, Lasso, Ridge
#  - Writes a tidy CSV of results per Target × Method
# ============================================================
getwd()
# Packages
need <- c("data.table","dplyr","readr","sandwich","lmtest",
          "DoubleML","mlr3","mlr3learners","ranger","glmnet","broom")
for (p in need) if (!requireNamespace(p, quietly=TRUE)) install.packages(p)
invisible(lapply(need, library, character.only=TRUE))
set.seed(123)

# Load data
DF <- readr::read_csv("final_final.csv", show_col_types = FALSE)

# Treatment/time
DF <- DF %>% mutate(
  T    = as.integer(Year >= 2019),
  Post = as.integer((PPP_T > 0 | GB_T > 0) & !is.na(PPP_T) & !is.na(GB_T)),
  DID  = T * Post
)

# Logs-only covariates
base_logs <- c("ln_PPP_I","ln_GB_I","ln_DCM","ln_GFCFM","ln_MFDI")
base_logs <- base_logs[base_logs %in% names(DF)]
struct   <- c("Year","country_num")[c("Year","country_num") %in% names(DF)]

# Targets (logs only)
all_targets <- c("ln_MIVA","ln_DCM","ln_GFCFM","ln_MFDI")
all_targets <- all_targets[all_targets %in% names(DF)]

# Robust DiD helper
fit_did <- function(dat, y){
  f <- as.formula(paste0(y, " ~ Post + T + DID"))
  m <- lm(f, data = dat)
  vc <- sandwich::vcovHC(m, type = "HC1")
  co <- coef(m)["DID"]
  se <- sqrt(diag(vc))["DID"]
  t  <- co / se
  p  <- 2*(1 - pnorm(abs(t)))
  tibble::tibble(Target=y, Method="Traditional DiD (log target)",
                 Coefficient=co, Std_Error=se,
                 P_value=p, CI_Lower=co-1.96*se, CI_Upper=co+1.96*se)
}

# DML helper (PLR) with a given pair of learners
fit_dml <- function(dat, y, ml_l, ml_m, label){
  dt <- data.table::as.data.table(dat)
  dml_data <- DoubleML::DoubleMLData$new(dt, y_col = y, d_cols = "DID", x_cols = setdiff(names(dat), c(y,"DID","Post","T")))
  plr <- DoubleML::DoubleMLPLR$new(data = dml_data, ml_l = ml_l, ml_m = ml_m, n_folds = 5, score = "partialling out")
  plr$fit()
  ci <- plr$confint(level = 0.95)
  tibble::tibble(Target=y, Method=label,
                 Coefficient=as.numeric(plr$coef), Std_Error=as.numeric(plr$se),
                 P_value=as.numeric(plr$pval), CI_Lower=as.numeric(ci[1,"2.5 %"]), CI_Upper=as.numeric(ci[1,"97.5 %"]))
}

rows <- list()
for (y in all_targets){
  # Exclude the outcome's own log from X
  x_vars <- c(setdiff(base_logs, y), struct)
  cols <- unique(c(y, "DID","Post","T", x_vars))
  dat <- DF[, cols]
  dat <- dat[stats::complete.cases(dat), , drop=FALSE]
  if (nrow(dat) == 0) next
  
  # Classical DiD
  rows[[length(rows)+1]] <- fit_did(dat, y)
  
  # Learners
  lrf_y <- mlr3::lrn("regr.ranger", num.trees=500)
  lrf_m <- mlr3::lrn("classif.ranger", num.trees=500, predict_type="prob")
  lla_y <- mlr3::lrn("regr.cv_glmnet", alpha=1)
  lla_m <- mlr3::lrn("classif.cv_glmnet", alpha=1, predict_type="prob")
  lri_y <- mlr3::lrn("regr.cv_glmnet", alpha=0)
  lri_m <- mlr3::lrn("classif.cv_glmnet", alpha=0, predict_type="prob")
  
  # DML estimates
  dat_dml <- dat[, c(y, "DID", x_vars)]
  # RF
  rows[[length(rows)+1]] <- fit_dml(dat_dml, y, lrf_y, lrf_m, "DML (Random Forest)")
  # Lasso
  rows[[length(rows)+1]] <- fit_dml(dat_dml, y, lla_y, lla_m, "DML (Lasso)")
  # Ridge
  rows[[length(rows)+1]] <- fit_dml(dat_dml, y, lri_y, lri_m, "DML (Ridge)")
}

results <- dplyr::bind_rows(rows)
readr::write_csv(results, "did_dml_fixed_results.csv")
print(results)
