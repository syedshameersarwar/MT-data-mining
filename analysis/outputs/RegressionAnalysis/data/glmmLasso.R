#! /usr/bin/Rscript
library(glmmLasso)
library(MASS)
library(nlme)
library(lme4)
library(lmerTest)

# Load the data
read_data <- function(file_name) {
  data <- read.csv(file_name)
  # convert concentration.um. to factor
  data$concentration.um. <- as.factor(data$concentration.um.)
  # convert bct_id to factor
  data$bct_id <- as.factor(data$bct_id)
  # remove first and 4th column
  data <- data[,-c(1, 4)]
  # standardize all columns except the first and 2nd still keeping as dataframe
  data[,-c(1, 2)] <- scale(data[,-c(1, 2)])
  return(data)
}

# running glmm L1 lasso with cross-validation for lambda on one feature as target
run_lasso_with_cv <- function(data, target, drug) {
  set.seed(123)
  N <- dim(data)[1]
  ind <- sample(N, N)
  lambda <- 10^seq(-3,5, length=10)
  family <- gaussian()
  
  ## set number of folds
  kk <- 5
  nk <- floor(N / kk)
  
  Devianz_ma <- matrix(Inf, ncol = kk, nrow = length(lambda))
  
  # Get predictor columns (everything except target and first two columns)
  predictor_cols <- names(data)[-(1:2)]  # Remove first two columns
  predictor_cols <-
    predictor_cols[predictor_cols != target]  # Remove target
  predictor_formula <- paste(predictor_cols, collapse = " + ")
  full_model_formula <-
    as.formula(paste(target, "~", predictor_formula))
  random_formula <- list(bct_id =  ~ 1)
  if (drug != "baseline") {
    print("Using random effect for concentration")
    random_formula <- list(bct_id =  ~ 1, concentration.um. =  ~ 1)
  }
  ## first fit good starting model
  init_model_formula <- as.formula(paste(target, "~1"))
  PQL <- glmmPQL(
    init_model_formula,
    random = random_formula,
    family = family,
    data = data
  )
  if (drug != "baseline") {
    Delta.start <-
      c(as.numeric(PQL$coef$fixed),
        rep(0, length(predictor_cols)),
        as.numeric(t(PQL$coef$random$bct_id)),
        as.numeric(t(PQL$coef$random$concentration.um.)))
    Q.start <-matrix(c(as.numeric(VarCorr(PQL)[2, 1]), 0.5, 0.5, as.numeric(VarCorr(PQL)[4, 1])), nrow=2, ncol=2, byrow = TRUE)
  }
  else {
    Delta.start <-
      c(as.numeric(PQL$coef$fixed),
        rep(0, length(predictor_cols)),
        as.numeric(t(PQL$coef$random$bct_id)))
    Q.start <-matrix(c(as.numeric(VarCorr(PQL)[1, 1])), nrow=1, ncol=1, byrow = TRUE)
  }
  # Delta.start <-
  #   c(as.numeric(PQL$coef$fixed),
  #     rep(0, length(predictor_cols)),
  #     as.numeric(t(PQL$coef$random$bct_id)),
  #     as.numeric(t(PQL$coef$random$concentration.um.)))
  # Q.start <-matrix(c(as.numeric(VarCorr(PQL)[2, 1]), 0.5, 0.5, as.numeric(VarCorr(PQL)[4, 1])), nrow=2, ncol=2, byrow = TRUE)
  
  
  for (j in 1:length(lambda))
  {
    print(paste("Iteration ", j, sep = ""))
    
    for (i in 1:kk)
    {
      if (i < kk)
      {
        indi <- ind[(i - 1) * nk + (1:nk)]
      } else{
        indi <- ind[((i - 1) * nk + 1):N]
      }
      
      data.train <- data[-indi, ]
      data.test <- data[indi, ]
      
      
      glm2 <-
        try(glmmLasso(
          full_model_formula,
          rnd = random_formula,
          family = gaussian(link = "identity"),
          data = data,
          lambda = lambda[j],
          switch.NR = TRUE,
          final.re = TRUE,
          control = list(start = Delta.start, q_start =
                           Q.start)
        )
        ,
        silent = TRUE)
      
      if (!inherits(glm2, "try-error"))
      {
        y.hat <- predict(glm2, data.test)
        
        Devianz_ma[j, i] <-
          sum(family$dev.resids(data.test[[target]], y.hat, wt = rep(1, length(y.hat))))
      }
    }
    print(sum(Devianz_ma[j, ]))
  }
  
  Devianz_vec <- apply(Devianz_ma, 1, sum)
  opt2 <- which.min(Devianz_vec)
  
  
  glm2_final <-
    glmmLasso(
      full_model_formula,
      rnd = random_formula,
      family = family,
      data = data,
      lambda = lambda[opt2],
      switch.NR = TRUE,
      final.re = TRUE,
      control = list(start = Delta.start, q_start =
                       Q.start)
    )
  
  # select non-zero coefficients and run lme4 model on them with same random effects
  # non_zero_coef <- glm2_final$coefficients[glm2_final$coefficients != 0]
  # non_zero_coef_names <- names(non_zero_coef)
  # non_zero_coef_names <- non_zero_coef_names[non_zero_coef_names != "(Intercept)"]
  # non_zero_coef_names <- paste(non_zero_coef_names, collapse = " + ")
  # non_zero_coef_formula <- as.formula(paste(target, "~", non_zero_coef_names, "+ (1|bct_id) + (1|concentration.um.)"))
  # glm2_final_lme <- lmer(non_zero_coef_formula, data = data, REML = FALSE)
  
  print("Optimal lambda: ")
  print(lambda[opt2])
  print("Deviance: ")
  print(sum(Devianz_ma[opt2, ]))
  # summary(glm2_final_lme)
  # return(glm2_final_lme)
  glm2_final$optimal_lambda <- lambda[opt2]
  summary(glm2_final)
  return(glm2_final)
}

# Run the lasso regression for each feature as target and store the results in d*d dataframe with coefficeints, for itself the coefficeint is 1
run_lasso_for_all_features <- function(data, drug) {
  features <- names(data)[-(1:2)]
  results <- matrix(NA, nrow = length(features), ncol = length(features))
  rownames(results) <- features
  colnames(results) <- features
  for (feature in features) {
    print(paste("Running lasso for feature: ", feature, sep = ""))
    result <- run_lasso_with_cv(data, feature, drug)
    # remove the intercept and store the coefficients
    results[feature, names(result$coefficients)[-1]] <- result$coefficients[-1]
    # set the coefficient for the feature itself to 1
    results[feature, feature] <- result$optimal_lambda
  }
  return(results)
}

# baseline
drug <- "baseline"
# Load the data
data <-
  read_data(
    paste(drug,".csv", sep = "")
  )
weight_matrix <- run_lasso_for_all_features(data, drug)
write.csv(weight_matrix, paste(drug,"_weight_matrix.csv", sep = ""), row.names = TRUE)
# nifedipine
drug <- "nifedipine"
# Load the data
data <-
  read_data(
    paste(drug,".csv", sep = "")
  )
weight_matrix <- run_lasso_for_all_features(data, drug)
# Save the weight matrix

write.csv(weight_matrix, paste(drug,"_weight_matrix.csv", sep = ""), row.names = TRUE)


# e4031
drug <- "e4031"
# Load the data
data <-
  read_data(
    paste(drug,".csv", sep = "")
  )
weight_matrix <- run_lasso_for_all_features(data, drug)
# Save the weight matrix

write.csv(weight_matrix, paste(drug,"_weight_matrix.csv", sep = ""), row.names = TRUE)

# ca_titration
drug <- "ca_titration"
# Load the data
data <-
  read_data(
    paste(drug,".csv", sep = "")
  )
weight_matrix <- run_lasso_for_all_features(data, drug)
write.csv(weight_matrix, paste(drug,"_weight_matrix.csv", sep = ""), row.names = TRUE)