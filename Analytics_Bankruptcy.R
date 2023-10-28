# Load necessary libraries and define functions
rm(list=ls()) 
setwd("D:/Study/EP-ABA_BL05/Forecasting by A.K. Laha/Bankruptcy_Problem")
load_libraries <- function() {
  if (!require("car")) install.packages("car", dependencies = TRUE)
  if (!require("dplyr")) install.packages("dplyr", dependencies = TRUE)
  if (!require("caret")) install.packages("caret", dependencies = TRUE)
  if (!require("ROCR")) install.packages("ROCR", dependencies = TRUE)
  
  library(car)
  library(dplyr)
  library(caret)
  library(ROCR)
}

# Load the dataset
load_file <- function(file_name) {
  dataset <- read.csv(file_name, header = TRUE)
  return(dataset)
}

# Stratified split for training and test datasets
stratified_split <- function(perct_train,data_set) {
  # Define a seed variable
  my_seed <- 456
  
  # Set the seed using the variable
  set.seed(my_seed)
  indices_train <- createDataPartition(data_set$D, p = perct_train, list = FALSE, times = 1)
  data_test <- data_set[-indices_train, ]
  data_train <- data_set[indices_train, ]
  
  return(list(data_train, data_test))
}

# Feature elimination based on VIF values
feature_elimination <- function(data_set, vif_threshold) {
  while (any(vif_values > vif_threshold)) {
    vif_values <- car::vif(lm(D ~ ., data = data_set))
    
    if (all(vif_values <= vif_threshold)) {
      cat("All remaining features meet the VIF threshold (VIF <= ", vif_threshold, ")\n")
      break
    }
    
    max_vif_feature <- names(vif_values)[which.max(vif_values)]
    data_set <- data_set[, !(colnames(data_set) %in% max_vif_feature)]
    
    cat("\nRemoved feature:", max_vif_feature, "with VIF =", max(vif_values), '\n')
    cat("Current VIF values:", vif_values, '\n')
  }
  
  return(data_set)
}


# Build a logistic regression model
build_logistic_model <- function(train_data) {
  model <- glm(D ~ ., data = train_data, family = "binomial")
  return(model)
}
model_fit <- function(df) {
  p_vals <- summary(glm(D ~ ., data = df, family = binomial))$coefficients[, "Pr(>|z|)"]
  return(p_vals)
}

removal_high_p <- function(data_set, sigf_level) {
  predictor_len <- length(predictors)
  model <- model_fit(data_set)
  max_p_value <- model[which.max(model)]
  
  for (i in 1:predictor_len) {
    model <- model_fit(data_set)
    max_p_value <- model[which.max(model)]
    removed_predictor <- names(model)[which.max(model)]
    
    if (sigf_level < max_p_value) {
      data_set <- data_set[, -which(names(data_set) == removed_predictor)]
      cat("\nRemoved predictor:", removed_predictor, "with p-value =", max_p_value, "\n")
    } else {
      break
    }
  }
  return(data_set)
}

# Generate a classification report
generate_classification_report <- function(model, df, threshold, label) {

  cnf_matrx <- confusionMatrix(data = as.factor(ifelse(predict(model, newdata = df, type = "response") >= threshold, 1, 0)), reference = as.factor(df$D), positive = '1')
  print(cnf_matrx)
 
}

auc_curve <- function(model, data_frame) {
  # Predict probabilities and create a ROC curve 
  predcn_obj <- prediction(predict(model, newdata = data_frame, type = "response"), as.factor(data_frame$D))
  
  # Calculate and display the AUC
  auc_value <- round(unlist(slot(performance(predcn_obj, "auc"), "y.values")), 2)
  
  # Plot the ROC curve
  plot(performance(predcn_obj, "tpr", "fpr"), main = "ROC Curve", col = "green")
  abline(a = 0, b = 1, lty = 2, col = "blue")
  text(0.5, 0.3, paste("AUC =", auc_value), col = "red")
}


# Load necessary libraries
load_libraries()
if (!length(grep("Bankruptcy_Problem", getwd()))) {
  # Set the working directory to "./Bankruptcy_Problem"
  setwd("./Bankruptcy_Problem")
}
getwd()
# Load the dataset
bankruptcy <- load_file('Bankruptcy.csv')

# Display dataset size
message("Dataset Size: ", dim(bankruptcy), "\n")

# Display the first few rows of the dataset
head(bankruptcy)

# Load the dplyr library
library(dplyr)

# Define the columns to be removed
columns_to_remove <- c("NO", "YR")  # Add more column names as needed

# Remove the specified columns from the dataset
bankruptcy <- bankruptcy %>%
  select(-one_of(columns_to_remove))

head(bankruptcy)
# Calculate initial Variance Inflation Factor (VIF) values
vif_values <- car::vif(lm(D ~ ., data = bankruptcy))
cat("Initial VIF values:\n")
print(sort(vif_values, decreasing=TRUE))

# Set a threshold for VIF
vif_threshold <- 4

# Feature elimination based on VIF values
bankruptcy <- feature_elimination(bankruptcy, vif_threshold)

# Display the final set of selected features
cat("\nFinal feature set after feature elimination: ")
cat(colnames(bankruptcy))

# Calculate final VIF values
vif_values <- car::vif(lm(D ~ ., data = bankruptcy))
cat("Final VIF values:\n")
print(sort(vif_values, decreasing = TRUE))

# Set the train_percentage
percent_train <- 0.75

# Stratified split for training and test datasets
split_data <- stratified_split(percent_train,bankruptcy)
train_data <- split_data[[1]]
test_data <- split_data[[2]]

# Display the first few rows of the training dataset
head(train_data)

# Display the first few rows of the test dataset
head(test_data)

# Display label counts and proportions in training and test datasets
train_label_counts <- table(train_data$D)
cat('Label counts in training data:', train_label_counts, '\n')
cat('Label proportion in training data:', train_label_counts / nrow(train_data), '\n')

test_label_counts <- table(test_data$D)
cat('Label counts in test data:', test_label_counts, '\n')
cat('Label proportion in test data:', test_label_counts / nrow(test_data), '\n')

# Build the base logistic regression model
base_model <- build_logistic_model(train_data)

# Display summary of the base model
summary(base_model)


# Eliminate features with high p-values
train_data <- removal_high_p(train_data, 0.05) #significance_level taken 0.05

# Get the final list of selected predictors
final_predictors <- names(train_data)[-which(names(train_data) == "D")]

cat("\nFinal list of selected predictors:", final_predictors)

# Build the final logistic regression model
final_model <- build_logistic_model(train_data)

# Display summary of the final model
summary(final_model)

# Display classification report on training data
generate_classification_report(final_model, train_data, 0.5, "Training Dataset")

# Select relevant columns from the test dataset
test_data <- test_data %>% select(all_of(c("D", final_predictors)))
                                  
 # Display classification report on the test data
generate_classification_report(final_model, test_data, 0.5, "Test Dataset")
                                  
# Plot the ROC-AUC curve
auc_curve(final_model, train_data)
                                  
