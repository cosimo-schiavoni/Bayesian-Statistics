#-------------- Define grid of hyperparameters -------------- 
#NB! This code needs more than 1 hour to run 1250 different hyperparameters combinations:
# 5x5x5x5 = 625(hyperparam combos) ×2 (lags) = 1250

# here are reported 5 values to be checked for each of the 4 hyperparameters 
grid_vals <- c(0.1, 1, 10, 20, 30)
# here are reported 2 values to be checked for lag horizon in BVAR model 
lag_vals <- c(10, 68)

results <- list()

forecast_horizon <- nrow(test_data)
combo_id <- 1

for (lag in lag_vals) {
  for (hp1 in grid_vals) {
    for (hp2 in grid_vals) {
      for (hp3 in grid_vals) {
        for (hp4 in grid_vals) {
          
          cat("Trying lag =", lag, "| HP1 =", hp1, "| HP2 =", hp2,
              "| HP3 =", hp3, "| HP4 =", hp4, "\n")
          
          if (nrow(train_data) <= lag * ncol(train_data[, 2:4])) {
            next  # Skip if not enough data
          }
          
          bvar_obj <- new(bvarm)
          bvar_obj$build(data.matrix(train_data[, 2:4]), TRUE, p = lag)
          
          # Prior: (coef_prior, var_type, decay_type, HP1, HP2, HP3, HP4)
          prior <- rep(1, 3)
          bvar_obj$prior(prior, 2, 2, abs(hp1), abs(hp2), abs(hp3), abs(hp4))
          
          bvar_obj$gibbs(10000)
          
          bvar_forecast <- tryCatch({
            forecast(bvar_obj,
                     shocks = TRUE,
                     var_names = colnames(train_data)[-1],
                     back_data = nrow(train_data) - forecast_horizon,
                     period = forecast_horizon,
                     save = FALSE)
          }, error = function(e) return(NULL))
          
          if (!is.null(bvar_forecast)) {
            forecast_means <- data.frame(bvar_forecast$forecast_mean)
            colnames(forecast_means) <- c("EURUSD", "VIX", "Oil")
            forecast_means$Date <- test_data$Date
            actuals <- test_data[, 2:4]
            
            rmse_vals <- sapply(1:3, function(i) rmse(actuals[[i]], forecast_means[[i]]))
            total_rmse <- sum(rmse_vals)
            
            results[[combo_id]] <- list(
              Lag = lag,
              HP1 = hp1, HP2 = hp2, HP3 = hp3, HP4 = hp4,
              RMSE_EURUSD = rmse_vals[1],
              RMSE_VIX = rmse_vals[2],
              RMSE_Oil = rmse_vals[3],
              Total_RMSE = total_rmse
            )
            combo_id <- combo_id + 1
          } else {
            cat("Skipping combo due to forecast error.\n")
          }
        }
      }
    }
  }
}

#-------------- Identify best result ===
results_df <- do.call(rbind, lapply(results, as.data.frame))
best_config <- results_df[which.min(results_df$Total_RMSE), ]

cat("\nBest Minnesota Prior Hyperparameter Configuration:\n")
print(best_config)
