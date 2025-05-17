#-------------- Load necessary packages --------------
#devtools::install_github("kthohr/BMR")   
library(quantmod)
library(dplyr)
library(tidyr)
library(lubridate)
library(BMR)
library(BVAR)
library(Metrics)
library(ggplot2)
library(torch)

#-------------- Set date range --------------
start_date <- as.Date("2021-01-01")
end_date <- Sys.Date() - 1

#-------------- Download data --------------
getSymbols(c("EURUSD=X", "^VIX", "CL=F"), from = start_date, to = end_date)

#-------------- Extract closing prices --------------
eurusd <- Cl(`EURUSD=X`)
vix    <- Cl(`VIX`)
oil    <- Cl(`CL=F`)

#-------------- Merge & clean data --------------
combined_data <- merge(eurusd, vix, oil)
colnames(combined_data) <- c("EURUSD", "VIX", "Oil")

df <- data.frame(Date = index(combined_data), coredata(combined_data)) %>%
  na.omit()

df[,-1] <- as.matrix(scale(df[,-1]))

gtsplot(df[, 2:4], dates = df[, 1])

#-------------- Prepare scaled input features for BNN --------------
df_clean <- df[, 2:4]
X <- as.matrix((df_clean))

# For each Hyperparameter, apply values obtained through the Grid Search
y_hp <- matrix(c(0.1, 0.1, 1, 0.1), nrow = 1)
lag_length <- 10

#-------------- Convert to torch tensors --------------
x_train <- torch_tensor(X, dtype = torch_float())
y_train <- torch_tensor(y_hp, dtype = torch_float())

#-------------- Define BNN model --------------
net <- nn_module(
  initialize = function() {
    self$fc1 <- nn_linear(ncol(X), 5000)
    self$fc2 <- nn_linear(5000, 4)
  },
  forward = function(x) {
    x %>% self$fc1() %>% nnf_relu() %>% self$fc2()
  }
)

model <- net()
optimizer <- optim_adam(model$parameters, lr = 0.3)

#-------------- Train the model --------------
for (epoch in 1:6000) {
  optimizer$zero_grad()
  output <- model(x_train)
  loss <- nnf_mse_loss(output, y_train)
  loss$backward()
  optimizer$step()
  if (epoch %% 1000 == 0) cat("Epoch:", epoch, "Loss:", loss$item(), "\n")
}

#-------------- Predict hyperparameters from BNN --------------
predicted_priors <- as.numeric(model(x_train)$to(device = "cpu")[nrow(df_clean),1:4])
predicted_priors[3] <- abs(predicted_priors[3])  # Ensure positive scale
names(predicted_priors) <- c("HP1", "HP2", "HP3", "HP4")
print(predicted_priors)

#-------------- Split train/test data --------------
train_data <- df %>% filter(year(Date) <= 2024)
test_data  <- df %>% filter(year(Date) >= 2025)

#-------------- BVAR Setup --------------
bvar_obj <- new(bvarm)


if (nrow(train_data) <= lag_length * ncol(train_data[, 2:4])) {
  stop("Not enough data for specified lag length.")
}

bvar_obj$build(data.matrix(train_data[, 2:4]), TRUE, p = lag_length)

#-------------- Set prior (Minnesota / NIW --------------
prior <- rep(1, 3)  # match number of variables

bvar_obj$prior(
  prior,
  var_type = 2,
  decay_type = 2,
  abs(predicted_priors[1]),
  abs(predicted_priors[2]),
  abs(predicted_priors[3]),
  abs(predicted_priors[4])
)

#-------------- Gibbs sampling --------------
bvar_obj$gibbs(10000)

str(bvar_obj$beta_draws)
#-------------- Forecast --------------
forecast_horizon <- nrow(test_data)
bvar_forecast <- forecast(bvar_obj,
                          shocks = TRUE,
                          var_names = colnames(train_data)[-1],
                          back_data = nrow(train_data) - forecast_horizon,
                          period = forecast_horizon,
                          save = FALSE)

#-------------- Extract forecast and evaluate --------------
forecast_means <- data.frame(bvar_forecast$forecast_mean)
colnames(forecast_means) <- c("EURUSD", "VIX", "Oil")
forecast_means$Date <- test_data$Date
actuals <- test_data[, 2:4]


#-------------- Evaluate forecast --------------
metrics <- data.frame(
  Variable = colnames(actuals),
  RMSE = sapply(1:3, function(i) rmse(actuals[[i]], forecast_means[[i]])),
  MAE = sapply(1:3, function(i) mae(actuals[[i]], forecast_means[[i]]))
)

print("Forecast Evaluation Metrics (2025):")
print(metrics)


#-------------- Final tidy plot --------------
forecast_plot <- cbind(Date = test_data$Date,
                       actuals,
                       Forecast_EURUSD = forecast_means$EURUSD,
                       Forecast_VIX = forecast_means$VIX,
                       Forecast_Oil = forecast_means$Oil)

forecast_plot_long <- forecast_plot %>%
  pivot_longer(cols = -Date, names_to = "Series", values_to = "Value") %>%
  mutate(Type = ifelse(grepl("Forecast", Series), "Forecast", "Actual"),
         Variable = gsub("Forecast_", "", Series),
         Variable = gsub(".*_", "", Variable))

plot_obj <- ggplot(forecast_plot_long, aes(x = Date, y = Value, color = Type)) +
  geom_line(size = 0.5) +
  facet_wrap(~Variable, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c("Actual" = "darkblue", "Forecast" = "red")) +
  theme_minimal(base_size = 14) +
  theme(
    #legend.position = "none",  
    panel.grid = element_blank(),                  
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),  # fixed line width
    panel.background = element_blank(),
    plot.background = element_blank(),
    legend.title = element_blank(),
    axis.title = element_blank(),
    plot.title = element_blank()
  )




#-------------------- 1) Gibbs plots Beta Coefficients --------------------


png("trace_acf_density_plots_organized.png", width = 1200, height = 1800)

# Set plotting layout
par(mfrow = c(10, 6), mar = c(3, 3, 2, 1), oma = c(3, 3, 5, 1))

# Labels for equations (optional for clarity)
equation_names <- c("EURUSD", "VIX", "Oil")

# Loop over 3 betas (rows), and for each equation (columns)
for (i in 1:10) {
  for (eq in 1:3) {
    beta_draw <- bvar_obj$beta_draws[i, eq, ]
    
    # Trace plot
    plot(beta_draw, type = "l",
         main = paste0("Trace: β", i, " (", equation_names[eq], ")"),
         ylab = "Value", xlab = "Iteration", col = "blue")
    
    # Density + histogram plot
    plot(density(beta_draw),
         main = paste0("Density: β", i, " (", equation_names[eq], ")"),
         xlab = "Value", col = "red", lwd = 2)
    hist(beta_draw, breaks = 30, freq = FALSE,
         add = TRUE, col = NA, border = "black")
    
  }
}

dev.off()


png("trace_acf_density_plots_organized_2.png", width = 1200, height = 1800)

# Set plotting layout
par(mfrow = c(10, 6), mar = c(3, 3, 2, 1), oma = c(3, 3, 5, 1))

# Labels for equations (optional for clarity)
equation_names <- c("EURUSD", "VIX", "Oil")

# Loop over 3 betas (rows), and for each equation (columns)
for (i in 11:20) {
  for (eq in 1:3) {
    beta_draw <- bvar_obj$beta_draws[i, eq, ]
    
    # Trace plot
    plot(beta_draw, type = "l",
         main = paste0("Trace: β", i, " (", equation_names[eq], ")"),
         ylab = "Value", xlab = "Iteration", col = "blue")
    
    # Density + histogram plot
    plot(density(beta_draw),
         main = paste0("Density: β", i, " (", equation_names[eq], ")"),
         xlab = "Value", col = "red", lwd = 2)
    hist(beta_draw, breaks = 30, freq = FALSE,
         add = TRUE, col = NA, border = "black")
    
  }
}

dev.off()


png("trace_acf_density_plots_organized_3.png", width = 1200, height = 1800)

# Set plotting layout
par(mfrow = c(11, 6), mar = c(3, 3, 2, 1), oma = c(3, 3, 5, 1))

# Labels for equations (optional for clarity)
equation_names <- c("EURUSD", "VIX", "Oil")

# Loop over 3 betas (rows), and for each equation (columns)
for (i in 21:31) {
  for (eq in 1:3) {
    beta_draw <- bvar_obj$beta_draws[i, eq, ]
    
    # Trace plot
    plot(beta_draw, type = "l",
         main = paste0("Trace: β", i, " (", equation_names[eq], ")"),
         ylab = "Value", xlab = "Iteration", col = "blue")
    
    # Density + histogram plot 
    plot(density(beta_draw),
         main = paste0("Density: β", i, " (", equation_names[eq], ")"),
         xlab = "Value", col = "red", lwd = 2)
    hist(beta_draw, breaks = 30, freq = FALSE,
         add = TRUE, col = NA, border = "black")
    
  }
}

dev.off()

#----------------------------------------- 2) IRF------------------------

# Set your real variable and shock names here
variable_names <- c("EUR/USD", "Oil", "VIX")  # Adjust as needed
shock_names <- c("EUR/USD Shock", "Oil Shock", "VIX Shock")

horizon <- forecast_horizon

# Compute Impulse Response Functions (IRFs)
irf_results <- bvar_obj$IRF(horizon)

# Extract IRF draws
irf_array_full <- irf_results$irf_vals  # [responses, shocks, draws]

# Define parameters
n_vars <- dim(irf_array_full)[1]
n_shocks <- dim(irf_array_full)[2]
n_draws <- dim(irf_array_full)[3]
draws_per_horizon <- n_draws / horizon

# Reshape to [responses, shocks, horizon, draws]
irf_array_reshaped <- array(irf_array_full,
                            dim = c(n_vars, n_shocks, horizon, draws_per_horizon))

# Compute mean IRFs across draws → [responses, shocks, horizon]
irf_means <- apply(irf_array_reshaped, c(1, 2, 3), mean)

# Load required packages
library(ggplot2)
library(reshape2)
library(dplyr)

# Compute quantiles and organize into a dataframe
irf_df_list <- list()

for (i in 1:n_vars) {
  for (j in 1:n_shocks) {
    for (h in 1:horizon) {
      draws <- irf_array_reshaped[i, j, h, ]
      irf_df_list[[length(irf_df_list) + 1]] <- data.frame(
        Response = variable_names[i],  # <-- Use real variable names
        Shock = shock_names[j],        # <-- Use real shock names
        Horizon = h,
        Mean = mean(draws),
        Lower = quantile(draws, 0.16),
        Upper = quantile(draws, 0.84)
      )
    }
  }
}

# Combine into one data frame
irf_plot_df <- do.call(rbind, irf_df_list)

# Plot using ggplot2
IRF <- ggplot(irf_plot_df, aes(x = Horizon, y = Mean)) +
  geom_line(color = "black", size = 0.5) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "gray", alpha = 0.3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray30") +
  facet_grid(Response ~ Shock, scales = "free_y") +
  labs(title = "Impulse Response Functions (with 68% CI)",
       x = "Horizon", y = "IRF") +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 0.5)
  )

# Save the plot
ggsave("IRF.png", IRF, width = 10, height = 6, dpi = 300)

