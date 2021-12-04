
dev.off()
rm(list =ls())
# install.packages("reticulate")
# install.packages("tsibbledata")
# install.packages("forecastML")
# install.packages("ranger")

options(warn = -1)
options(repr.plot.width = 12, repr.plot.height = 6)
# remotes::install_github("nredell/forecastML")
library(forecastML)

library(reticulate)  # For working with Python.
library(tsibbledata)  # Source of the dataset.
library(lubridate)
library(ggplot2)
library(forecastML)  # v0.9.1

library(ranger)  # Random Forest model in R.
library(future)  # Parallel model training.

vic_elec <- read.csv(file.choose() ,header = T,sep = ",")
data <- as.data.frame(vic_elec)

nrow(data)
head(data)

frequency <- "1 min"# A string that works in base::seq(..., by = 'frequency').

records_per_day <-50

data$Time <- as.POSIXct(data$Time, tz = "UTC")  # Our date index.


data$hour <- lubridate::hour(data$Time)
data$weekday <- factor(lubridate::wday(data$Time))  # Factor for modeling purposes.
data$month <- factor(lubridate::month(data$Time))  # Factor for modeling purposes.
data$year <- lubridate::year(data$Time)

train_indices <- 1:(nrow(data) - records_per_day * 3)

train <- data[train_indices, ]
test <- data[-train_indices, ]

data$dataset <- "Test"
data$dataset[train_indices] <- "Train"
data$dataset <- factor(data$dataset, levels = c("Train", "Test"), ordered = TRUE)

p <- ggplot(data, aes(Time,Voltage))
p <- p + geom_line(alpha = .5)
p <- p + facet_wrap(~ dataset, scales = "free_x")
p <- p + theme_bw() + theme(
  strip.text = element_text(size = 14, face = "bold"),
  axis.title = element_text(size = 14, face = "bold"),
  axis.text = element_text(size = 14)
)
p+theme(
  # Hide panel borders and remove grid lines
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),text = element_text(size =15))

outcome_col <- 1

horizons <- c(1, 7, 30)
lookback <- c(seq(50, records_per_day * 1, 20), records_per_day * 6, records_per_day * 4)
# lookback <- c(seq(24, records_per_day * 1, 50000), records_per_day * 0.5, records_per_day * 1)

dates <- train$Time

train$Time <- NULL
train$Date <- NULL

# Features that change through time but which will not be lagged.
dynamic_features <- c("Current","hour", "weekday", "month", "year")

data_train <- forecastML::create_lagged_df(train, type = "train", outcome_col = outcome_col,
                                           lookback = lookback, horizon = horizons,
                                           dates = dates, frequency = frequency,
                                           dynamic_features = dynamic_features
)

head(data_train$horizon_1)  # View the horizon-48 dataset


windows <- forecastML::create_windows(data_train,
                                      window_length = 0,  # 0 spans from window_start to window_stop.
                                      # window_start = as.POSIXct("2015-11-13 5 12:45:06", "%Y-%m-%d %H:%M:%S", tz = "UTC"),
                                      # window_stop = as.POSIXct("2015-11-13 13:03:55", "%Y-%m-%d %H:%M:%S",tz = "UTC")
                                      # # as.POSIXct("3-20-2017 08:03:27", tz = "UTC")

)

plot(windows, data_train, show_labels = TRUE,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank())+theme(
       # Hide panel borders and remove grid lines
       panel.grid.major = element_blank(),
       panel.grid.minor = element_blank(),text = element_text(size =15),
       panel.background = element_rect(color = "white"))


r_model_fun <- function(data) {

  outcome_names <- names(data)[1]
  model_formula <- formula(paste0(outcome_names,  "~ ."))

  model <- ranger::ranger(model_formula, data = data, num.trees = 300)
  return(model)
}

model_results_r <- forecastML::train_model(data_train,
                                           windows = windows,
                                           model_name = "R Random Forest",  # Can be any name.
                                           model_function = r_model_fun,
                                           use_future = TRUE  # Parallel training.
)

r_predict_fun <- function(model, data) {

  data_pred <- data.frame("y_pred" = predict(model, data)$predictions)
  return(data_pred)
}

data_valid <- predict( model_results_r,
                       prediction_function = list(r_predict_fun),
                       data = data_train)

# 1st plot.
plot(data_valid,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank(),lwd = 2)+theme(
       # Hide panel borders and remove grid lines
       panel.grid.major = element_blank(),
       panel.grid.minor = element_blank(),text = element_text(size =15),
       panel.background = element_rect(color = "white"))

view_dates <- seq(as.POSIXct("2015-11-13 09:58:41", tz = "UTC"),
                  as.POSIXct("2015-11-13 13:23:45", tz = "UTC"),
                  by = "1 min")

# 2nd plot.
plot(data_valid , valid_indices = view_dates,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank(),lwd = 2) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),text = element_text(size =15),
        panel.background = element_rect(color = "white"))+
  ggtitle("FORECAST: EVERY 1 MINUTES")

# 3rd plot.
plot(data_valid, facet = horizon ~ ., valid_indices = view_dates,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank(),lwd = 2) +theme(axis.text.x = element_text(angle = 45, hjust = 1),panel.grid.major = element_blank(),
                                                        panel.grid.minor = element_blank(),text = element_text(size =15),
                                                        panel.background = element_rect(color = "white"))+
  ggtitle(" Forecasts: Every 1 Minutes ") +
  scale_color_manual(values = c("R Random Forest" = "#276DC2"))


data_error <- forecastML::return_error(data_valid)
data_error
plot(data_error, type = "horizon", facet = ~ horizon,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank()) + theme(axis.text.x = element_text(angle = 0),panel.grid.major = element_blank(),
                                                 panel.grid.minor = element_blank(),text = element_text(size =15),
                                                 panel.background = element_rect(color = "white"))+
  scale_fill_manual(values = c("R Random Forest" = "#276DC2"))

plot(data_error, type = "global", facet = ~ horizon,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank()) + theme(axis.text.x = element_text(angle = 0),panel.grid.major = element_blank(),
                                                 panel.grid.minor = element_blank(),text = element_text(size =15),
                                                 panel.background = element_rect(color = "white")) +
  scale_fill_manual(values = c("R Random Forest" = "#276DC2"))


data_forecast <- forecastML::create_lagged_df(train, type = "forecast", outcome_col = 1,
                                              lookback = lookback, horizon = horizons,
                                              dates = dates, frequency = frequency,
                                              dynamic_features = dynamic_features)

head(data_forecast$horizon_1)  # View the horizon-144 dataset.

# Loop over the 48-, 96, and 144-step-ahead horizon datasets from create_lagged_df().
for (i in seq_along(horizons)) {

  data_forecast[[i]]$hour <- lubridate::hour(data_forecast[[i]]$index)
  data_forecast[[i]]$weekday <- factor(lubridate::wday(data_forecast[[i]]$index))
  data_forecast[[i]]$month <- factor(lubridate::month(data_forecast[[i]]$index))
  data_forecast[[i]]$year <- lubridate::year(data_forecast[[i]]$index)

  data_forecast[[i]]$Current <- with(data_forecast[[i]],
                                     ifelse(lubridate::mday(data_forecast[[i]]$index) == 31,
                                            TRUE, FALSE))
  data_forecast[[i]]$Soc <- with(data_forecast[[i]],
                                 ifelse(lubridate::mday(data_forecast[[i]]$index) == 31,
                                        TRUE, FALSE))

}

data_forecasts_r <- predict(model_results_r,
                            prediction_function = list(r_predict_fun),
                            data = data_forecast)
# data_forecasts_r
plot(data_forecasts_r, facet = ~ horizon,
     data_actual = test,
     actual_indices = test$Time,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("R Random Forest" = "#276DC2")) +
  theme(axis.text.x = element_text(angle = 0, hjust = 1),panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),text = element_text(size =10),
        panel.background = element_rect(color = "white"))



data_combined_r <- forecastML::combine_forecasts(data_forecasts_r)


plot(data_combined_r, data_actual = test, actual_indices = test$Time,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank()) +theme(axis.text.x = element_text(angle = 0, hjust = 1),panel.grid.major = element_blank(),
                                                panel.grid.minor = element_blank(),text = element_text(size =15),
                                                panel.background = element_rect(color = "white"))+

  geom_line(data = data_combined_r,
            aes(x = forecast_period, y = Voltage_pred), color = "#276DC2") +
  geom_point(data = data_combined_r,
             aes(x = forecast_period, y = Voltage_pred), color = "#276DC2") +
  facet_wrap(~ NULL) + theme(legend.position = "none")

data_error_r <- forecastML::return_error(data_combined_r, data_test = test, test_indices = test$Time)

data_plot <- dplyr::bind_rows( data_error_r$error_by_horizon)

p <- ggplot(data_plot, aes(horizon / records_per_day, mae, color = model))
p <- p + geom_line(size = 1.05)
p <- p + scale_color_manual(values = c("R Random Forest" = "#276DC2"))
p <- p + theme_bw()
p <- p + xlab("Forecast horizon ") + ylab("MAE") + labs(color = NULL) +
  ggtitle("Forecast Error on Test Data - By forecast horizon")
p+theme(axis.text.x = element_text(angle = 0, hjust = 1),panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),text = element_text(size =15),
        panel.background = element_rect(color = "white"))



data_plot <- dplyr::bind_rows(data_error_r$error_global)


p <- ggplot(data_plot, aes(model, mae, fill = model))
p <- p + geom_col()
p <- p + scale_fill_manual(values = c("R Random Forest" = "#276DC2"))

p <- p + theme_bw()
p <- p + xlab("Model") + ylab("MAE") + labs(fill = NULL) +
  ggtitle("Forecast Error on Test Data - Collapsed across forecast horizons")
p+theme(axis.text.x = element_text(angle = 0, hjust = 1),panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),text = element_text(size =15),
        panel.background = element_rect(color = "white"))


data_all <- dplyr::bind_rows(train, test)


data_all$Time <- NULL  # Still in the test dataset at this point.
data_all$Date <- NULL  # Still in the test dataset at this point.

dates <- as.POSIXct(data[, "Time", drop = TRUE])  # All dates from train + test.

data_train <- forecastML::create_lagged_df(data_all, type = "train", outcome_col = 1,
                                           lookback = lookback, horizons = horizons,
                                           dates = dates, frequency = frequency,
                                           dynamic_features = dynamic_features)


windows <- forecastML::create_windows(data_train, window_length = 0)

plot(windows, data_train,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank())+theme(axis.text.x = element_text(angle = 0, hjust = 1),panel.grid.major = element_blank(),
                                               panel.grid.minor = element_blank(),text = element_text(size =15),
                                               panel.background = element_rect(color = "white"))



model_results_r_final <- forecastML::train_model(data_train,
                                                 windows = windows,
                                                 model_name = "R Random Forest",
                                                 model_function = r_model_fun,
                                                 use_future = TRUE
)


data_historical <- predict( model_results_r_final,
                            prediction_function = list( r_predict_fun),
                            data = data_train)

plot(data_historical,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank())+theme(axis.text.x = element_text(angle = 0, hjust = 1),panel.grid.major = element_blank(),
                                               panel.grid.minor = element_blank(),text = element_text(size =15),
                                               panel.background = element_rect(color = "white"))


view_dates <- seq(as.POSIXct(min(dates), tz = "UTC"),
                  as.POSIXct(max(dates), tz = "UTC"),
                  by = "1 day")


data_forecast <- forecastML::create_lagged_df(data_all, type = "forecast", outcome_col = 1,
                                              lookback = lookback, horizon = horizons,
                                              dates = dates, frequency = frequency,
                                              dynamic_features = dynamic_features)

head(data_forecast$horizon_150)

for (i in seq_along(horizons)) {

  data_forecast[[i]]$hour <- lubridate::hour(data_forecast[[i]]$index)
  data_forecast[[i]]$weekday <- factor(lubridate::wday(data_forecast[[i]]$index))
  data_forecast[[i]]$month <- factor(lubridate::month(data_forecast[[i]]$index))
  data_forecast[[i]]$year <- lubridate::year(data_forecast[[i]]$index)

  data_forecast[[i]]$Current <- with(data_forecast[[i]],
                                     ifelse(lubridate::mday(data_forecast[[i]]$index) == 1,
                                            TRUE, FALSE))
  data_forecast[[i]]$Soc <- with(data_forecast[[i]],
                                 ifelse(lubridate::mday(data_forecast[[i]]$index) == 1,
                                        TRUE, FALSE))
}

# data_forecasts_py <- predict(model_results_py_final,
#                              prediction_function = list(r_wrapper_for_py_predict_fun),
#                              data = data_forecast)

data_forecasts_r <- predict(model_results_r_final,
                            prediction_function = list(r_predict_fun),
                            data = data_forecast)


plot(data_forecasts_r, facet = ~ horizon,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("R Random Forest" = "#276DC2"))+
  theme(axis.text.x = element_text(angle = 0, hjust = 1),panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),text = element_text(size =10),
        panel.background = element_rect(color = "white"))



# data_combined_py <- forecastML::combine_forecasts(data_forecasts_py)

data_combined_r <- forecastML::combine_forecasts(data_forecasts_r)

data_combined <- forecastML::combine_forecasts(data_forecasts_r, aggregate = stats::median)

plot(data_combined_r,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank())+
  theme(axis.text.x = element_text(angle = 0, hjust = 1),panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),text = element_text(size =15),
        panel.background = element_rect(color = "white"))



#
data_residuals_valid <- residuals(data_valid)
data_residuals_fit <- residuals(data_historical)

set.seed(224)

data_combined_r_valid <- forecastML::calculate_intervals(data_combined_r, data_residuals_valid,
                                                         levels = c(.50, .80, .95), times = 100L)

plot(data_combined_r_valid,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank())+ theme(
       # Hide panel borders and remove grid lines
       panel.grid.major = element_blank(),
       panel.grid.minor = element_blank(),text = element_text(size =15))




# Runtime: ~15 seconds.
set.seed(224)
data_combined_r_fit <- forecastML::calculate_intervals(data_combined_r, data_residuals_fit,
                                                       levels = c(.50, .80, .95), times = 100L)

plot(data_combined_r_fit,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank())+
  theme(
    # Hide panel borders and remove grid lines
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),text = element_text(size =15))


# Runtime: ~10 seconds.
set.seed(224)
data_combined_ensemble <- forecastML::calculate_intervals(data_combined, data_residuals_valid,
                                                          levels = c(.50, .80, .95), times = 300L)

plot(data_combined_ensemble,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank())+
  theme(
    # Hide panel borders and remove grid lines
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),text = element_text(size =15))



remove_dates <- -(1:(nrow(data) - records_per_day * 10))

data_actual <- data[remove_dates, ]
# p <- ggplot(data, aes(Time, Voltage))
# # p <- p + geom_line(alpha = .5)
# p <- p + facet_wrap(~ dataset, scales = "free_x")
# p <- p + theme_bw() + theme(
#   strip.text = element_text(size = 14, face = "bold"),
#   axis.title = element_text(size = 14, face = "bold"),
#   axis.text = element_text(size = 14)
# )
# p

p <- plot(data_combined_ensemble, data_actual = data_actual, actual_indices = data_actual$Time)
p <- p + theme_bw() + theme(
  plot.title = element_text(size = 16, face = "bold"),
  axis.title = element_text(size = 14, face = "bold"),
  axis.text = element_text(size = 12),
  legend.text = element_text(size = 12),
  legend.position = "bottom"
)
p <- ggplot(data, aes(Time, Voltage))
p <-p+ xlab("Date") + ylab("vOLTAGE") + labs(color = NULL) +
  ggtitle("3-Day-Ahead Forecasted PER MINUTE VOLTAGE")
p+theme(
  # Hide panel borders and remove grid lines
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),text = element_text(size =15))

ggsave(file = "3-Day-Ahead Forecasted PER MINUTE VOLTAGE.png", width = 9, height = 6)
