# =====================================================================
# Lahore Air Quality: End-to-End Analysis (R)
# ---------------------------------------------------------------------
# Cleans raw sensor data, explores seasonal and spatial patterns in
# PM2.5, and models what drives air pollution in Lahore.
#
# Author: Faiza Jabeen
# =====================================================================

library(tidyverse)
library(lubridate)
library(randomForest)

WHO_GUIDELINE <- 15
OUT <- "outputs"


# ---------------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------------
load_and_clean <- function(path) {
  raw <- read_csv(path, show_col_types = FALSE)
  message("Raw data: ", nrow(raw), " rows")

  clean <- raw %>%
    # Station names arrive with inconsistent casing and stray whitespace
    mutate(station = str_to_title(str_trim(station))) %>%
    # -999 means the sensor failed, not that the air was clean
    mutate(pm25 = na_if(pm25, -999)) %>%
    distinct() %>%
    mutate(date = ymd(date))

  message("Removed ", nrow(raw) - nrow(clean), " duplicate rows")

  # Report missingness before deciding what to do about it
  miss <- colSums(is.na(clean))
  message("\nMissing values:")
  print(miss[miss > 0])

  clean <- clean %>%
    # PM2.5 is the target: rows without it cannot be modelled
    filter(!is.na(pm25)) %>%
    arrange(station, date) %>%
    # Weather gaps filled per-station over time, since weather is
    # autocorrelated day to day
    group_by(station) %>%
    mutate(across(
      c(humidity_pct, wind_speed_kmh, temperature_c),
      ~ zoo::na.approx(.x, na.rm = FALSE) %>% zoo::na.locf(na.rm = FALSE) %>%
        zoo::na.locf(fromLast = TRUE, na.rm = FALSE)
    )) %>%
    ungroup()

  message("Clean data: ", nrow(clean), " rows")
  clean
}


# ---------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------
add_features <- function(df) {
  df %>%
    mutate(
      month          = month(date),
      day_of_year    = yday(date),
      day_of_week    = wday(date, week_start = 1),
      is_weekend     = as.integer(day_of_week >= 6),
      # Smog season in Lahore runs roughly November to February
      is_smog_season = as.integer(month %in% c(11, 12, 1, 2)),
      # Cyclical encoding: day 365 and day 1 are neighbours, not opposites
      doy_sin        = sin(2 * pi * day_of_year / 365),
      doy_cos        = cos(2 * pi * day_of_year / 365)
    ) %>%
    arrange(station, date) %>%
    group_by(station) %>%
    mutate(
      # Yesterday's pollution is the strongest single predictor of today's
      pm25_lag1  = lag(pm25),
      pm25_roll7 = zoo::rollmean(lag(pm25), 7, fill = NA, align = "right")
    ) %>%
    ungroup() %>%
    filter(!is.na(pm25_lag1), !is.na(pm25_roll7))
}


# ---------------------------------------------------------------------
# 3. EXPLORATORY ANALYSIS
# ---------------------------------------------------------------------
explore <- function(df) {

  # --- Seasonal cycle: the headline finding ---
  monthly <- df %>%
    group_by(month) %>%
    summarise(mean_pm = mean(pm25), sd_pm = sd(pm25), .groups = "drop")

  p1 <- ggplot(monthly, aes(month, mean_pm)) +
    geom_ribbon(aes(ymin = mean_pm - sd_pm, ymax = mean_pm + sd_pm),
                fill = "#B23A48", alpha = 0.18) +
    geom_line(colour = "#B23A48", linewidth = 1.2) +
    geom_point(colour = "#B23A48", size = 2.5) +
    geom_hline(yintercept = WHO_GUIDELINE, linetype = "dashed",
               colour = "#2A9D8F", linewidth = 0.9) +
    scale_x_continuous(breaks = 1:12, labels = month.abb) +
    labs(
      title = "Lahore PM2.5 by month: smog season is a different world",
      subtitle = paste0("Dashed line = WHO 24h guideline (",
                        WHO_GUIDELINE, " µg/m³)"),
      x = NULL, y = "PM2.5 (µg/m³)"
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))

  ggsave(file.path(OUT, "R_01_seasonal_cycle.png"), p1,
         width = 10, height = 5, dpi = 150)

  # --- Which neighbourhoods bear the burden ---
  p2 <- df %>%
    mutate(station = fct_reorder(station, pm25, .fun = median)) %>%
    ggplot(aes(station, pm25, fill = station)) +
    geom_boxplot(outlier.alpha = 0.25, show.legend = FALSE) +
    geom_hline(yintercept = WHO_GUIDELINE, linetype = "dashed",
               colour = "#2A9D8F", linewidth = 0.9) +
    scale_fill_brewer(palette = "OrRd") +
    labs(
      title = "Pollution is not shared equally across Lahore",
      x = NULL, y = "PM2.5 (µg/m³)"
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))

  ggsave(file.path(OUT, "R_02_station_comparison.png"), p2,
         width = 9, height = 5, dpi = 150)

  # --- What weather does to the air ---
  p3 <- df %>%
    select(pm25, wind_speed_kmh, humidity_pct, temperature_c) %>%
    pivot_longer(-pm25, names_to = "variable", values_to = "value") %>%
    mutate(variable = recode(variable,
      wind_speed_kmh = "Wind speed (km/h)",
      humidity_pct   = "Humidity (%)",
      temperature_c  = "Temperature (°C)"
    )) %>%
    ggplot(aes(value, pm25)) +
    geom_point(alpha = 0.2, size = 0.7, colour = "#4A5859") +
    geom_smooth(method = "loess", formula = y ~ x,
                colour = "#B23A48", se = FALSE, linewidth = 1.2) +
    facet_wrap(~ variable, scales = "free_x") +
    labs(
      title = "Wind clears the air; humidity traps it",
      x = NULL, y = "PM2.5 (µg/m³)"
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))

  ggsave(file.path(OUT, "R_03_weather_relationships.png"), p3,
         width = 12, height = 4.2, dpi = 150)

  message("Figures written to ", OUT, "/")
}


# ---------------------------------------------------------------------
# 4. MODELLING
# ---------------------------------------------------------------------
# Validation respects time order. A random split would let the model see
# future days to predict past ones, inflating scores in a way that would
# never survive real deployment.
# ---------------------------------------------------------------------
time_series_cv <- function(df, features, n_splits = 5) {
  df <- df %>% arrange(date)
  n  <- nrow(df)
  fold_size <- floor(n / (n_splits + 1))

  results <- map_dfr(seq_len(n_splits), function(i) {
    train_end <- fold_size * i
    test_end  <- min(fold_size * (i + 1), n)

    train <- df[1:train_end, ]
    test  <- df[(train_end + 1):test_end, ]

    fml <- as.formula(paste("pm25 ~", paste(features, collapse = " + ")))

    lm_fit  <- lm(fml, data = train)
    lm_pred <- predict(lm_fit, test)

    rf_fit  <- randomForest(fml, data = train, ntree = 300, nodesize = 2)
    rf_pred <- predict(rf_fit, test)

    tibble(
      fold = i,
      lm_mae = mean(abs(test$pm25 - lm_pred)),
      lm_r2  = 1 - sum((test$pm25 - lm_pred)^2) /
                   sum((test$pm25 - mean(test$pm25))^2),
      rf_mae = mean(abs(test$pm25 - rf_pred)),
      rf_r2  = 1 - sum((test$pm25 - rf_pred)^2) /
                   sum((test$pm25 - mean(test$pm25))^2)
    )
  })

  summary <- tibble(
    model = c("Linear Regression", "Random Forest"),
    MAE   = c(mean(results$lm_mae), mean(results$rf_mae)),
    R2    = c(mean(results$lm_r2),  mean(results$rf_r2))
  )

  print(summary)
  write_csv(summary, file.path(OUT, "R_model_results.csv"))

  # Feature importance from a model fitted on everything
  fml <- as.formula(paste("pm25 ~", paste(features, collapse = " + ")))
  rf_full <- randomForest(fml, data = df, ntree = 300, importance = TRUE)

  imp <- importance(rf_full) %>%
    as.data.frame() %>%
    rownames_to_column("feature") %>%
    arrange(desc(IncNodePurity))

  p <- ggplot(imp, aes(reorder(feature, IncNodePurity), IncNodePurity)) +
    geom_col(fill = "#7A6C5D") +
    coord_flip() +
    labs(title = "What actually drives PM2.5?", x = NULL, y = "Importance") +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))

  ggsave(file.path(OUT, "R_04_feature_importance.png"), p,
         width = 8, height = 5, dpi = 150)

  summary
}


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
main <- function() {
  message(strrep("=", 60))
  message("LAHORE AIR QUALITY ANALYSIS")
  message(strrep("=", 60))

  df <- load_and_clean("data/lahore_air_quality_raw.csv") %>%
    add_features()

  write_csv(df, "data/lahore_air_quality_clean.csv")

  # Key numbers
  smog  <- df %>% filter(is_smog_season == 1) %>% pull(pm25) %>% mean()
  clear <- df %>% filter(is_smog_season == 0) %>% pull(pm25) %>% mean()

  message("\nMean PM2.5             : ", round(mean(df$pm25), 1), " µg/m³")
  message("Smog season (Nov-Feb)  : ", round(smog, 1), " µg/m³")
  message("Rest of year           : ", round(clear, 1), " µg/m³")
  message("Ratio                  : ", round(smog / clear, 2), "x worse")
  message("Above WHO guideline    : ",
          round(mean(df$pm25 > WHO_GUIDELINE) * 100, 1), "% of days")

  explore(df)

  features <- c("temperature_c", "humidity_pct", "wind_speed_kmh",
                "rainfall_mm", "doy_sin", "doy_cos", "is_weekend",
                "is_smog_season", "pm25_lag1", "pm25_roll7")

  time_series_cv(df, features)

  message("\nDone.")
}

main()
