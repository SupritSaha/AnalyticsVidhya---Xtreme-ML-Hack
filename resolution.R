# Analytics Vidhya Xtreme ML Hack
# Author - Suprit Saha

# Loading required packages
# ==============================================================
library(data.table)
library(forecast)
library(plyr)
library(lubridate)
library(ggplot2)

# Reading training and testing data
# =============================================================
dt_train <- fread("Train/Resolution_Pre_2017.csv",header = TRUE,na.strings = "")
dt_test <- fread("Test/Resolution2017.csv",header = TRUE,na.strings = "")

# Changing data type
# =============================================================
dt_train[,Date := as.Date(Date)]
dt_test[,Date := as.Date(Date)]


# 2557 weeks of data
# Need to be rolled for daily level forecasting
# Missing weeks for category/subject combination existed. These would be problematic for time series forecasting and needs to be treated

# Aggregating data at daily level
# ==============================================================
dt_train <- dt_train[,.(Resolution = sum(Resolution,na.rm = TRUE)),by = c("Date","Category","Subject")]

# Adding missing weeks
# ==============================================================
category_type <- unique(dt_train[['Category']])
subject_type <- unique(dt_train[['Subject']])

dt_train_full <- data.table(NULL) # Initialising data table
for(type in category_type)
{
  for(sub in subject_type)
  {
    dt_type <- subset(dt_train,Category == type & Subject == sub)
    if(nrow(dt_type) == 0)
    {
      next
    }
    min_date <- min(dt_type$Date)
    max_date <- max(dt_train$Date)
    dt_temp <- data.table(Date = seq(min_date,max_date,by = 1),Category = type,Subject = sub)
    dt_temp <- join(x = dt_temp,y = dt_type,type = "left", by = c("Date","Category","Subject"))
    dt_train_full <- rbindlist(list(dt_train_full,dt_temp))
  }
}

rm(dt_train,dt_type)

# Feature Engineering
# ==============================================================
dt_train_full[,Year := year(Date)]
dt_train_full[,Month := as.factor(month(Date))]
dt_train_full[,DayOfWeek := as.factor(wday(Date))]
dt_train_full[,Week := as.factor(week(Date))]

dt_test[,Year := year(Date)]
dt_test[,Month := as.factor(month(Date))]
dt_test[,DayOfWeek := as.factor(wday(Date))]
dt_test[,Week := as.factor(week(Date))]

# Yearly variation
# ===============================================================
year_variation <- dt_train_full[,.(yearly_avg_resolution = round(mean(Resolution,na.rm = TRUE))), by = c("Year","Category","Subject")]
year_variation <- year_variation[order(Category,Subject,Year),]
ggplot(data = year_variation,aes(x = Year,y = yearly_avg_resolution)) + geom_line() + facet_wrap(~Category+Subject,scales = "free")

# Most category/subject have vastly different patterns across the years
# Yearly variation  exists
# Resolutions are generally low volume

# Enough to consider last two years: 2015 and 2016 for our analysis
# ========================================================================
dt_train_2015 <- subset(dt_train_full, Year >= 2015)
rm(dt_train_full)

# Missing value imputation
# ========================================================================
dt_train_2015$Resolution[is.na(dt_train_2015$Resolution)] <- 0

# Monthly variation
# ========================================================================
monthly_variation <- dt_train_2015[,.(monthly_avg_resolution = round(mean(Resolution,na.rm = TRUE))), by = c("Month","Category","Subject")]
monthly_variation <- monthly_variation[order(Category,Subject,Month),]
ggplot(data = monthly_variation,aes(x = as.integer(Month),y = monthly_avg_resolution)) + geom_line() + facet_wrap(~Category + Subject,scales = "free") + scale_x_continuous(breaks = 1:12) + ggtitle("Monthly variation in Resolution") 


# Definite monthly variation
# Inverted cone shape in months of July,August and September for some category/subject
# Some series are correlated

# Day of Week variation
# ===========================================================================
dow_variation <- dt_train_2015[,.(dow_avg_resolution = round(mean(Resolution,na.rm = TRUE))), by = c("DayOfWeek","Category","Subject")]
dow_variation <- dow_variation[order(Category,Subject,DayOfWeek),]
ggplot(data = dow_variation,aes(x = as.integer(DayOfWeek),y = dow_avg_resolution)) + geom_line() + facet_wrap(~Category + Subject,scales = "free") + scale_x_continuous(breaks = 1:7) + ggtitle("DOW variation in resolution") 

# Day of the Week variation exists
# Weekends have low/zero resolution
# Resolutions generally taper off during the week

# Summary
# -----------------------------------------------------------------
# No use of working with full history
# Yearly,monthly and day of the week variation exists
# Most of the resolutions are low volume

# Forecasting
# ==============================================================================
tbatsForecast <- function(x,horizon = 74,seasonality = c(7,30,365.25))
{
  x <- msts(x,seasonal.periods = seasonality)
  fit <- tbats(x)
  fcst <- round(forecast(fit,h = horizon)$mean)
  return(fcst)
}

arimaRegForecast <- function(x,horizon = 74,frequency = 7,xreg = NULL,xreg_test = NULL)
{
  x <- ts(x,frequency = 7)
  fit <- auto.arima(x,xreg = xreg)
  fcst <- round(forecast(fit,h = horizon,xreg = xreg_test)$mean)
  return(fcst)
}

lastYearForecast <- function(train,test,year = 2016,levels = NULL)
{
  train <- subset(train,Year %in% year)
  fit <- train[,.(last_yr = mean(Resolution,na.rm = TRUE)),by = levels]
  fcst <- round(join(test,fit,by = levels,type = "left")$last_yr)
  return(fcst)
}

dt_forecast <- data.table(NULL)
for(type in category_type)
{
  for(sub in subject_type)
  {
    dt_type <- subset(dt_train_2015,Category == type & Subject == sub)
    dt_type_test <- subset(dt_test,Category == type & Subject == sub)
    
    if(nrow(dt_type) == 0)
    {
      next
    }
    if(mean(dt_type$Resolution,na.rm = TRUE) < 5) # avoid forecasting for low vol
    {
      next
    }
  
    # TBATS with daily,monthly,yearly seasonality
    # ==================================================================
    tbats_forecast <- tbatsForecast(dt_type$Resolution)
  
    # ARIMA with regressors
    # ==================================================================
    dt_type[,TestMonthFlag := ifelse(as.numeric(as.character(dt_type$Month)) <= 3,1,0)]
    dt_type_test[,TestMonthFlag := ifelse(as.numeric(as.character(dt_type_test$Month)) <= 3,1,0)]
  
    dt_type[,TestWeekFlag := ifelse(as.numeric(as.character(dt_type$Week)) <= 11,1,0)]
    dt_type_test[,TestWeekFlag := ifelse(as.numeric(as.character(dt_type_test$Week)) <= 11,1,0)]
  
    xreg <- cbind(model.matrix(~dt_type$DayOfWeek),dt_type$TestMonthFlag,dt_type$TestWeekFlag)
    xreg <- xreg[,-1]
    colnames(xreg) <- c("Mon","Tue","Wed","Thu","Fri","Sat","TestMonthFlag","TestWeekFlag")
  
    xreg_test <- cbind(model.matrix(~dt_type_test$DayOfWeek),dt_type_test$TestMonthFlag,dt_type_test$TestWeekFlag)
    xreg_test <- xreg_test[,-1]
    colnames(xreg_test) <- c("Mon","Tue","Wed","Thu","Fri","Sat","TestMonthFlag","TestWeekFlag")
  
    arima_xreg_forecast <- arimaRegForecast(x = dt_type$Resolution,xreg = xreg,xreg_test = xreg_test)

  
    fcst_output <- data.table(dt_type_test,tbats_forecast,arima_xreg_forecast)
    dt_forecast <- rbindlist(list(dt_forecast,fcst_output))
  }
}

# Forecast adjustment
# =======================================================================
# TBATS and ARIMA give some negative forecasts especially for weekends
# Replace weekends with last year's actuals
# Replace low volume items with last yr actuals
# Replace NA values with 0

dt_last_yr <- subset(dt_train_2015,Year == 2016)[,.(last_year_forecast = mean(Resolution,na.rm = TRUE)),by = c("Month","DayOfWeek","Week","Category","Subject")]
dt_last_yr <- join(dt_test,dt_last_yr,by = c("Month","DayOfWeek","Week","Category","Subject"),type = "left")
dt_forecast <- join(dt_last_yr,dt_forecast,by = "ID",type ="left")

# Averaging forecasts
# ========================================================================
dt_forecast[,final_forecast := apply(dt_forecast[,c("tbats_forecast","arima_xreg_forecast","last_year_forecast"),with = F],1,mean,na.rm = TRUE)]
dt_forecast$final_forecast <- round(ifelse(dt_forecast$DayOfWeek %in% c(1,7) | dt_forecast$final_forecast < 0,dt_forecast$last_year_forecast,dt_forecast$final_forecast))
dt_forecast$final_forecast[is.na(dt_forecast$final_forecast)] <- 0

# Submission file
# =======================================================================
dt_submission <- dt_forecast[,c("ID","final_forecast"),with = FALSE]
names(dt_submission) <- c("ID","Resolution")

fwrite(x = dt_submission,file = "Resolution.csv",row.names = FALSE)

rm(list = ls())

# Improvements possible
# ===============================================================
# Use time series cross validation
# Use holiday effects
# Try other algorithms like DLM
# Better missing value imputation
# Better forecast adjustment
# Better ensembling
# Work on correlated series
# Specific forecasting models for each category/subject





