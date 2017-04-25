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
dt_train <- fread("Train/Contacts_Pre_2017.csv",header = TRUE,na.strings = "")
dt_test <- fread("Test/Contacts2017.csv",header = TRUE,na.strings = "")

# Changing data type
# =============================================================
dt_train[,START.DATE := as.Date(START.DATE)]
dt_train[,END.DATE := as.Date(END.DATE)]
dt_test[,Date := as.Date(Date)]

# Exploratory Analysis
# ==============================================================
freqTable <- as.data.frame.matrix(table(dt_train[['START.DATE']],dt_train[['CONTACT.TYPE']]))

# 2557 weeks of data
# Few of the categories had > 1 contacts per day which needed to rolled for daily level forecasting
# Missing weeks for contact types existed. These would be problematic for time series forecasting and needs to be treated

# Aggregating data at daily level
# ==============================================================
dt_train <- dt_train[,.(Contacts = sum(Contacts,na.rm = TRUE)),by = c("START.DATE","CONTACT.TYPE")]
names(dt_train) <- c("Date","CONTACT.TYPE","Contacts")

# Adding missing weeks
# ==============================================================
unique_type <- unique(dt_train[['CONTACT.TYPE']])

dt_train_full <- data.table(NULL) # Initialising data table
for(type in unique_type)
{
  dt_type <- subset(dt_train, CONTACT.TYPE == type)
  min_date <- min(dt_type$Date)
  max_date <- max(dt_train$Date)
  dt_temp <- data.table(Date = seq(min_date,max_date,by = 1),CONTACT.TYPE = type)
  dt_temp <- join(x = dt_temp,y = dt_type,type = "left", by = c("Date","CONTACT.TYPE"))
  dt_train_full <- rbindlist(list(dt_train_full,dt_temp))
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
year_variation <- dt_train_full[,.(yearly_avg_contacts = round(mean(Contacts,na.rm = TRUE))), by = c("Year","CONTACT.TYPE")]
year_variation <- year_variation[order(CONTACT.TYPE,Year),]
ggplot(data = year_variation,aes(x = Year,y = yearly_avg_contacts)) + geom_line() + facet_wrap(~CONTACT.TYPE,scales = "free")

# Most contact types have vastly different patterns across the years
# Yearly variation and contact type variation exists
# Tweet input and Installation report are very low volume
# Increasing trend in web input,mail recieved, visit, internal management especially in last two years
# Decreasing trend in call input, fax, mail input

# Hence enough to consider last two years: 2015 and 2016 for our analysis
# ========================================================================
dt_train_2015 <- subset(dt_train_full, Year >= 2015)
rm(dt_train_full)

# Missing value imputation
# ========================================================================
dt_train_2015$Contacts[is.na(dt_train_2015$Contacts)] <- 0

# Monthly variation
# ========================================================================
monthly_variation <- dt_train_2015[,.(monthly_avg_contacts = round(mean(Contacts,na.rm = TRUE))), by = c("Month","CONTACT.TYPE")]
monthly_variation <- monthly_variation[order(CONTACT.TYPE,Month),]
ggplot(data = monthly_variation,aes(x = as.integer(Month),y = monthly_avg_contacts)) + geom_line() + facet_wrap(~CONTACT.TYPE,scales = "free") + scale_x_continuous(breaks = 1:12) + ggtitle("Monthly variation in contacts") 


# Definite monthly variation
# Inverted cone shape in months of July,August and September
# Lowest in August
# Installation report and tweet input flat

# Day of Week variation
# ===========================================================================
dow_variation <- dt_train_2015[,.(dow_avg_contacts = round(mean(Contacts,na.rm = TRUE))), by = c("DayOfWeek","CONTACT.TYPE")]
dow_variation <- dow_variation[order(CONTACT.TYPE,DayOfWeek),]
ggplot(data = dow_variation,aes(x = as.integer(DayOfWeek),y = dow_avg_contacts)) + geom_line() + facet_wrap(~CONTACT.TYPE,scales = "free") + scale_x_continuous(breaks = 1:7) + ggtitle("DOW variation in contacts") 
 
# Day of the Week variation exists
# Weekends have low/zero contacts
# Contacts generally taper off during the week

# Summary
# -----------------------------------------------------------------
# No use of working with full history
# Yearly,monthly and day of the week variation exists
# Some contact types like tweet input and installation report are extremely low volume
# Call input has high volume and would contribute most to the RMSE

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
  fit <- train[,.(last_yr = mean(Contacts,na.rm = TRUE)),by = levels]
  fcst <- round(join(test,fit,by = levels,type = "left")$last_yr)
  return(fcst)
}

dt_forecast <- data.table(NULL)
for(type in unique_type)
{
  dt_type <- subset(dt_train_2015, CONTACT.TYPE == type)
  dt_type_test <- subset(dt_test, CONTACT.TYPE == type)
  
  # TBATS with daily,monthly,yearly seasonality
  # ==================================================================
  tbats_forecast <- tbatsForecast(dt_type$Contacts)
  
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
  
  arima_xreg_forecast <- arimaRegForecast(x = dt_type$Contacts,xreg = xreg,xreg_test = xreg_test)
  
  # Last Year Forecast
  # ======================================================================
  last_year_forecast <- lastYearForecast(train = dt_type,test = dt_type_test,levels = c("CONTACT.TYPE","DayOfWeek","Month","Week"))
  
  fcst_output <- data.table(dt_test,tbats_forecast,arima_xreg_forecast,last_year_forecast)
  dt_forecast <- rbindlist(list(dt_forecast,fcst_output))
}

# Forecast adjustment
# =======================================================================
# TBATS and ARIMA give some negative forecasts especially for weekends
# Weekends generally have low volume
# Replace weekends with last year's actuals
# Replace NA values with 0

# Averaging forecasts
# ========================================================================
dt_forecast[,final_forecast := apply(dt_forecast[,c("tbats_forecast","arima_xreg_forecast","last_year_forecast"),with = F],1,mean,na.rm = TRUE)]
dt_forecast$final_forecast <- round(ifelse(dt_forecast$DayOfWeek %in% c(1,7) | dt_forecast$final_forecast < 0,dt_forecast$last_year_forecast,dt_forecast$final_forecast))
dt_forecast$final_forecast[is.na(dt_forecast$final_forecast)] <- 0

# Submission file
# =======================================================================
dt_submission <- dt_forecast[,c("ID","final_forecast"),with = FALSE]
names(dt_submission) <- c("ID","Contacts")

fwrite(x = dt_submission,file = "Contacts.csv",row.names = FALSE)

rm(list = ls())

# Improvements possible
# ===============================================================
# Use time series cross validation
# Use holiday effects
# Try other time series algorithms like DLM
# Better missing value imputation
# Better forecast adjustment
# Better ensembling
# Specific forecasting models for each contact type





