# Databricks notebook source
# install.packages("dplyr", repos = "https://cloud.r-project.org")
# install.packages("reshape", repos = "https://cloud.r-project.org")

# COMMAND ----------

library('httr')
library('jsonlite')
library('data.table')
library('tidyverse')
library('lubridate')
library('stringr')
library('reshape')

# COMMAND ----------

apiKey <- "639f33c64596431a31af3de642eb862642776235"
years <- seq(2017,2023, by=1)

startDate <- as.Date('2017-01-01')
endDate <- as.Date('2023-12-31')
fullDates <- seq(startDate, endDate, "days")

# COMMAND ----------

returnAPICall <- function(years, country, apiKey){
  call <- paste0("https://calendarific.com/api/v2/holidays?",
                  "&api_key=",apiKey,
                 "&country=",country,
                 "&year=",years)
  return (call)
}

# COMMAND ----------

#Map to years
apiCalls <- tibble::as_tibble(years)  

# COMMAND ----------

# MAGIC %md #### Functions to create Weekly and Monthly Holiday Datasets

# COMMAND ----------

country_hols <-function(country, TIME_VAR)
  {
    #Generate API html link
    apiCalls <- apiCalls %>% mutate(data.call = map(value, returnAPICall, country, apiKey))
  
    #GET html link
    apiCalls <- apiCalls %>% mutate(data.get = map(data.call, GET))
  
    #Convert to JSON
    apiCalls <- apiCalls %>% mutate(data.txt = map(data.get, content, "text"))
  
    apiCalls <- apiCalls %>% mutate(data.json = map(data.txt, fromJSON, flatten = TRUE))
  
    #Convert to Data Table
    apiCalls <- apiCalls %>% mutate(data.df = map(data.json, data.frame))

    apiCalls <- apiCalls %>% mutate(data.dt = map(data.df, as.data.table))

    #Clean data into desired format
    holidaysOut <- apiCalls %>% unnest(data.dt)
    holidaysOut2 <- as.data.table(holidaysOut)
  
    keepVars <- c("response.holidays.name","response.holidays.date.datetime.year", "response.holidays.date.datetime.month",
                  "response.holidays.date.datetime.day")
    holidaysOut2 <- holidaysOut2[,..keepVars]
  
    #Rename
    names(holidaysOut2) <- c("Holiday_Name", "Year", "Month", "Day")

    #Rename the holiday name so we can cast
    holidaysOut2$Holiday_Name <- str_replace_all(holidaysOut2$Holiday_Name, "'","")
    holidaysOut2$Holiday_Name <- str_replace_all(holidaysOut2$Holiday_Name, " ","_")
    #holidaysOut2$Holiday_Name <- str_replace_all(holidaysOut2$Holiday_Name, ".","")

    holidaysOut2$date <- parse_date_time(paste0(holidaysOut2$Year,"/", holidaysOut2$Month, "/", holidaysOut2$Day), 'ymd')

    #Cast
    holidaysOut3 <- holidaysOut2
    holidaysOut3 <- holidaysOut3[,c("Holiday_Name","date")]
    holidaysOut3$value <- 1

    holidaysOut3 <- cast(holidaysOut3, date ~Holiday_Name, mean)

    #Cleanse full date range
    fullDates <- as.data.table(fullDates)
    setnames(fullDates,"fullDates","date2")

    #Merge holiday to full date range
    holidaysOut3$date2 <- ymd(holidaysOut3$date)

    holidaysOut4 <- merge(fullDates, holidaysOut3, by=c("date2"), all=TRUE)
    holidaysOut4 <- holidaysOut4[order(date2)]

    #Create variables
    holidaysOut4 <- holidaysOut4[,-c("date")]
    setnames(holidaysOut4, "date2", "Date")

    #Create date variables
    holidaysOut4$Year <- year(holidaysOut4$Date)
    holidaysOut4$Month <- month(holidaysOut4$Date)
    holidaysOut4$Week <- week(holidaysOut4$Date)
    holidaysOut4$Day <- day(holidaysOut4$Date)
  
    # Create TIME_VAR column  
    if (TIME_VAR == "Week_Of_Year"){
      holidaysOut4$Week_Of_Year <- paste0(holidaysOut4$Year, str_pad(holidaysOut4$Week, 2, pad = "0"))
      } else {
      holidaysOut4$Month_Of_Year <- paste0(holidaysOut4$Year, str_pad(holidaysOut4$Month, 2, pad = "0"))
      }
  
    #Set country
    holidaysOut4$country <- country

    #Impute
    holidaysOut4[is.na(holidaysOut4)] <- 0

    #Select columns for table write
    keepCols <- c("country","Date","Year","Month","Week","Day", TIME_VAR)
    leftKeys <- holidaysOut4[,..keepCols]
    holidaysOut5 <- merge(leftKeys, holidaysOut4, by=keepCols)
    holidaysOut5 <- subset(holidaysOut5, select = -c(country, Date, Year, Month, Week, Day))
  
    return (holidaysOut5)

    }

# COMMAND ----------

get_combined <-function(country1, country2, TIME_VAR)
  {
    one_two_common = subset(country1, select = intersect(names(country1), names(country2)))
    two_one_common = subset(country2, select = intersect(names(country1), names(country2)))
  
    common = rbind(one_two_common, two_one_common)
    common = as.data.frame(common)
  
    if (TIME_VAR == "Week_Of_Year"){
      common = aggregate(.~Week_Of_Year, common, max)
    } else {
      common = aggregate(.~Month_Of_Year, common, max)      
    }
  
    both_hols = intersect(names(country1), names(country2)) 
  
    combined <- merge(subset(country1, select = setdiff(names(country1), both_hols[-1])), subset(country2, select = setdiff(names(country2), both_hols[-1])), by = TIME_VAR, all = TRUE)
    combined <- merge(combined, common, by = TIME_VAR, all = TRUE)
  
    return (combined)
  }

# COMMAND ----------

cleanse_cols <-function(df)
  {
  names(df) <- gsub(x = names(df), pattern = "\\(", replacement = "")
  names(df) <- gsub(x = names(df), pattern = "\\)", replacement = "")
  names(df) <- gsub(x = names(df), pattern = "\\.", replacement = "")
  names(df) <- gsub(x = names(df), pattern = " ", replacement = "_")
  names(df) <- gsub(x = names(df), pattern = "\\,", replacement = "")
  
  return (df)
  }

# COMMAND ----------

# MAGIC %md #### Weekly Aggregation

# COMMAND ----------

pt_hols_weekly = country_hols('PT', TIME_VAR = "Week_Of_Year")
es_hols_weekly = country_hols('ES', TIME_VAR = "Week_Of_Year")

pt_hols_weekly = aggregate(.~Week_Of_Year, pt_hols_weekly, max)
pt_hols_weekly$portugal_hol_flag<-rowSums(pt_hols_weekly[ , -1])

es_hols_weekly = aggregate(.~Week_Of_Year, es_hols_weekly, max)
es_hols_weekly$spain_hol_flag<-rowSums(es_hols_weekly[ , -1])

# COMMAND ----------

holidays_weekly = get_combined(pt_hols_weekly, es_hols_weekly, TIME_VAR = "Week_Of_Year")
holidays_weekly = holidays_weekly %>% relocate(spain_hol_flag, .after=Week_Of_Year) %>% relocate(portugal_hol_flag, .after=spain_hol_flag)
holidays_weekly = cleanse_cols(holidays_weekly)
holidays_weekly$Week_Of_Year = as.integer(holidays_weekly$Week_Of_Year)
holidays_weekly[is.na(holidays_weekly)] <- 0

# COMMAND ----------

display(holidays_weekly)

# COMMAND ----------

# MAGIC %md #### Monthly Aggregation

# COMMAND ----------

pt_hols_monthly = country_hols('PT', TIME_VAR = "Month_Of_Year")
es_hols_monthly = country_hols('ES', TIME_VAR = "Month_Of_Year")

pt_hols_monthly = aggregate(.~Month_Of_Year, pt_hols_monthly, max)
pt_hols_monthly$portugal_hol_flag<-rowSums(pt_hols_monthly[ , -1])

es_hols_monthly = aggregate(.~Month_Of_Year, es_hols_monthly, max)
es_hols_monthly$spain_hol_flag<-rowSums(es_hols_monthly[ , -1])

# COMMAND ----------

holidays_monthly = get_combined(pt_hols_monthly, es_hols_monthly, TIME_VAR = "Month_Of_Year")
holidays_monthly = holidays_monthly %>% relocate(spain_hol_flag, .after=Month_Of_Year) %>% relocate(portugal_hol_flag, .after=spain_hol_flag)
holidays_monthly = cleanse_cols(holidays_monthly)
holidays_monthly$Month_Of_Year = as.integer(holidays_monthly$Month_Of_Year)
holidays_monthly[is.na(holidays_monthly)] <- 0

# COMMAND ----------

display(holidays_monthly)

# COMMAND ----------

# MAGIC %md #### Write out the Holidays Datasets

# COMMAND ----------

library(SparkR)
sparkR.session()

# COMMAND ----------

holidays_path_weekly = "dbfs:/mnt/adls/Tables/DBI_HOLIDAYS_WEEKLY"
holidays_path_monthly = "dbfs:/mnt/adls/Tables/DBI_HOLIDAYS_MONTHLY"

# COMMAND ----------

saveDF(createDataFrame(holidays_weekly), source = "delta", path = holidays_path_weekly, mode = 'overwrite', overwriteSchema = 'true')
saveDF(createDataFrame(holidays_monthly), source = "delta", path = holidays_path_monthly, mode = 'overwrite', overwriteSchema = 'true')

# COMMAND ----------

