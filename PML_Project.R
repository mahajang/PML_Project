library(dplyr)
library(ggplot2)
library(lubridate)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(corrplot)

data.train<- read.csv("C:/1. HP- Work files/HP WORK 1st Feb 2015/COURSERA_COURSES/8. Machine_Learning/Week4_Assignment/pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))

data.test<- read.csv("C:/1. HP- Work files/HP WORK 1st Feb 2015/COURSERA_COURSES/8. Machine_Learning/Week4_Assignment/pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))

#Data understanding
dim(data.train)
glimpse(data.train)

#Data Transformation : Convert date
data.train$cvtd_timestamp<- as.Date(data.train$cvtd_timestamp, format = "%m/%d/%Y %H:%M")
data.train$Day<-factor(weekdays(data.train$cvtd_timestamp)) #Add day variable

#Exploratory Data Analysis
table(data.train$classe) 
prop.table(table(data.train$classe)) 
prop.table(table(data.train$user_name)) 
prop.table(table(data.train$user_name,data.train$classe),1) 
prop.table(table(data.train$user_name,data.train$classe),2) 
prop.table(table(data.train$classe, data.train$Day),1) 
qplot(x=Day, fill=classe, data = data.train)

##Key Insights from Exploratory Data Analysis:
### 1.Class-A activity is the most frequently used activity (28.5%) and is most frequently used by user-Jeremy
### 2.Adelmo is the most frequent user of across acitivities (20%) but he uses Class "C" activity most frequently.
### 3.Majority of the actitivies happened during Saturday's and Classes A and B are the most frequently used activites.

##Data Prepration:

##Data Cleaning:
#### Remove columns with NA missing values
data.train <- data.train[, colSums(is.na(data.train)) == 0]
data.test <- data.test[, colSums(is.na(data.test)) == 0] 

#### Remove columns that are not relevant to accelerometer measurements.
classe<- data.train$classe
trainRemove<- grepl("^X|timestamp|window", names(data.train))
data.train<- data.train[, !trainRemove]
trainCleaned<- data.train[, sapply(data.train, is.numeric)]
trainCleaned$classe<- classe
testRemove<- grepl("^X|timestamp|window", names(data.test))
data.test<- data.test[, !testRemove]
testCleaned<- data.test[, sapply(data.test, is.numeric)]

####Now, the cleaned data contains 19622 observations and 53 variables for both train and test datasets

##Create Train and Test data sets:
set.seed(22519)
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]

##Data Modelling
#####Indetifying significant variables:
##### We will fit a predictive model using Random Forest algorithm as it gives important variables and removes multicollinearity  and outliers. We will also use 5-fold cross validation when applying the algorithm.

controlRf <- trainControl(method="cv", 5)
rfmod<- train(classe ~., data=trainData, method="rf", trControl=controlRf, importance=TRUE, ntree=100)
rfmod


##### Estimate the performance of the model on the Validation data set:
predictRfmod<- predict(rfmod, testData)
confusionMatrix(testData$classe, predictRfmod)

accuracy <- postResample(predictRfmod, testData$classe)
accuracy

Error <- 1 - as.numeric(confusionMatrix(testData$classe, predictRfmod)$overall[1])
Error

###So, the estimated accuracy of the model is 99.32% and the estimated out-of-sample error is 0.68%.


#Predicting on Test Data Set
result <- predict(rfmod, testCleaned[, -length(names(testCleaned))])
result


#Appendix

### Correlation Matrix
corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="circle")

### Tree Visulization 
rtree<- rpart(classe ~ ., data=trainData, method="class")
prp(rtree)

