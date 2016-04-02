# PML_Project
mahajang  
April 2, 2016  

###Inroduction : 
#####Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

###Data Loading:

```r
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library(ggplot2)
library(lubridate)
library(caret)
```

```
## Loading required package: lattice
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.2.4
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```r
library(rpart)
library(rpart.plot)
library(corrplot)

data.train<- read.csv("C:/1. HP- Work files/HP WORK 1st Feb 2015/COURSERA_COURSES/8. Machine_Learning/Week4_Assignment/pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))

data.test<- read.csv("C:/1. HP- Work files/HP WORK 1st Feb 2015/COURSERA_COURSES/8. Machine_Learning/Week4_Assignment/pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))
```

###Data Understandaing:

```r
dim(data.train)
```

```
## [1] 19622   160
```

###Data Transformation : Convert date and add new variable (Day)

```r
data.train$cvtd_timestamp<- as.Date(data.train$cvtd_timestamp, format = "%m/%d/%Y %H:%M")
data.train$Day<-factor(weekdays(data.train$cvtd_timestamp)) #Add day variable
```

###Exploratory Data Analysis

```r
table(data.train$classe) 
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
prop.table(table(data.train$classe)) 
```

```
## 
##         A         B         C         D         E 
## 0.2843747 0.1935073 0.1743961 0.1638977 0.1838243
```

```r
prop.table(table(data.train$user_name)) 
```

```
## 
##    adelmo  carlitos   charles    eurico    jeremy     pedro 
## 0.1983488 0.1585975 0.1802059 0.1564570 0.1733768 0.1330140
```

```r
prop.table(table(data.train$user_name,data.train$classe),1) 
```

```
##           
##                    A         B         C         D         E
##   adelmo   0.2993320 0.1993834 0.1927030 0.1323227 0.1762590
##   carlitos 0.2679949 0.2217224 0.1584190 0.1561697 0.1956941
##   charles  0.2542421 0.2106900 0.1524321 0.1815611 0.2010747
##   eurico   0.2817590 0.1928339 0.1592834 0.1895765 0.1765472
##   jeremy   0.3459730 0.1437390 0.1916520 0.1534392 0.1651969
##   pedro    0.2452107 0.1934866 0.1911877 0.1796935 0.1904215
```

```r
prop.table(table(data.train$user_name,data.train$classe),2) 
```

```
##           
##                    A         B         C         D         E
##   adelmo   0.2087814 0.2043719 0.2191701 0.1601368 0.1901857
##   carlitos 0.1494624 0.1817224 0.1440678 0.1511194 0.1688384
##   charles  0.1611111 0.1962075 0.1575102 0.1996269 0.1971167
##   eurico   0.1550179 0.1559126 0.1428989 0.1809701 0.1502634
##   jeremy   0.2109319 0.1287859 0.1905319 0.1623134 0.1558082
##   pedro    0.1146953 0.1329997 0.1458212 0.1458333 0.1377876
```

```r
prop.table(table(data.train$classe, data.train$Day),1) 
```

```
##    
##      Saturday  Thursday
##   A 0.5833804 0.4166196
##   B 0.5600147 0.4399853
##   C 0.5651030 0.4348970
##   D 0.5478220 0.4521780
##   E 0.5581302 0.4418698
```

```r
qplot(x=Day, fill=classe, data = data.train)
```

![](PML_Project_v2_files/figure-html/unnamed-chunk-4-1.png)

###Key Insights from Exploratory Data Analysis:

```r
### 1.Class-A activity is the most frequently used activity (28.5%) and is most frequently used by user-Jeremy
### 2.Adelmo is the most frequent user of across acitivities (20%) but he uses Class "C" activity most frequently.
### 3.Majority of the actitivies happened during Saturday's and Classes A and B are the most frequently used activites.
```

###Data Cleaning:

```r
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
```

####Now, the cleaned data contains 19622 observations and 53 variables for both train and test datasets

###Create Train and Test data sets:

```r
set.seed(22519)
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]
```

###Data Modelling:

```r
#####Indetifying significant variables:
##### We will fit a predictive model using Random Forest algorithm as it gives important variables and removes multicollinearity  and outliers. We will also use 5-fold cross validation when applying the algorithm.

controlRf <- trainControl(method="cv", 5)
rfmod<- train(classe ~., data=trainData, method="rf", trControl=controlRf, importance=TRUE, ntree=100)
rfmod
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10989, 10989, 10991, 10990, 10989 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa     Accuracy SD  Kappa SD   
##    2    0.9894444  0.986646  0.001783856  0.002258937
##   27    0.9908279  0.988396  0.001417511  0.001795118
##   52    0.9852224  0.981303  0.003115559  0.003943156
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

###Accuacy of the model on Validation data set:

```r
predictRfmod<- predict(rfmod, testData)
confusionMatrix(testData$classe, predictRfmod)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    1    0    0    1
##          B    7 1127    5    0    0
##          C    0    1 1022    3    0
##          D    0    0   14  948    2
##          E    0    0    1    5 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9932          
##                  95% CI : (0.9908, 0.9951)
##     No Information Rate : 0.2853          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9914          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9958   0.9982   0.9808   0.9916   0.9972
## Specificity            0.9995   0.9975   0.9992   0.9968   0.9988
## Pos Pred Value         0.9988   0.9895   0.9961   0.9834   0.9945
## Neg Pred Value         0.9983   0.9996   0.9959   0.9984   0.9994
## Prevalence             0.2853   0.1918   0.1771   0.1624   0.1833
## Detection Rate         0.2841   0.1915   0.1737   0.1611   0.1828
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9977   0.9979   0.9900   0.9942   0.9980
```

```r
accuracy <- postResample(predictRfmod, testData$classe)
accuracy
```

```
##  Accuracy     Kappa 
## 0.9932031 0.9914020
```

```r
Error <- 1 - as.numeric(confusionMatrix(testData$classe, predictRfmod)$overall[1])
Error
```

```
## [1] 0.006796941
```
####So, the estimated accuracy of the model is 99.32% and the estimated out-of-sample error is 0.68%.


###Predicting on Test Data Set

```r
result <- predict(rfmod, testCleaned[, -length(names(testCleaned))])
result
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

###Appendix

#### Correlation Matrix

```r
corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="circle")
```

![](PML_Project_v2_files/figure-html/unnamed-chunk-11-1.png)

#### Tree Visualization

```r
rtree<- rpart(classe ~ ., data=trainData, method="class")
prp(rtree)
```

![](PML_Project_v2_files/figure-html/unnamed-chunk-12-1.png)
