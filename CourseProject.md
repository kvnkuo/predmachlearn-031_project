---
title: "The predictions of Weight Lifting Exercise dataset"
author: "Kevin Kuo"
date: "2015/08/20"
output: html_document:
    pandoc_args: [
      "+RTS", "-K64m",
      "-RTS"
    ]
---


```
## Loading required package: lattice
## Loading required package: ggplot2
```
The primary goal of this report is to find a appropriate model to correctly predict the outcomes of the Weight Lifting Exercise dataset. The WLE dataset comes from Velloso, E. et al. [**R. 1**].

We start by reading the data into R and perform some basic explorarations.

```r
pmldata <- read.csv("pml-training.csv",
                      dec=".", stringsAsFactors = FALSE, 
                      allowEscapes = TRUE, sep=",", 
                      quote = "\"", header=TRUE, fill = TRUE)
```

The dimension of the whole dataset

```r
dim(pmldata)
```

```
## [1] 19622   160
```

The prediction is to output classification results. So, we must make sure the classe column is of factor type.

```r
pmldata$classe <- factor(pmldata$classe)
```

Before doing the simulation, we import the parallel processing library to speed up the processing.

```r
library(doParallel)
registerDoParallel(cores=4)
```

The whole data will be divided into two groups: traing(80%) and testing(20%).

```r
library(caret)
inTrain <- createDataPartition(y=pmldata$classe, p=0.8, list=FALSE)
training <- pmldata[inTrain,]
testing <- pmldata[-inTrain,]
```


```r
dim(training)
```

```
## [1] 15699   160
```

Both selection of model and method for feature extraction are also inspired by the paper [**R. 1**]. We use Random Forest method to build our model. The original method for feature extraction is based on the followings rules:  
1. In the belt, were selected the mean and variance of the roll, maximum, range and variance of the accelerometer vector, variance of the gyro and variance of the magnetometer.  
2. the  arm,  the  variance  of  the  accelerometer  vector  and  the maximum and minimum of the magnetometer were selected.  
3. In the dumbbell, the selected features were the maximum of the  acceleration,  variance  of  the  gyro  and  maximum  and minimum of the magnetometer, while in the glove, the sum of  the  pitch  and  the  maximum  and  minimum  of  the  gyro were selected.  

Instead of using max, min, sum, mean, var and range of the measured data to build the model, we try an alternative method. For max, min, sum and range values, we simply use the measured data as equivalent ones. For var values, we use the square of the measured data as equivalent ones. For example, in the belt, we use roll_belt instead of mean(roll_belt) in specific time window. Also, in the belt, we use roll_belt^2 instead of var(roll_belt) in specific time window. Our simplified assumption is based on the idea that var is linear equivalent to square.  

Following is the model call(to save processing time, we load the saved one.)

```r
modFit_rf$call
```

```
## train.formula(form = classe ~ roll_belt + roll_belt^2 + total_accel_belt + 
##     accel_belt_x^2 + accel_belt_y^2 + accel_belt_z^2 + gyros_belt_x^2 + 
##     gyros_belt_y^2 + gyros_belt_z^2 + magnet_belt_x^2 + magnet_belt_y^2 + 
##     magnet_belt_z^2 + accel_arm_x^2 + accel_arm_y^2 + accel_arm_z^2 + 
##     magnet_arm_x + magnet_arm_y + magnet_arm_z + total_accel_dumbbell + 
##     gyros_dumbbell_x^2 + gyros_dumbbell_y^2 + gyros_dumbbell_z^2 + 
##     magnet_dumbbell_x + magnet_dumbbell_y + magnet_dumbbell_z + 
##     pitch_forearm + gyros_forearm_x + gyros_forearm_y + gyros_forearm_z, 
##     data = training, method = "rf", prox = FALSE)
```

The confusion matrix of trained model is listed below. The error rates are not bad.

```r
modFit_rf$finalModel$confusion
```

```
##      A    B    C    D    E class.error
## A 4444    5    5    9    1 0.004480287
## B   23 2969   42    4    0 0.022712311
## C    3   23 2699   13    0 0.014243974
## D    7    5   59 2500    2 0.028371551
## E    0    2    5    5 2874 0.004158004
```

We investigate the mode with testing data

```r
pred_rf <- predict(modFit_rf, testing)
```
 and check it with confusion matrix.

```r
# Confusion matrix
confusionMatrix(pred_rf, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1110    9    0    1    0
##          B    5  743    6    0    0
##          C    0    7  676   13    2
##          D    1    0    2  629    1
##          E    0    0    0    0  718
## 
## Overall Statistics
##                                           
##                Accuracy : 0.988           
##                  95% CI : (0.9841, 0.9912)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9848          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9946   0.9789   0.9883   0.9782   0.9958
## Specificity            0.9964   0.9965   0.9932   0.9988   1.0000
## Pos Pred Value         0.9911   0.9854   0.9685   0.9937   1.0000
## Neg Pred Value         0.9979   0.9950   0.9975   0.9957   0.9991
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2829   0.1894   0.1723   0.1603   0.1830
## Detection Prevalence   0.2855   0.1922   0.1779   0.1614   0.1830
## Balanced Accuracy      0.9955   0.9877   0.9908   0.9885   0.9979
```
The error rates approximately match with error rates in the traing data. The accuracy is 98.8% and its 95% C.I. is (0.9841, 0.9912). Also, the Sensitivity and Specificity for each outcome class are both high. We can conclude that the model's performance is acceptable. Let's proceed to the final step, to predict the test data.

Now, we load the test data 

```r
pmldata_test <- read.csv("pml-testing.csv",
                    dec=".", stringsAsFactors = FALSE, 
                    allowEscapes = TRUE, sep=",", 
                    quote = "\"", header=TRUE, fill = TRUE)
```
and perform the prediction.

```r
pred_rf_test <- predict(modFit_rf, pmldata_test)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
pred_rf_test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Note: We also use Conditional Inference Trees method to build our model **a. 2**, **a. 3**. The predicted outcomes are no as good as the aboves. However, its predicted test results are the same as our Random Forest based model.           
### Appendix
**a. 1** The main prediction model (Random Forest)

```r
modFit_rf <- train(
    classe ~ roll_belt +
        roll_belt ^ 2 +
        total_accel_belt +
        accel_belt_x ^ 2 +
        accel_belt_y ^ 2 +
        accel_belt_z ^ 2 +
        gyros_belt_x ^ 2 +
        gyros_belt_y ^ 2 +
        gyros_belt_z ^ 2 +
        magnet_belt_x ^ 2 +
        magnet_belt_y ^ 2 +
        magnet_belt_z ^ 2 +
        accel_arm_x ^ 2 +
        accel_arm_y ^ 2 +
        accel_arm_z ^ 2 +
        magnet_arm_x +
        magnet_arm_y +
        magnet_arm_z +
        total_accel_dumbbell +
        gyros_dumbbell_x ^ 2 +
        gyros_dumbbell_y ^ 2 +
        gyros_dumbbell_z ^ 2 +
        magnet_dumbbell_x +
        magnet_dumbbell_y +
        magnet_dumbbell_z +
        pitch_forearm +
        gyros_forearm_x +
        gyros_forearm_y +
        gyros_forearm_z
    , data = training, method = "rf", prox = FALSE
)
```

**a. 2** Another prediction model (Conditional Inference Trees)

```r
modFit_ctree$call
```

```
## train.formula(form = classe ~ roll_belt + roll_belt^2 + total_accel_belt + 
##     accel_belt_x^2 + accel_belt_y^2 + accel_belt_z^2 + gyros_belt_x^2 + 
##     gyros_belt_y^2 + gyros_belt_z^2 + magnet_belt_x^2 + magnet_belt_y^2 + 
##     magnet_belt_z^2 + accel_arm_x^2 + accel_arm_y^2 + accel_arm_z^2 + 
##     magnet_arm_x + magnet_arm_y + magnet_arm_z + total_accel_dumbbell + 
##     gyros_dumbbell_x^2 + gyros_dumbbell_y^2 + gyros_dumbbell_z^2 + 
##     magnet_dumbbell_x + magnet_dumbbell_y + magnet_dumbbell_z + 
##     pitch_forearm + gyros_forearm_x + gyros_forearm_y + gyros_forearm_z, 
##     data = training, method = "ctree")
```


```r
pred_ctree <- predict(modFit_ctree, testing)
```


```r
# cross validation 
confusionMatrix(pred_ctree, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1052   41   12   17    8
##          B   28  642   28   23   14
##          C    9   40  602   46    9
##          D   19   23   23  542   13
##          E    8   13   19   15  677
## 
## Overall Statistics
##                                          
##                Accuracy : 0.896          
##                  95% CI : (0.886, 0.9054)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2e-16        
##                                          
##                   Kappa : 0.8684         
##  Mcnemar's Test P-Value : 0.08557        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9427   0.8458   0.8801   0.8429   0.9390
## Specificity            0.9722   0.9706   0.9679   0.9762   0.9828
## Pos Pred Value         0.9310   0.8735   0.8527   0.8742   0.9249
## Neg Pred Value         0.9771   0.9633   0.9745   0.9694   0.9862
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2682   0.1637   0.1535   0.1382   0.1726
## Detection Prevalence   0.2880   0.1874   0.1800   0.1580   0.1866
## Balanced Accuracy      0.9574   0.9082   0.9240   0.9096   0.9609
```

**a. 3** Apply previous ctree model to test data (20 test cases)

```r
pred_ctree_test <- predict(modFit_ctree, pmldata_test)
```

```
## Loading required package: party
## Loading required package: grid
## Loading required package: mvtnorm
## Loading required package: modeltools
## Loading required package: stats4
## Loading required package: strucchange
## Loading required package: zoo
## 
## Attaching package: 'zoo'
## 
## The following objects are masked from 'package:base':
## 
##     as.Date, as.Date.numeric
## 
## Loading required package: sandwich
```

```r
pred_ctree_test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

### Reference

**R. 1**
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
Read more: http://groupware.les.inf.puc-rio.br/har#wle_paper_section#ixzz3jKxnqNYs
