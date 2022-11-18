# Code for SLDM2 - homework2
# JARED HANSEN, 10/25/2018
















#=======================================================================
#=======================================================================
#======== DATA GENERATION FOR ALL OF PROBLEM 4D ========================
#=======================================================================
#=======================================================================


# THIS FUNCTION GENERATES ONE DATA SET 
gen.data <- function(n){
  y <- rbinom(p = 0.7, size = 1, n = n)
  dat <- data.frame(x = double(n), y = y)
  for(i in 1:n){
    if(dat$y[i] == 1){
      dat$x[i] <- rnorm(n = 1, mean = 1.5, sd = 1)
    } else {
      dat$x[i] <- rnorm(n = 1, mean = 0, sd = 1)
    }
  }
  return(dat)
}


#========================================
#===== Generating 100 N=100 samples =====
#========================================
samples100 = list()
# Generating 100 samples for N=100
for(i in 1:100){
  samples100[[i]] <- gen.data(100)
}
samples100 <- as.data.frame(samples100)


#========================================
#===== Generating 100 N=200 samples =====
#========================================
samples200 = list()
# Generating 100 samples for N=200
for(i in 1:100){
  samples200[[i]] <- gen.data(200)
}
samples200 <- as.data.frame(samples200)


#========================================
#===== Generating 100 N=500 samples =====
#========================================
samples500 = list()
# Generating 100 samples for N=500
for(i in 1:100){
  samples500[[i]] <- gen.data(500)
}
samples500 <- as.data.frame(samples500)


#========================================
#===== Generating 100 N=1000 samples ====
#========================================
samples1000 = list()
# Generating 100 samples for N=1000
for(i in 1:100){
  samples1000[[i]] <- gen.data(1000)
}
samples1000 <- as.data.frame(samples1000)


































########################################
# BAYES CLASSIFIER #####################
########################################

Bayes.fctn <- function(givenData){
  
  # How many obs in the data
  gD.length <- nrow(givenData)
  
  # Initialize empty vector for predictions
  bayes.preds <- vector("integer", gD.length)
  
  # Classify each value of the data
  for(i in 1:gD.length){
    if(givenData$x[i] < 0.4){
      bayes.preds[i] <- 0
    } else{
      bayes.preds[i] <- 1
    }
  }
  
  # See which obs are misclassified, put in vector
  choices <- (bayes.preds == givenData$y)
  # Count the number of FALSE classifications
  wrong <- length(choices[choices == "FALSE"])
  
  # Prediction error
  predError <- (wrong/gD.length)
  
  return(predError)
}



#====== Errors for Bayes Classifier at N=100 ======
# Initialize empty vector for N=100 errors
errors100 <- vector("numeric", 100)

# Calculating error for the N=100 samples
for(i in 1:100){
  errors100[i] <- Bayes.fctn(samples100[ , c(2*i-1,2*i)])
}
avgBayes100error <- mean(errors100)
avgBayes100error
sd(errors100)


#====== Errors for Bayes Classifier at N=200 ======
# Initialize empty vector for N=200 errors
errors200 <- vector("numeric", 100)

# Calculating error for the N=200 samples
for(i in 1:100){
  errors200[i] <- Bayes.fctn(samples200[ , c(2*i-1,2*i)])
}
avgBayes200error <- mean(errors200)
avgBayes200error
sd(errors200)


#====== Errors for Bayes Classifier at N=500 ======
# Initialize empty vector for N=500 errors
errors500 <- vector("numeric", 100)

# Calculating error for the N=500 samples
for(i in 1:100){
  errors500[i] <- Bayes.fctn(samples500[ , c(2*i-1,2*i)])
}
avgBayes500error <- mean(errors500)
avgBayes500error
sd(errors500)


#====== Errors for Bayes Classifier at N=1000 ======
# Initialize empty vector for N=1000 errors
errors1000 <- vector("numeric", 100)

# Calculating error for the N=1000 samples
for(i in 1:100){
  errors1000[i] <- Bayes.fctn(samples1000[ , c(2*i-1,2*i)])
}
avgBayes1000error <- mean(errors1000)
avgBayes1000error
sd(errors1000)




























########################################
# K-NEAREST NEIGHBORS CLASSIFIER #######
########################################
# Run this 100 times (reset seed each time)
# Keep track of 'k' for each trial: need to report average value of k

library(caret)

# 
# # k-NN using caret:
# library(ISLR)
# library(caret)
# 
# # Split the data:
# data(iris)
# indxTrain <- createDataPartition(y = iris$Sepal.Length,p = 0.75,list = FALSE)
# training <- iris[indxTrain,]
# testing <- iris[-indxTrain,]
# 
# # Run k-NN:
# set.seed(400)
# ctrl <- trainControl(method="repeatedcv",repeats = 3)
# knnFit <- train(Species ~ ., data = training, method = "knn", trControl = ctrl, preProcess = c("center","scale"),tuneLength = 20)
# knnFit
# 
# #Use plots to see optimal number of clusters:
# #Plotting yields Number of Neighbours Vs accuracy (based on repeated cross validation)
# plot(knnFit)




knnErrorFctn <- function(dataPassed) {
  
  names(dataPassed) <- c("x", "y")
  
  knnModel <- train(as.factor(y) ~ x,
                    data = dataPassed,
                    method = 'knn',
                    trControl = trainControl(method = 'cv', number = 5))
  
  # Prediction error
  predError <- 1 - max(knnModel$results$Accuracy)
  
  return(predError)
}










#====== Errors for KNN Classifier at N=100 ======

# Initialize empty vector for N=100 errors
knnErrors100 <- vector("numeric", 100)
# Calculating error for the N=100 samples
for(i in 1:100){
  knnErrors100[i] <- knnErrorFctn(samples100[ , c(2*i-1,2*i)])
}
knnErrors100
avgknn100error <- mean(knnErrors100)
avgknn100error
sd(knnErrors100)






#====== Errors for KNN Classifier at N=200 ======

# Initialize empty vector for N=200 errors
knnErrors200 <- vector("numeric", 100)
# Calculating error for the N=200 samples
for(i in 1:100){
  knnErrors200[i] <- knnErrorFctn(samples200[ , c(2*i-1,2*i)])
}
knnErrors200
avgknn200error <- mean(knnErrors200)
avgknn200error
sd(knnErrors200)




#====== Errors for KNN Classifier at N=500 ======

# Initialize empty vector for N=500 errors
knnErrors500 <- vector("numeric", 100)
# Calculating error for the N=500 samples
for(i in 1:100){
  knnErrors500[i] <- knnErrorFctn(samples500[ , c(2*i-1,2*i)])
}
knnErrors500
avgknn500error <- mean(knnErrors500)
avgknn500error
sd(knnErrors500)





#====== Errors for KNN Classifier at N=1000 ======

# Initialize empty vector for N=1000 errors
knnErrors1000 <- vector("numeric", 100)
# Calculating error for the N=1000 samples
for(i in 1:100){
  knnErrors1000[i] <- knnErrorFctn(samples1000[ , c(2*i-1,2*i)])
}
knnErrors1000
avgknn1000error <- mean(knnErrors1000)
avgknn1000error
sd(knnErrors1000)







#===================================================================
# COULDN'T GET THE CODE CHUNK BELOW (TO GET THE K'S) TO WORK :'(
#===================================================================
knnKFctn <- function(dataPassed) {
  names(dataPassed) <- c("x", "y")
  knnModel <- train(as.factor(y) ~ x,
                    data = dataPassed,
                    method = 'knn',
                    trControl = trainControl(method = 'cv', number = 5))
  
  # The k value that maximizes accuracy

  sample(knnModel$results$k[which(knnModel$results$Accuracy == max(knnModel$results$Accuracy))], size = 1) 
  return(max_k)
}


# Initialize empty vector for N=200 k's
knnKs200 <- vector("integer", 100)
# Calculating k for the N=200 samples
for(i in 1:100){
  knnKs200[i] <- knnKFctn(samples200[ , (2*i-1):(2*i)])
}
knnKs200
avgknn200ks <- mean(knnKs200)
avgknn200ks








########################################
# LOGISTIC REGRESSION ##################
########################################
# Run this 100 times (reset seed each time)
# Keep track of classification error for each trial (error calculated using 5-fold CV)
# Report the mean and stdDev of the error



logisticFctn <- function(dataPassed) {
  
  names(dataPassed) <- c("x", "y")
  
  # How many obs in the data
  gD.length <- nrow(dataPassed)
  
  # Initialize empty vector for predictions
  lgstcReg.preds <- vector("integer", gD.length)
  
  glmModel <- train(as.factor(y) ~ x,
                    data = dataPassed,
                    method = 'glm',
                    family = binomial,
                    trControl = trainControl(method = 'cv', number = 5))
  
  # Prediction error
  predError <- (1-(glmModel$results$Accuracy))
  
  return(predError)
}



#====== Errors for LogisticRegr Classifier at N=100 ======
# Initialize empty vector for N=100 errors
logstcErrors100 <- vector("numeric", 100)

# Calculating error for the N=100 samples
for(i in 1:100){
  logstcErrors100[i] <- logisticFctn(samples100[ , c(2*i-1,2*i)])
}
logstcErrors100
avgLreg100error <- mean(logstcErrors100)
avgLreg100error
sd(logstcErrors100)




#====== Errors for LogisticRegr Classifier at N=200 ======
# Initialize empty vector for N=200 errors
logstcErrors200 <- vector("numeric", 100)

# Calculating error for the N=200 samples
for(i in 1:100){
  logstcErrors200[i] <- logisticFctn(samples200[ , c(2*i-1,2*i)])
}
logstcErrors200
avgLreg200error <- mean(logstcErrors200)
avgLreg200error
sd(logstcErrors200)



#====== Errors for LogisticRegr Classifier at N=500 ======
# Initialize empty vector for N=500 errors
logstcErrors500 <- vector("numeric", 100)

# Calculating error for the N=500 samples
for(i in 1:100){
  logstcErrors500[i] <- logisticFctn(samples500[ , c(2*i-1,2*i)])
}
logstcErrors500
avgLreg500error <- mean(logstcErrors500)
avgLreg500error
sd(logstcErrors500)




#====== Errors for LogisticRegr Classifier at N=1000 ======
# Initialize empty vector for N=1000 errors
logstcErrors1000 <- vector("numeric", 100)

# Calculating error for the N=1000 samples
for(i in 1:100){
  logstcErrors1000[i] <- logisticFctn(samples1000[ , c(2*i-1,2*i)])
}
logstcErrors1000
avgLreg1000error <- mean(logstcErrors1000)
avgLreg1000error
sd(logstcErrors1000)















#===================================#
## PLOTTING SHIT ####################
#===================================#

N100_df <- data.frame(errors100, knnErrors100, logstcErrors100)
View(N100_df)
boxplot(N100_df,
        main = "N = 100 Samples",
        xlab = "Classifier (Bayes, KNN, and LReg respectively)",
        ylab = "Error Rate")


N200_df <- data.frame(errors200, knnErrors200, logstcErrors200)
View(N200_df)
boxplot(N200_df,
        main = "N = 200 Samples",
        xlab = "Classifier (Bayes, KNN, and LReg respectively)",
        ylab = "Error Rate")


N500_df <- data.frame(errors500, knnErrors500, logstcErrors500)
View(N500_df)
boxplot(N500_df,
        main = "N = 500 Samples",
        xlab = "Classifier (Bayes, KNN, and LReg respectively)",
        ylab = "Error Rate")


N1000_df <- data.frame(errors1000, knnErrors1000, logstcErrors1000)
View(N1000_df)
boxplot(N1000_df,
        main = "N = 1000 Samples",
        xlab = "Classifier (Bayes, KNN, and LReg respectively)",
        ylab = "Error Rate")


