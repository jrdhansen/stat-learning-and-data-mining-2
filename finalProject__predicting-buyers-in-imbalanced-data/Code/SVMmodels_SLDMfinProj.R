## STAT LEARNING 2, FINAL PROJECT
## Last modified by: Jared Hansen
## Thursday, 8:00 PM, 12/06/2018



# Necessary libraries
library(caret)        # Used for upsampling, will be used for model development
library("e1071")      # Used for training SVM models
library(dplyr)        # Used for data manipulation



#===============================================================================
#======= DATA PRE-PROCESSING ===================================================
#===============================================================================

# Read in the data
#+++++++ CHANGE THIS LINE ON YOUR MACHINE ++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
bankFilePath <- paste0("C:/Users/jrdha/OneDrive/Desktop/USU_Fa2018/",
                       "Moon__SLDM2/finalProject/Kaggle_bank_SLDM_data/",
                       "bank-full/bank-full.csv")
bank <- read.csv(bankFilePath, header = TRUE)

# Make sure that columns with numeric values have the right data type(s).
sapply(bank, class)

# Split into training, test, and validation data, SET SEED for reproducibility
set.seed(2345)


# First thing to check: is there a class imbalance? I'd assume yes: most people
# aren't going to buy a product that's marketed to them over the phone.
# Let's check this assumption.
# bank$response <- ifelse(bank$y == "no", 0, 1)
# prop_of_1s <- (sum(bank$response) / nrow(bank))
# prop_of_1s

# We can see that we have about 11.7% 1's (people who do buy), and the rest are 
# 0's (people who don't buy). We'll account for this by adjusting our training
# data below (upsampling).


# Doing an 80% training +  20% test split (parameters will be tuned w/CV)
lengthTrain_80 <- round(nrow(bank) * 0.8)
lengthTest_20 <- nrow(bank) - lengthTrain_80

# Make indices for training data subsetting
set.seed(2345)
train_ind <- sample(seq_len(nrow(bank)), size = lengthTrain_80)

# Split up the data into training and test
# I've named this 'train_80_data' since some of our training data will be 80%
# of the original data, and some of the training data will be 60% of the orig.
train_80_Data <- bank[train_ind, ]
test_20_Data <- bank[-train_ind, ]


# We "technically" shouldn't do this, but just to be safe, let's check to see
# that the training, validation, and test data each have roughly the same
# proportion of 0's and 1's as the original data does (11% 1's, 89% 0's).
# As we'll see by running the code, below, this checks out. They're all really
# close to having about 11.7% 1's, so we did the random splitting right.
#
# train_80_Data$response <- ifelse(train_80_Data$y == "no", 0, 1)
# prop_of_1s <- (sum(train_80_Data$response) / nrow(train_80_Data))
# prop_of_1s
# 
# 
# test_20_Data$response <- ifelse(test_20_Data$y == "no", 0, 1)
# prop_of_1s <- (sum(test_20_Data$response) / nrow(test_20_Data))
# prop_of_1s





#============= DOWNSAMPLING ====================================================

# Index of the column of the response variable (it's named "y" in orig data)
colOfResp <- grep("^y$", colnames(train_80_Data))

# This gives upsampled training data for the 80% original
up_train_80 <- upSample(x = train_80_Data[ , -colOfResp],
                        y = train_80_Data$y)

# This gives downsampled training data for the 80% original
down_train_80 <- downSample(x = train_80_Data[ , -colOfResp],
                            y = train_80_Data$y)




#===============================================================================
#===================== SVM =====================================================
#===============================================================================


# SVM is taking forever to run, so let's get things working on a smaller 
# data set first.
len_smallData <- 500

# Make indices for training data subsetting
set.seed(2345)
ind_smallData <- sample(seq_len(nrow(down_train_80)), size = len_smallData)

# Split up the data into training and test
# I've named this 'train_80_data' since some of our training data will be 80%
# of the original data, and some of the training data will be 60% of the orig.
smallData <- down_train_80[ind_smallData, ]







#========== TUNING LINEAR-KERNEL MODEL =========================================
#===============================================================================


# Tuning LINEAR KERNEL: Round 1 (used this to get a set of values that did well)
# Ran it by changing cost = 1.digit*10^(-3:3) where I set digit = 0,1,2,....,9
set.seed(2345)
svm_tune_linear <- tune.svm(Class~.,
                            data = smallData,
                            kernel = "linear", 
                            cost = seq(0.5,6,0.25))
print(svm_tune_linear)




# There are 1008 people in our test data who did say "yes" to the marketing 
# campaign. These are the ones who we care about identifying accurately.
# Thus, the TRUE measure of each model is how well we identify these individuals
# from the test data.
num_yes_inTest <- nrow(filter(test_20_Data, y == "yes"))




#===== LINEAR KERNEL PREDICTION USING ALL OF THE TRAINING DATA =================
train_80_Data$y <- as.factor(train_80_Data$y)
svm_model <- svm(y ~., train_80_Data,
                 kernel = "linear",
                 cost = 1.25)
test_20_Data$pred <- predict(svm_model, test_20_Data)
test_20_Data$correct <- with(test_20_Data,
                             ifelse(test_20_Data$y == pred, 1, 0))
# Keep this dataset for further analysis (seeing what was being predicted wrong)
testPreds_allTrain <- test_20_Data
testPreds_allWRONG <- filter(testPreds_allTrain, correct == 0)
error_all <- 931/9042
# Analysis of incorrect predictions:
num_predNo_wasYes <- nrow(filter(testPreds_allWRONG, pred == "no"))
pct_buyers_all <- 1 - (num_predNo_wasYes / num_yes_inTest)
pct_buyers_all
# Generate the test error
# Total number of correct classifications
totalCorrect <- sum(test_20_Data$correct)
testError_SVM <- 1 - (totalCorrect / nrow(test_20_Data))
testError_SVM
# Remove the "correct" column for the SVM portion
test_20_Data <- subset(test_20_Data, select =  -c(correct, pred))




#===== LINEAR KERNEL PREDICTION USING UPSAMPLED TRAINING DATA ==================
up_train_80$Class <- as.factor(up_train_80$Class)
svm_model <- svm(Class ~., up_train_80,
                 kernel = "linear",
                 cost = 1.25)
test_20_Data$pred <- predict(svm_model, test_20_Data)
test_20_Data$correct <- with(test_20_Data,
                             ifelse(test_20_Data$y == pred, 1, 0))
# Keep this dataset for further analysis (seeing what was being predicted wrong)
testPreds_upSampled <- test_20_Data
testPreds_upWRONG <- filter(testPreds_upSampled, correct == 0)
error_up <- 1525/9042
# Analysis of incorrect predictions:
num_predNo_wasYes <- nrow(filter(testPreds_upWRONG, pred == "no"))
pct_buyers_up <- 1 - (num_predNo_wasYes / num_yes_inTest)
pct_buyers_up
# Generate the test error
# Total number of correct classifications
totalCorrect <- sum(test_20_Data$correct)
testError_SVM <- 1 - (totalCorrect / nrow(test_20_Data))
testError_SVM
# Remove the "correct" column for the SVM portion
test_20_Data <- subset(test_20_Data, select =  -c(correct, pred))




#===== LINEAR KERNEL PREDICTION USING DOWNSAMPLED TRAINING DATA ================
down_train_80$Class <- as.factor(down_train_80$Class)
svm_model <- svm(Class ~., down_train_80,
                 kernel = "linear",
                 cost = 1.25)
test_20_Data$pred <- predict(svm_model, test_20_Data)
test_20_Data$correct <- with(test_20_Data,
                             ifelse(test_20_Data$y == pred, 1, 0))
# Keep this dataset for further analysis (seeing what was being predicted wrong)
testPreds_downSampled <- test_20_Data
testPreds_downWRONG <- filter(testPreds_downSampled, correct == 0)
error_down <- 1483/9042
# Analysis of incorrect predictions:
num_predNo_wasYes <- nrow(filter(testPreds_downWRONG, pred == "no"))
pct_buyers_down <- 1 - (num_predNo_wasYes / num_yes_inTest)
pct_buyers_down
# Generate the test error
# Total number of correct classifications
totalCorrect <- sum(test_20_Data$correct)
testError_SVM <- 1 - (totalCorrect / nrow(test_20_Data))
testError_SVM
# Remove the "correct" column for the SVM portion
test_20_Data <- subset(test_20_Data, select =  -c(correct, pred))













#========== TUNING GAUSSIAN-KERNEL MODEL =======================================
#===============================================================================

# INITIAL PASS (leave base values at 10)
set.seed(2345)
svm_tune_gaussian <- tune.svm(Class~., data = smallData,
                              cost = 1.0 * 10^(-3:7),
                              gamma = 1.0 * 10^(-5:2))
print(svm_tune_gaussian)
# Gives cost = 1000 and gamma = 0.001 as best values



#===== GAUSSIAN KERNEL PREDICTION USING ALL OF THE TRAINING DATA ===============
train_80_Data$y <- as.factor(train_80_Data$y)
svm_model <- svm(y ~., train_80_Data,
                 cost = 1000,
                 gamma = 0.001)
test_20_Data$pred <- predict(svm_model, test_20_Data)
test_20_Data$correct <- with(test_20_Data,
                             ifelse(test_20_Data$y == pred, 1, 0))
# Keep this dataset for further analysis (seeing what was being predicted wrong)
testPreds_allTrain <- test_20_Data
testPreds_allWRONG <- filter(testPreds_allTrain, correct == 0)
error_all <- 892/9042
# Analysis of incorrect predictions:
num_predNo_wasYes <- nrow(filter(testPreds_allWRONG, pred == "no"))
pct_buyers_all <- 1 - (num_predNo_wasYes / num_yes_inTest)
pct_buyers_all
# Generate the test error
# Total number of correct classifications
totalCorrect <- sum(test_20_Data$correct)
testError_SVM <- 1 - (totalCorrect / nrow(test_20_Data))
testError_SVM
# Remove the "correct" column for the SVM portion
test_20_Data <- subset(test_20_Data, select =  -c(correct, pred))



#===== GAUSSIAN KERNEL PREDICTION USING UPSAMPLED TRAINING DATA ================
up_train_80$Class <- as.factor(up_train_80$Class)
svm_model <- svm(Class ~., up_train_80,
                 cost = 1000,
                 gamma = 0.001)
test_20_Data$pred <- predict(svm_model, test_20_Data)
test_20_Data$correct <- with(test_20_Data,
                             ifelse(test_20_Data$y == pred, 1, 0))
# Keep this dataset for further analysis (seeing what was being predicted wrong)
testPreds_upSampled <- test_20_Data
testPreds_upWRONG <- filter(testPreds_upSampled, correct == 0)
error_up <- 1477/9042
# Analysis of incorrect predictions:
num_predNo_wasYes <- nrow(filter(testPreds_upWRONG, pred == "no"))
pct_buyers_up <- 1 - (num_predNo_wasYes / num_yes_inTest)
pct_buyers_up
# Generate the test error
# Total number of correct classifications
totalCorrect <- sum(test_20_Data$correct)
testError_SVM <- 1 - (totalCorrect / nrow(test_20_Data))
testError_SVM
# Remove the "correct" column for the SVM portion
test_20_Data <- subset(test_20_Data, select =  -c(correct, pred))



#===== GAUSSIAN KERNEL PREDICTION USING DOWNSAMPLED TRAINING DATA ==============
down_train_80$Class <- as.factor(down_train_80$Class)
svm_model <- svm(Class ~., down_train_80,
                 cost = 1000,
                 gamma = 0.001)
test_20_Data$pred <- predict(svm_model, test_20_Data)
test_20_Data$correct <- with(test_20_Data,
                             ifelse(test_20_Data$y == pred, 1, 0))
# Keep this dataset for further analysis (seeing what was being predicted wrong)
testPreds_downSampled <- test_20_Data
testPreds_downWRONG <- filter(testPreds_downSampled, correct == 0)
error_down <- 1486/9042
# Analysis of incorrect predictions:
num_predNo_wasYes <- nrow(filter(testPreds_downWRONG, pred == "no"))
pct_buyers_down <- 1 - (num_predNo_wasYes / num_yes_inTest)
pct_buyers_down
# Generate the test error
# Total number of correct classifications
totalCorrect <- sum(test_20_Data$correct)
testError_SVM <- 1 - (totalCorrect / nrow(test_20_Data))
testError_SVM
# Remove the "correct" column for the SVM portion
test_20_Data <- subset(test_20_Data, select =  -c(correct, pred))
