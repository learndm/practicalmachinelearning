# first clear your workspace

rm(list = ls())

# set the working directory 

setwd("~/Desktop/PRACTICAL_MACHINE_LEARNING_PROJECT")

# set seed before start your machine learning model building

set.seed(1000)

# download training data set

training_data_url <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
destination_file_training <- 'pml-training.csv'

download.file(training_data_url, destination_file_training, method="curl", quiet = TRUE)

if (file.exists(destination_file_training)){
    cat("Training data set is available")
} else {
   stop("Training data set is not available, please check the Download !!")
}

# download test data set 

test_data_url <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'

destination_file_testing <- 'pml-testing.csv'

download.file(test_data_url , destination_file_testing, method="curl", quiet = TRUE)

if (file.exists(destination_file_testing)){
    cat("Testing data set is available")
} else {
   stop("Testing data set is not available, please check the Download !!")
}

# load caret package

library(caret)

# load training raw data

training_raw_data <- read.csv(file="pml-training.csv", header=TRUE, as.is = TRUE, stringsAsFactors = FALSE, sep=',', na.strings=c('NA','','#DIV/0!'))

# load testing raw data

testing_raw_data <- read.csv(file="pml-testing.csv", header=TRUE, as.is = TRUE, stringsAsFactors = FALSE, sep=',', na.strings=c('NA','','#DIV/0!'))

# read classe as fcator
training_raw_data$classe <- as.factor(training_raw_data$classe)  

# inspect the traing data to identify NA and useless variables for the purpose of the prediction

summary(training_raw_data)
head(training_raw_data)

# data cleaning on training data set

NAidx <- apply(training_raw_data,2,function(x) {sum(is.na(x))}) 
training_raw_data <- training_raw_data[,which(NAidx == 0)]

# data cleaning on testing data set

NAidx <- apply(testing_raw_data,2,function(x) {sum(is.na(x))}) 
testing_raw_data<- testing_raw_data[,which(NAidx == 0)]

# preprocessing of Predictors 

vidx <- which(lapply(training_raw_data, class) %in% "numeric")

preProcessData <-preProcess(training_raw_data[,vidx],method=c('knnImpute', 'center', 'scale'))
trainModel <- predict(preProcessData, training_raw_data[,vidx])
trainModel$classe <- training_raw_data$classe

testModel <-predict(preProcessData ,testing_raw_data[,vidx])

# remove near zero values, if any
nzv <- nearZeroVar(trainModel,saveMetrics=TRUE)
trainModel<- trainModel[,nzv$nzv==FALSE]

nzv <- nearZeroVar(testModel,saveMetrics=TRUE)
testModel <- testModel[,nzv$nzv==FALSE]

# create a data partition for cross validation

inTrain = createDataPartition(trainModel$classe, p = 3/4, list=FALSE)
training = trainModel[inTrain,]
crossValidation = trainModel[-inTrain,]

# Train the model using Random Forest method because of its accuracy

modelFit <- train(classe ~., method="rf", data=training, trControl=trainControl(method='cv'), number=5, allowParallel=TRUE )

# Check the model's accuracy on training set 

trainingPrediction <- predict(modelFit, training)
confusionMatrix(trainingPrediction, training$classe)

# Check the model's accuracy on training set 

crossValidationPrediction <- predict(modelFit, crossValidation)
confusionMatrix(crossValidationPrediction, crossValidation$classe)

# Final result on testing data
testingPrediction <- predict(modelFit, testModel)
testingPrediction
