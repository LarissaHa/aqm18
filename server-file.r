##################################################################

# Server R-File
# Final Submission for AQM
# Larissa Haas

##################################################################

##################################################################
# Setting up the Working Environment
##################################################################

#install.packages("h2o")
#install.packages("ROSE")
#install.packages("e1071")
library("h2o")
library("ROSE")
library("e1071")

ACC <- function(table){
    tp <- table[2,2]
    tn <- table[1,1]
    fp <- table[2,1]
    fn <- table[1,2]
    acc <- (tn + tp) / (tn + tp + fn + fp)
    return(acc)
}

##################################################################
# Starting H2O and Loading the Data
##################################################################

c1 <- h2o.init(ip='localhost', nthreads=-1,
                min_mem_size='7G', max_mem_size='14G')

data.scaled <-read.table("ess-scaled.csv", header=TRUE, sep=",") 
data.scaled <- data.scaled[,-c(1,3,23)]

# Splitting in Train and Test
smp_size <- floor(0.9 * nrow(data.scaled))
set.seed(1234)
train_ind <- sample(seq_len(nrow(data.scaled)), size = smp_size)
train <- data.scaled[train_ind, ]
test <- data.scaled[-train_ind, ]

# Splitting up in X and y
train.X <- train[, -1]
train.y <- train[, 1]
train.y <- factor(train.y, levels = 0:1)
train.X.bal <- train[, -1]
train.y.bal <- train[, 1]
train.y.bal <- factor(train.y, levels = 0:1)
test.X <- test[, -1]
test.y <- test[, 1]
test.y <- factor(test.y, levels = 0:1)

# Transforming the Data into H2O Data Frames
train <- cbind(train.X, Outcome = factor(train.y))
train.bal <- cbind(train.X.bal, Outcome = factor(train.y.bal))
test <- cbind(test.X, Outcome = factor(test.y))
h2oactivity.train <- as.h2o(train)
h2oactivity.train.bal <- as.h2o(train.bal)
h2oactivity.test <- as.h2o( test)

##################################################################
# Random Forest Grids
##################################################################

# 1st grid
g <- h2o.grid("randomForest", 
              hyper_params = list(
                ntrees = c(5, 10, 15, 20, 30, 40, 50), 
                max_depth = c(10, 20, 30)),
              x = colnames(train.X), y = colnames(train.y), 
              training_frame = h2oactivity.train,
              nfolds = 10)
g

rf1 <- h2o.randomForest(        
  training_frame = h2oactivity.train,        
  validation_frame = h2oactivity.test,      
  x = colnames(train.X),                        
  y = colnames(train.y),                         
  model_id = "rf_covType_v1",    
  ntrees = 50,            
  max_depth = 20,
  stopping_rounds = 2,           
  score_each_iteration = T,     
  seed = 1000000) 
rf1

##################################################################

# 2nd grid
g2 <- h2o.grid("randomForest", 
              hyper_params = list(
                ntrees = c(50, 100, 150, 200), 
                max_depth = c(10, 20, 30)),
              x = colnames(train.X), y = colnames(train.y), 
              training_frame = h2oactivity.train,
              nfolds = 10)
g2

rf2 <- h2o.randomForest(        
  training_frame = h2oactivity.train,        
  validation_frame = h2oactivity.test,      
  x = colnames(train.X),                        
  y = colnames(train.y),                          
  model_id = "rf_covType_v1",    
  ntrees = 150,            
  max_depth = 20,
  stopping_rounds = 2,           
  score_each_iteration = T,     
  seed = 1000000) 

rf3 <- h2o.randomForest(        
  training_frame = h2oactivity.train,        
  validation_frame = h2oactivity.test,      
  x = colnames(train.X),                        
  y = colnames(train.y),                         
  model_id = "rf_covType_v2",    
  ntrees = 10,            
  max_depth = 10,
  stopping_rounds = 2,           
  score_each_iteration = T,     
  seed = 1000000) 

##################################################################
# SVM Models
##################################################################

# Unbalanced Model
model_svm <- svm(train.y ~ as.matrix(train.X))
prediction <- predict(model_svm, as.matrix(test.X))
prediction2 <- predict(model_svm, as.matrix(train.X))

ACC(table(prediction[1:9704], test.y))
# 0.6082028
table(prediction[1:9704], test.y)

ACC(table(prediction2, train.y))
# 0.7042873
table(prediction2, train.y)

# Balanced Model
model_svm <- svm(train.y.bal ~ as.matrix(train.X.bal))
prediction <- predict(model_svm, as.matrix(test.X))
prediction2 <- predict(model_svm, as.matrix(train.X.bal))

ACC(table(prediction[1:9704], test.y))
# 0.6082028
table(prediction[1:9704], test.y)

ACC(table(prediction2, train.y))
# 0.7042873
table(prediction2, train.y)

# Attempt to Tune SVM
tuneResult <- tune(svm, train.y ~ as.matrix(train.X),
                   ranges = list(epsilon = seq(0,0.2,0.1), cost = 2^(2:9))
) 
# --> failed!
print(tuneResult)

##################################################################