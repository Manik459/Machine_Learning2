library(caTools)
library(caret)
library(rpart)
# Importing the dataset

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#For skin/non-skin data set
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
dataset = read.csv('skinnonskin.csv')
options(digits = 15)

# --------------------------------------------------------------------------------
# for 10 - fold
# --------------------------------------------------------------------------------

# define training control
train_control<- trainControl(method="cv", number=10, savePredictions = TRUE)

# train the model 
model<- train(B~., data=dataset, trControl=train_control, method="rpart")


RMSE <- sqrt(mean((model$pred$obs - model$pred$pred)^2))
print(paste("RMSE = ",RMSE))

MAE<-mean(abs(model$pred$obs - model$pred$pred))
print(paste("MAE = ",MAE))

# --------------------------------------------------------------------------------
# for 70/30
# --------------------------------------------------------------------------------

set.seed(123)
split = sample.split(dataset$B, SplitRatio = 0.7)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = B ~ .,
               data = training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)
y_test <- subset(dataset$B, split==FALSE)

RMSE <- sqrt(mean((y_test - y_pred)^2))
print(paste("RMSE = ",RMSE))

MAE<-mean(abs(y_test - y_pred))
print(paste("MAE = ",MAE))