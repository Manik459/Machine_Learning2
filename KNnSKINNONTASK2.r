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

knn<-function(list,k){
  n=nrow(list)
  if (n<=k) stop("k can not be more than n-1")
  neigh<- matrix(0,nrow=n,ncol=k)
  for(i in 1:n){
    dist<-matrix(0,ncol=2,nrow=n)
    for (j in 1:n){
      dist[j,1]<-j
      dist[j,2]<-sum((list[i,]-list[j,])^2)
      #dist[j,2]<-dtw(list[i,],list[j,])$b
    }
    sorted<-dist[order(dist[,2]),]
    neigh[i,]<-sorted[2:(k+1),1]
  }
  return(neigh)
}
RMSE <- sqrt(mean((y_test - y_pred)^2))
print(paste("RMSE = ",RMSE))

MAE<-mean(abs(y_test - y_pred))
print(paste("MAE = ",MAE))