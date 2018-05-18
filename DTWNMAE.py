import numpy as np
import pandas as pd
import array as array
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
ch = [[100,500,1000,5000,10000,50000,100000,500000,1000000]]

#import CSV
d=pd.read_csv('sum1.csv',sep=';')
#specify target and train coloumn
target_column = ['Noisy Target']
train_column = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']

X= d[train_column]
Y =d[target_column]

#Divide in chunks
for x in ch:
    
    for i in x:
        print(i)
        XX= d[target_column].iloc[0:i]
        YY= d[train_column].iloc[0:i]
    

        #split data 70/30
        XX_train,XX_test,yy_train,yy_test = train_test_split(XX,YY,test_size = 0.30,random_state=42)

        #Create decision tree model
        clf = tree.DecisionTreeRegressor()

        # Train the model using the training sets
        clf.fit(XX, YY)

        # Make predictions using the testing set
        yy_pred=clf.predict(XX_test)
        print(yy_pred[0:10])
        print(yy_test[0:10])
        #print(clf.predict(X_test[0:10]))
        #print(y_test[0:10])
        #print(clf.score(X_test,y_test))
        

        # mean absolute error metric for 70/30
        print("Mean Absolute Error : %.5f" % mean_absolute_error(yy_test,yy_pred))
        