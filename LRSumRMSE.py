import numpy as np
import pandas as pd
import array as array
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from math
 import sqrt
ch = [[100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000]]

#import CSV
d=pd.read_csv('sum1.csv',sep=';')
#specify target and train coloumn
target_column = ['Target']
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

        #Create linear regression object
        clf = linear_model.LinearRegression()

        # Train the model using the training sets
        clf.fit(XX, YY)

        # Make predictions using the testing set
        yy_pred=clf.predict(XX_test)
        print(yy_pred[0:10])
        print(yy_test[0:10])
        #print(clf.predict(X_test[0:10]))
        #print(y_test[0:10])
        #print(clf.score(X_test,y_test))
        #print('Coefficients: \n', clf.coef_)

        # mean squared error metric for 70/30
        print("Mean squared Error : %.5f" % sqrt(mean_squared_error(yy_test,yy_pred)))
        