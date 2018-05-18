import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split

d=pd.read_csv('sum1.csv',sep=';')
target_column =  ['Noisy Target Class']
train_column = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']

X = d[train_column]
Y = d[target_column]
y_true = d[target_column]
a = np.array([100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000])




#clf=linear_model.LogisticRegression()
#clf.fit(X_train,Y_train)
#print(clf.predict(X_test[0:5]))
#print(Y_test[0:5])
#print(clf.score(X_test[0:5],Y_test[0:5]))
#print(clf.coef_)

for i in a:
    #print(i)
    XX = X.iloc[0:i] #clustering loop for X
    YY=  Y.iloc[0:i]
    X_train,X_test,Y_train,Y_test = train_test_split(XX,YY,test_size = 0.30, random_state = 20)
    #print(XX)
    #print(YY)
    clf=linear_model.LogisticRegression()
    clf.fit(X_train,Y_train)
    #print(clf.predict(XX))
    y_pred=clf.predict(X_test)
    #print(clf.score(XX,YY))
    
    AA= accuracy_score(Y_test, y_pred)
    print('Accuracy Score:')
    print(AA)
    
    BB = f1_score(Y_test, y_pred, average='micro')
    print('F1 Score')
    print(BB)
