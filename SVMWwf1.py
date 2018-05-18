from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

splits = [[100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000]]

d = pd.read_csv('winewhite.csv',sep=';')
target_column = ['quality']
train_column = ['fixed acidity','residual sugar','density','pH','alcohol']

print(list(d))
levels = d['quality'].unique()
print(levels)
print('Splitting')
for x in splits:
	for i in x:
		X= d[train_column].iloc[0:i]
		Y = d[target_column].iloc[0:i]
		print(i)
		X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.33,random_state=42)

		clf = svm.LinearSVC(max_iter = 75)
		print('Fitting')
		clf.fit(X_train,y_train)
		y_pred = clf.predict(X_test)
		print(y_pred.shape)
		print(y_test.shape)
		cf = metrics.f1_score(y_test,y_pred,average='micro')
		print(cf)
		