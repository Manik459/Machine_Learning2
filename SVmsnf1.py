from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

splits = [[100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000]]

d = pd.read_csv('sum1.csv',sep=';')
target_column = ['Noisy Target Class']
train_column = ['Instance','Feature 1','Feature 2','Feature 3','Feature 4','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']

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