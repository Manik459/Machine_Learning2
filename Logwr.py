from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split

d=pd.read_csv('winered.csv',sep=';')
target_column =  ['quality']
train_column = ['fixed acidity','residual sugar','density','pH','alcohol']

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
    