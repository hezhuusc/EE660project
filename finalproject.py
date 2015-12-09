import numpy as np
import time
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier)
from sklearn import preprocessing
from sklearn import clone
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import (cross_val_score, KFold)
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from operator import itemgetter
from sklearn.cross_validation import cross_val_score
from sklearn import linear_model

#Import dataset
traindata = np.loadtxt(open('DATA2.csv','r'),delimiter=",")

traindata_X=traindata[:,0:58]
traindata_Y=traindata[:,59]
T=1400

num1 = 31716
num2 = 3964
traindatay = np.zeros(num1);
for i in range(0,num1):
    if  traindata_Y[i]>T:
        traindatay[i]=1
        
print traindatay

#load the test dataset
testdata = np.loadtxt(open('testset1.csv','r'),delimiter=",")

testdata_X=testdata[:,0:58]
testdata_Y=testdata[:,59]
testdatay = np.zeros(num2);
for j in range(0,num2):
    if  testdata_Y[j]>T:
        testdatay[j]=1
        
print testdatay

#standardize the data
trainx=preprocessing.scale(traindata_X)
testx=preprocessing.scale(testdata_X)

kf = KFold(n=31716,n_folds = 5)


#model 
for train_index,test_index in kf:
    X_train, X_test = trainx[train_index], trainx[test_index]
    y_train, y_test = traindatay[train_index],traindatay[test_index]
    for i, n_estimators in enumerate((10,20,30,50,100,200,400)):
        for n, max_features in enumerate((2,5,8,10)): 
            model1 = RandomForestClassifier(n_estimators=n_estimators) 
            model2 = AdaBoostClassifier(n_estimators=n_estimators)    
            RF = model1.fit(X_train,y_train)
            AB = model2.fit(X_train,y_train)
            ytree1=model1.predict(X_test)
            #scores1 = cross_val_score(model1, X_train, y_train).mean()
            #print"scoreofRF", scores1
            ytree2=model2.predict(X_test)
            errtrain1 = mean_squared_error(ytree1,y_test)
            errtrain2 = mean_squared_error(ytree2,y_test)
            print"n_estimators", n_estimators
            print" feature", max_features
            print "errtrainRF",errtrain1
            print "errtrainAB",errtrain2
            print("---------------------")

    for j,max_features in enumerate((2,5,8,10)):
        deci_1 = DecisionTreeClassifier(max_features=max_features)
        deci = deci_1.fit(X_train,y_train)
        ytree3=deci_1.predict(X_test)
        errtrain3 = mean_squared_error(ytree3,y_test)
        print "errtraintree",errtrain3
        print " maxfeatures",max_features
        print("-----------------------")
    
    for m, C in enumerate((10,1, 0.01)):
        logreg = linear_model.LogisticRegression(C=C,penalty ='l2')
        model = logreg.fit(X_train,y_train)
        regytrain = logreg.predict(X_test)
        errortrain4 = mean_squared_error(regytrain,y_test)
        print errortrain4
        print("------------------------")
