import numpy as np
import time
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier)
from sklearn import preprocessing
from sklearn import clone
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import (cross_val_score, KFold)
from sklearn.tree import DecisionTreeClassifier
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

model1 = RandomForestClassifier(n_estimators=200,max_features=8) 
model2 = AdaBoostClassifier(n_estimators=200)   
model3 = DecisionTreeClassifier(max_features=10)
model4 = linear_model.LogisticRegression(C=0.1,penalty ='l2')        
RF = model1.fit(trainx,traindatay)
AB = model2.fit(trainx,traindatay)
deci = model3.fit(trainx,traindatay)
logreg = model4.fit(trainx,traindatay)  
ytree1=model1.predict(testx)
ytree2=model2.predict(testx) 
ytree3=model3.predict(testx)
regytrain = model4.predict(testx)
errtrain1 = mean_squared_error(ytree1,testdatay)
errtrain2 = mean_squared_error(ytree2,testdatay)  
errtrain3 = mean_squared_error(ytree3,testdatay)
errtrain4 = mean_squared_error(regytrain,testdatay) 
print "errtrainRF",errtrain1
print "errtrainAB",errtrain2  
print "errtraintree",errtrain3  
print errtrain4  