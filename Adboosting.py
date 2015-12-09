import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier

#Import dataset
traindata = np.loadtxt(open('trainset.csv','r'),delimiter=",")

traindata_X=traindata[:,0:58]
traindata_Y=traindata[:,59]
T=1400

num = 3964
traindatay = np.zeros(num);
for i in range(0,num):
    if  traindata_Y[i]>T:
        traindatay[i]=1
        
print traindatay

#load the test dataset
testdata = np.loadtxt(open('testset1.csv','r'),delimiter=",")

testdata_X=testdata[:,0:58]
testdata_Y=testdata[:,59]
testdatay = np.zeros(num);
for j in range(0,num):
    if  testdata_Y[j]>T:
        testdatay[j]=1
        
print testdatay

#standardize the data
trainx=preprocessing.scale(traindata_X)
testx=preprocessing.scale(testdata_X)


#RF1
n_estimators = 20
model1 = AdaBoostClassifier(n_estimators=n_estimators)
model1.fit(trainx,traindatay)
ytree=model1.predict(testx)
print ytree

errtest = mean_squared_error(ytree,testdatay)
print " errtest1",errtest
scores = cross_val_score(model1, testx, testdatay)
scores.mean()

#
a=model1.feature_importances_
print a