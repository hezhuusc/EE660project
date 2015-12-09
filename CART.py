#Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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


deci_1 = DecisionTreeClassifier(max_features=5)
deci_2 = DecisionTreeClassifier(max_features=10)
deci_1.fit(trainx,traindatay)
deci_2.fit(trainx,traindatay)

#Predict
ypre1 = deci_1.predict(testx)
ypre2 = deci_2.predict(testx)
print(ypre1)
print(ypre2)

###error rate
errtest1 = mean_squared_error(ypre1,testdatay)
print(errtest1)
errtest2 = mean_squared_error(ypre2,testdatay)
print(errtest2)