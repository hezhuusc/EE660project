#Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Import dataset
traindata = np.loadtxt(open('trainset.csv','r'),delimiter=",")
print(traindata.shape)

traindata_X=traindata[:,0:58]
traindata_Y=traindata[:,59]

#load the test dataset
testdata = np.loadtxt(open('testset1.csv','r'),delimiter=",")

testdata_X=testdata[:,0:58]
testdata_Y=testdata[:,59]

#standardize the data
trainx=preprocessing.scale(traindata_X)
trainy=preprocessing.scale(traindata_Y)
testx=preprocessing.scale(testdata_X)
testy=preprocessing.scale(testdata_Y)

regr_1 = DecisionTreeRegressor(max_depth=10)
regr_2 = DecisionTreeRegressor(max_depth=50)
regr_1.fit(trainx,trainy)
regr_2.fit(trainx,trainy)

#Predict
ypre1 = regr_1.predict(testx)
ypre2 = regr_2.predict(testx)
print(ypre1)
print(ypre2)

###error rate
errtest1 = mean_squared_error(ypre1,testy)
print(errtest1)
errtest2 = mean_squared_error(ypre2,testy)
print(errtest2)