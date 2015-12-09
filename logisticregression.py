import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score
from sklearn import linear_model
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


#logistic regression
for C in enumerate((100,1, 0.01)):
    logreg = linear_model.LogisticRegression(C=C,penalty ='l2')
    model = logreg.fit(trainx,traindatay)
    regytrain = logreg.predict(trainx)
    errortrain = mean_squared_error(regytrain,traindatay)
    print errortrain