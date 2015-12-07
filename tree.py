import numpy as np
import time
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier)
from sklearn import preprocessing
from sklearn import clone
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV

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


#model 
n_estimators = 10
models = [RandomForestClassifier(n_estimators=n_estimators),
          AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                             n_estimators=n_estimators)]
print n_estimators
for model in models:
        start = time.clock()
        clf = clone (model)
        clf = model.fit(trainx,traindatay)
        yre = model.predict(testx)
        errtest = mean_squared_error(yre,testdatay)
        print errtest
        scores = clf.score(trainx,traindatay)
        print scores
        stop = time.clock()
        print "run in : %f s"%(stop - start)
n_estimators = 20
print n_estimators
for model in models:
        start = time.clock()
        clf = clone (model)
        clf = model.fit(trainx,traindatay)
        yre = model.predict(testx)
        errtest = mean_squared_error(yre,testdatay)
        print errtest
        scores = clf.score(trainx,traindatay)
        print scores
        stop = time.clock()
        print "run in : %f s"%(stop - start)
n_estimators = 30
print n_estimators
for model in models:
        start = time.clock()
        clf = clone (model)
        clf = model.fit(trainx,traindatay)
        yre = model.predict(testx)
        errtest = mean_squared_error(yre,testdatay)
        print errtest
        scores = clf.score(trainx,traindatay)
n_estimators = 50
print n_estimators
for model in models:
        start = time.clock()
        clf = clone (model)
        clf = model.fit(trainx,traindatay)
        yre = model.predict(testx)
        errtest = mean_squared_error(yre,testdatay)
        print errtest
        scores = clf.score(trainx,traindatay)
n_estimators = 50
print n_estimators
for model in models:
        start = time.clock()
        clf = clone (model)
        clf = model.fit(trainx,traindatay)
        yre = model.predict(testx)
        errtest = mean_squared_error(yre,testdatay)
        print errtest
        scores = clf.score(trainx,traindatay)

        print scores
        stop = time.clock()
        print "run in : %f s"%(stop - start)