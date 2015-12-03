import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.metrics import mean_squared_error
# Load the training dataset
traindata = np.loadtxt(open('trainset.csv','r'),delimiter=",")
print(traindata.shape)

traindata_X=traindata[:,0:58]
traindata_Y=traindata[:,59]

#load the test dataset
testdata = np.loadtxt(open('testset1.csv','r'),delimiter=",")

testdata_X=testdata[:,0:58]
testdata_Y=testdata[:,59]

#normalize the data
normalized_trainX=preprocessing.normalize(traindata_X)
normalized_trainY=preprocessing.normalize(traindata_Y)
normalized_testX=preprocessing.normalize(testdata_X)
normalized_testY=preprocessing.normalize(testdata_Y)

#standardize the data
trainx=preprocessing.scale(traindata_X)
trainy=preprocessing.scale(traindata_Y)
testx=preprocessing.scale(testdata_X)
testy=preprocessing.scale(testdata_Y)


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(trainx, trainy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      )

# predict
regr_y=regr.predict(testx)
plt.plot(testx,regr_y,".b")
plt.plot(testx,testy,".r")
plt.show()

# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean square error
#print("Residual sum of squares: %.2f"
      #% np.mean((regr.predict(testx) - testy) ** 2))
# The MSE

mean_squared_error(regr_y,testy)

## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % regr.score(testx, testy))

##Plot outputs
#plt.scatter(testx, testy,  color='black')
#plt.plot(testx, regr.predict(testy), color='blue',
#        linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()
