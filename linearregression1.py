import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the training dataset
traindata = np.loadtxt(open('trainset.csv','r'),delimiter=",")
print(traindata.shape)

traindata_X=traindata[:,0:58]
traindata_Y=traindata[:,59]

#load the test dataset
testdata = np.loadtxt(open('testset1.csv','r'),delimiter=",")

testdata_X=testdata[:,0:58]
testdata_Y=testdata[:,59]

# Split the data into training/testing sets
#traindata_X_train = traindata_X[:-20]
#traindata_X_test = traindata_X[-20:]

# Split the targets into training/testing sets
#traindata_Y_train = traindata_Y[:-20]
#traindata_Y_test = traindata_Y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(traindata_X, traindata_Y)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(testdata_X) - testdata_Y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(testdata_X, testdata_Y))

# Plot outputs
#plt.scatter(traindata_X_test, traindata_Y_test,  color='black')
#plt.plot(traindata_X_test, regr.predict(traindata_X_test), color='blue',
#        linewidth=3)

#plt.xticks(())
#plt.yticks(())

plt.show()
