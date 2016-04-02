from Tools import LoadCSV
from GradientDescent.Batch import BatchClass
import matplotlib.pyplot as plt

###
#
#    Observations must be distributed by rows and features by columns
# 
###

#Loading data from CSV file
XTrain = LoadCSV('data/Xtrain.csv', 50)
numObs = len(XTrain)
print ('Number of training observations: %s'%numObs) 
numFeat = len(XTrain.T)
print ('Number of features: %s'%numFeat)

YTrain = LoadCSV('data/Ytrain.csv', 1)

stepLen = 1.0/32
minThreshold=1e-5
maxEpoc = 30

BachHandle = BatchClass()
Model,numEpoc, Lsq = BachHandle.train(stepLen, minThreshold, maxEpoc, XTrain, YTrain)

x = [i+1 for i in range(numEpoc)]
plt.plot(x, Lsq)
plt.xlabel('Number of epochs')
plt.ylabel('Least squares')
plt.title('Training history')
plt.show()


#Loading data from CSV file
XTest = LoadCSV('data/Xtest.csv', 50)
numObs = len(XTest)
print ('Number of training observations: %s'%numObs) 
numFeat = len(XTest.T)
print ('Number of features: %s'%numFeat)
YTest = LoadCSV('data/Ytest.csv', 1)

percentage,success,errors = BachHandle.valid(Model, XTest, YTest)
print('Number of observations classified correctly: %g'%success)
print('Number of observations classified wrongly: %g'%errors)
print('Percentage correctly classified: %g '%percentage)
