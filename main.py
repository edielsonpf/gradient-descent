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
minThreshold=0.001
maxEpoc = 10

BachHandle = BatchClass()
Model,numEpoc, Lsq = BachHandle.train(stepLen, minThreshold, maxEpoc, XTrain, YTrain)

x = [i+1 for i in range(numEpoc)]
plt.plot(x, Lsq)
plt.xlabel('Number of epochs')
plt.ylabel('Least squares')
plt.title('Training history')
plt.show()