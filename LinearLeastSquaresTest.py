from GradientDescent.LinearLeastSquares import LLSClass
import matplotlib.pyplot as plt

def LinearLeastSquaresTest(Xtrain,Ytrain,Xtest,Ytest):
    
    ###
    #
    #    Observations must be distributed by rows and features by columns
    # 
    ###
    
    #Loading data from CSV file
    numObs = len(Xtrain)
    print ('Number of training observations: %s'%numObs) 
    numFeat = len(Xtrain.T)
    print ('Number of features: %s'%numFeat)
    
    
    ###########################
    #
    #       BATCH
    #
    ###########################
    
    print('Batch training procedure')
    stepLen = 1.0/32
    minThreshold=1e-5
    maxEpoc = 10
    
    LLSHandle = LLSClass()
    print('Starting training procedure...')
    Model,numEpoc, Lsq = LLSHandle.trainBatch(stepLen, minThreshold, maxEpoc, Xtrain, Ytrain)
    print('Done!\n')
    x = [i+1 for i in range(numEpoc)]
    plt.plot(x, Lsq)
    plt.xlabel('Number of epochs')
    plt.ylabel('Least squares')
    plt.title('Training history')
    plt.show()
    
    
    #Loading data from CSV file
    numObs = len(Xtest)
    print ('Number of training observations: %s'%numObs) 
    numFeat = len(Xtest.T)
    print ('Number of features: %s'%numFeat)
    
    print('Validating model...')
    percentage,success,errors = LLSHandle.valid(Model, Xtest, Ytest)
    print('Number of observations classified correctly: %g'%success)
    print('Number of observations classified wrongly: %g'%errors)
    print('Percentage correctly classified: %g '%percentage)
    
    
    ###########################
    #
    #       STOCHASTIC
    #
    ###########################
    print('Stochastic training procedure')
    stepLen = 1.0/128
    minThreshold=1e-8
    maxEpoc = 10
    
    print('Starting training procedure...')
    Model,numEpoc, Lsq = LLSHandle.trainStochastisc(stepLen, minThreshold, maxEpoc, Xtrain, Ytrain)
    print('Done!\n')
    x = [i+1 for i in range(numEpoc)]
    plt.plot(x, Lsq)
    plt.xlabel('Number of epochs')
    plt.ylabel('Least squares')
    plt.title('Training history')
    plt.show()
    
    
    numObs = len(Xtest)
    print ('Number of training observations: %s'%numObs) 
    numFeat = len(Xtest.T)
    print ('Number of features: %s'%numFeat)
    
    print('Validating model...')
    percentage,success,errors = LLSHandle.valid(Model, Xtest, Ytest)
    print('Number of observations classified correctly: %g'%success)
    print('Number of observations classified wrongly: %g'%errors)
    print('Percentage correctly classified: %g '%percentage)

