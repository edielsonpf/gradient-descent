'''
Created on Apr 1, 2016

@author: Edielson
'''
import numpy as np

class BatchClass(object):
    '''
    classdocs
    '''
    

    def __init__(self):
        '''
        Constructor
        '''
    def __Sigmoid(self,x):
        if x >= 0:
            return 1
        else:
            return -1
    
    def __RandomlySelect(self,num_observations):
        return np.random.randint(0,num_observations-1,1)
    
    def Loss(self,theta,x,y):
        
        leastSquares = 0
        for i in range(len(x)):
            leastSquares=leastSquares+0.5*(np.dot(x[i],np.matrix(theta).transpose())-y[i])**2
        return np.asscalar(leastSquares/len(x))    
        
    def train(self,step_lenght,min_threshold,max_epochs,x,y):
            
        #number of observations: must be disposed by rows
        numObs = len(x)
        #number of features: must be disposed by columns
        numFeat = len(x.T)
        # counter for the number of epochs
        numEpochs=0
        threshold = 1000
        
        mu, sigma = 0, 0.1 # mean and standard deviation
        theta = np.random.normal(mu, sigma, numFeat)
        
        lsq=[]
        lsq.append(self.Loss(theta, x, y))
        
        numEpochs=numEpochs+1
        
        #each interaction is equivalent to one training epoch
        while (numEpochs < max_epochs) and (threshold > min_threshold):
        
            print('Epoch: %s'%numEpochs)
            print('Least squares: %g'%lsq[numEpochs-1])
            #Calculating the gradient of J             
            gradJ = 0
            for i in range(numObs):
                RndObs = self.__RandomlySelect(numObs)
                gradJ = gradJ + (np.dot(x[RndObs],np.matrix(theta).transpose())-y[RndObs])*x[RndObs]
            gradJ=gradJ/len(x)
            
            alpha_k=step_lenght
            
            #Finding the new theta        
            theta=theta-alpha_k*gradJ
            
            lsq.append(self.Loss(theta, x, y))
            numEpochs=numEpochs+1
            threshold = np.sqrt((lsq[numEpochs-1]-lsq[numEpochs-2])**2)
            
        print('End of training procedure!')
        if numEpochs >= max_epochs:
            print('Exceeded the maximum number of epochs: %s' %numEpochs)
        if threshold <= min_threshold:
            print('Achieved the desired threshold for the least squares: %s' %threshold)
        
        return theta,numEpochs,lsq
        
    def valid(self,theta,x,y):
        #number of observations: must be disposed by rows
        numObs = len(x)
        
        success=0
        for i in range(numObs):
                h = np.dot(x[i],np.matrix(theta).transpose())
                prediction = self.__Sigmoid(h)
                
                if  prediction == y[i]:
                    success = success+1
        percentage=(1.0*success/numObs)*100
        errors=numObs-success
                    
        return percentage,success,errors
                    