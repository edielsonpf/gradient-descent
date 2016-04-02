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
    def __RandomlySelect(self,num_observations):
        return np.random.randint(0,num_observations-1,1)
    
    def __leastSquares(self,theta,x,y):
        
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
            
            
            mu, sigma = 0, 0.1 # mean and standard deviation
            theta = np.random.normal(mu, sigma, numFeat)
            
            lsq=[]
            lsq.append(self.__leastSquares(theta, x, y))
            
            numEpochs=numEpochs+1
            
            #each interaction is equivalent to one training epoch
            while (numEpochs < max_epochs) and (lsq[numEpochs-1] > min_threshold):
            
                print('Epoch: %s'%numEpochs)
                print('Least squares: %g'%lsq[numEpochs-1])
                #Calculating the gradient of J             
                gradJ = 0
                for i in range(numObs):
                    RndObs = self.__RandomlySelect(numObs)
                    gradJ = gradJ + (np.dot(x[RndObs],np.matrix(theta).transpose())-y[RndObs])*x[RndObs]
                gradJ=gradJ/len(x)
                
                #Finding the new theta        
                theta=theta-step_lenght*gradJ
                
                lsq.append(self.__leastSquares(theta, x, y))
                numEpochs=numEpochs+1
                
            print('End of training procedure!')
            if numEpochs >= max_epochs:
                print('Exceeded the maximum number of epochs: %s' %numEpochs)
            if lsq[numEpochs-1] <= min_threshold:
                print('Achieved the desired threshold to the least squares: %s' %numEpochs)
            
            return theta,numEpochs,lsq        