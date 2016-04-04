'''
Created on Apr 3, 2016

@author: Edielson
'''
import numpy as np

class LRClass(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
    def __Sigmoid(self,X): 
        '''Compute the sigmoid function ''' 
        den = 1.0 + np.exp(-1.0 * X) 
        return (1.0 / den) 
    
    def __Predict(self,h):
        if h > 0.5:
            y=1
        else:
            y=-1
        return y

    def __RandomlySelect(self,num_observations):
        return np.random.randint(0,num_observations-1,1)
    
    def Loss(self,theta,X,Y):
        
        loss = 0
        for i in range(len(X)):
#             loss=loss+np.log(1+np.exp(-Y[i]*np.dot(X[i],np.matriX(theta).transpose())))
            loss=loss+Y[i]*np.dot(X[i],np.matrix(theta).transpose())-np.log(1 + self.__Sigmoid(np.dot(X[i],np.matrix(theta).transpose())))
        return np.asscalar(loss/len(X))    
        
    def trainBatch(self,step_lenght,maX_epochs,X,Y):
            
        #number of observations: must be disposed bY rows
        numObs = len(X)
        #number of features: must be disposed bY columns
        numFeat = len(X.T)
        # counter for the number of epochs
        numEpochs=0
                
#         mu, sigma = 0, 0.1 # mean and standard deviation
#         theta = np.random.normal(mu, sigma, numFeat)
        theta = np.zeros(numFeat)
        
        lsq=[]
        lsq.append(self.Loss(theta, X, Y))
        
        numEpochs=numEpochs+1
        
        #each interaction is equivalent to one training epoch
        while (numEpochs < maX_epochs):
        
            print('Epoch: %s'%numEpochs)
            print('Loss: %g'%lsq[numEpochs-1])
            
            #Calculating the gradient of J             
            gradJ = 0
            for i in range(numObs):
                RndObs = self.__RandomlySelect(numObs)
                gradJ = gradJ + (Y[RndObs] - self.__Sigmoid(np.dot(X[RndObs],np.matrix(theta).transpose())))*X[RndObs]
            gradJ=gradJ/len(X)
            
            alpha_k=step_lenght
            
            #Finding the new theta        
            theta=theta+alpha_k*gradJ
            
            lsq.append(self.Loss(theta, X, Y))
            numEpochs=numEpochs+1
                        
        print('End of training procedure!')
        if numEpochs >= maX_epochs:
            print('Exceeded the maximum number of epochs: %s' %numEpochs)
                
        return theta,numEpochs,lsq
    
    def trainStochastisc(self,step_lenght,maX_epochs,X,Y):
            
        #number of observations: must be disposed bY rows
        numObs = len(X)
        #number of features: must be disposed bY columns
        numFeat = len(X.T)
        # counter for the number of epochs
        numEpochs=0
                
#         mu, sigma = 0, 0.1 # mean and standard deviation
#         theta = np.random.normal(mu, sigma, numFeat)
        theta = np.zeros(numFeat)
        
        lsq=[]
        lsq.append(self.Loss(theta, X, Y))
        
        numEpochs=numEpochs+1
        
        #each interaction is equivalent to one training epoch
        while (numEpochs < maX_epochs):
        
            print('Epoch: %s'%numEpochs)
            print('Loss: %g'%lsq[numEpochs-1])
            
            #Calculating the gradient of J             
            for i in range(numObs):
                
                RndObs = self.__RandomlySelect(numObs)
                
                g_k = (Y[RndObs] - self.__Sigmoid(np.dot(X[RndObs],np.matrix(theta).transpose())))*X[RndObs]

                alpha_k=step_lenght/(np.sqrt((i+1)*(numEpochs)))
                
                #Finding the new theta        
                theta=theta+alpha_k*g_k
            
            lsq.append(self.Loss(theta, X, Y))
            numEpochs=numEpochs+1
                        
        print('End of training procedure!')
        if numEpochs >= maX_epochs:
            print('Exceeded the maximum number of epochs: %s' %numEpochs)
        
        
        return theta,numEpochs,lsq
        
    def valid(self,theta,X,Y):
        #number of observations: must be disposed bY rows
        numObs = len(X)
        
        success=0
        for i in range(numObs):
                p = self.__Sigmoid(np.dot(X[i],np.matrix(theta).transpose()))
                prediction = self.__Predict(p)
                if  prediction == Y[i]:
                    success = success+1
        
        percentage=(1.0*success/numObs)*100
        errors=numObs-success
        return percentage,success,errors
                        