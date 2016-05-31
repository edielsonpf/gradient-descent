'''
This is the stochastic and batch gradient descent algorithms for HW1, IE 490 Machine
Learning, Spring 2016. The loss function is for the linear least squares model.

Created on Apr 1, 2016

@author: Edielson
'''
import numpy as np

class LLSClass(object):
    '''
    classdocs
    '''
    

    def __init__(self):
        '''
        Constructor
        '''
    def __Predict(self,x,theta):
        
        h = np.dot(x,np.matrix(theta).transpose())
                
        if h >= 0:
            return 1
        else:
            return -1
    
    def __RandomlySelect(self,num_observations):
        return np.random.randint(0,num_observations-1,1)
    
    def __Gradient(self,xi,yi,Theta):
        
        #Calculating the gradient of J             
        return (np.dot(xi,np.matrix(Theta).transpose())-yi)*xi
    
    def Loss(self,Theta,X,Y):
        
        loss = 0
        for i in range(len(X)):
            loss=loss+0.5*(np.dot(X[i],np.matrix(Theta).transpose())-Y[i])**2
        return np.asscalar(loss/len(X))    
        
    def trainBatch(self,step_lenght,min_threshold,max_epochs,x,y):
            
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
            
            #Calculating 1/m \sum_{i=1}^{m} (gradient of J)             
            gradJ = 0
            for i in range(numObs):
                # Using random sample with replacement
                RndObs = self.__RandomlySelect(numObs)
                gradJ = gradJ + self.__Gradient(x[RndObs], y[RndObs], theta)
            gradJ=gradJ/numObs
            
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
    
    def trainStochastisc(self,step_lenght,min_threshold,max_epochs,x,y):
            
        loosUpdate = 100
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
            
            for i in range(numObs):
                
                RndObs = self.__RandomlySelect(numObs)
                
                #Calculating the gradient of J     
                g_k=self.__Gradient(x[RndObs], y[RndObs], theta) #(np.dot(x[RndObs],np.matrix(theta).transpose())-y[RndObs])*x[RndObs]

                alpha_k=step_lenght/(np.sqrt((i+1)*(numEpochs)))
                
                #Finding the new theta        
                theta=theta-alpha_k*g_k
            
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
                prediction = self.__Predict(x[i],theta)
                if  prediction == y[i]:
                    success = success+1
        percentage=(1.0*success/numObs)*100
        errors=numObs-success
                    
        return percentage,success,errors
                    