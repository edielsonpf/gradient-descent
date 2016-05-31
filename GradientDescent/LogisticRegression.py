'''
This is the stochastic and batch gradient descent algorithms for HW1, IE 490 Machine
Learning, Spring 2016. The loss function is for the logistic regression model.

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
    def __Gradient(self,Xi,Ti,Theta):
        
        #Calculating the gradient of J             
#         gradient = (Xn*Tn)*np.asscalar((np.exp(-Tn*np.dot(Xn,np.matrix(Theta).transpose())))/(1+np.exp(-Tn*np.dot(Xn,np.matrix(Theta).transpose()))))
        gradient = (Xi*Ti)*np.asscalar((np.exp(-Ti*np.dot(Xi,np.matrix(Theta).transpose())))/(1+np.exp(-Ti*np.dot(Xi,np.matrix(Theta).transpose()))))
        return gradient
    
    def __Sigmoid(self,x): 
        '''Compute the sigmoid function ''' 
        sig = (1.0 + np.exp(-1.0 * x))**(-1)
        return sig 
    
    def __Predict(self,x,Theta):
        
        h = self.__Sigmoid(np.dot(x,np.matrix(Theta).transpose()))
        # h is in the range from 0 to 1. If 1, returns 1, otherwise -1
        return round(h)*2-1

    def __RandomlySelect(self,num_observations):
        return np.random.randint(0,num_observations-1,1)
    
    def Loss(self,theta,X,T):
        ''' This function calculates the value of the loss function considering all samples'''
        loss = 0
        for i in range(len(X)):
            y=self.__Sigmoid(np.dot(X[i],np.matrix(theta).transpose()))
            loss=loss+np.nan_to_num(T[i]*np.log(y)+(1-T[i])*np.log(1-y))
        return -1.0*np.asscalar(loss)/len(X)    
        
    def trainBatch(self,step_lenght,max_epochs,X,Y):
            
        #number of observations: must be disposed bY rows
        numObs = len(X)
        #number of features: must be disposed bY columns
        numFeat = len(X.T)
        # counter for the number of epochs
        numEpochs=0
                
        mu, sigma = 0, 0.1 # mean and standard deviation
        theta = np.random.normal(mu, sigma, numFeat)
#       
        
        lsq=[]
        lsq.append(self.Loss(theta, X, Y))
        
        numEpochs=numEpochs+1
        
        #each interaction is equivalent to one training epoch
        while (numEpochs < max_epochs):
        
            print('Epoch: %s'%numEpochs)
            print('Loss: %g'%lsq[numEpochs-1])
            
            #Calculating the gradient of J             
            gradJ = 0
            for i in range(numObs):
                RndObs = self.__RandomlySelect(numObs)
                gradJ = gradJ + self.__Gradient(X[RndObs], Y[RndObs], theta) #(Y[RndObs] - self.__Sigmoid(np.dot(X[RndObs],np.matrix(theta).transpose())))*X[RndObs]
            gradJ=gradJ/len(X)
            
            alpha_k=step_lenght
            
            #Finding the new theta        
            theta=theta+alpha_k*gradJ
            
            lsq.append(self.Loss(theta, X, Y))
            numEpochs=numEpochs+1
                        
        print('End of training procedure!')
        if numEpochs >= max_epochs:
            print('Exceeded the maximum number of epochs: %s' %numEpochs)
                
        return theta,numEpochs,lsq
    
    def trainStochastisc(self,step_lenght,maX_epochs,X,Y):
            
        #number of observations: must be disposed bY rows
        numObs = len(X)
        #number of features: must be disposed bY columns
        numFeat = len(X.T)
        # counter for the number of epochs
        numEpochs=0
                
        mu, sigma = 0, 0.1 # mean and standard deviation
        theta = np.random.normal(mu, sigma, numFeat)
#         theta = np.zeros(numFeat)
        
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
                
                g_k = self.__Gradient(X[RndObs], Y[RndObs], theta) #(Y[RndObs] - self.__Sigmoid(np.dot(X[RndObs],np.matrix(theta).transpose())))*X[RndObs]

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
                prediction = self.__Predict(X[i],theta)
                if  prediction == Y[i]:
                    success = success+1
        
        percentage=(1.0*success/numObs)*100
        errors=numObs-success
        return percentage,success,errors
                        