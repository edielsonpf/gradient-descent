'''
Created on Apr 2, 2016

@author: Edielson
'''

import csv
import numpy as np

def LoadCSV(fileName,numFeatures):
    
    with open(fileName, 'rb') as csvfile:
        dataReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        dataTrain=np.empty(shape=[0, numFeatures])
        for row in dataReader:
            listAux=[]
            for column in row:
                listAux.append(float(column))
            dataTrain = np.append(dataTrain, [listAux], axis=0)
    return dataTrain        
