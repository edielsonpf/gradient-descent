# gradient-descent

## Quick start

### Running the test script

```Python

from Tools import LoadCSV
from LinearLeastSquaresTest import LinearLeastSquaresTest
from LogisticRegressionTest import LogisticRegressionTest

###
#
#    Observations must be distributed by rows and features by columns
# 
###

TEST_LEAST_SQUARES = 1
TEST_LOGISTIC_REGRESSION = 1

#Loading data from CSV file
Xtrain = LoadCSV('data/Xtrain.csv', 50)
Ytrain = LoadCSV('data/Ytrain.csv', 1)

Xtest = LoadCSV('data/Xtest.csv', 50)
Ytest = LoadCSV('data/Ytest.csv', 1)

if TEST_LEAST_SQUARES == 1:
    print('Batch gradient descent algorithm \n')
    LinearLeastSquaresTest(Xtrain, Ytrain, Xtest, Ytest)

if TEST_LOGISTIC_REGRESSION == 1:
    print('Stochastic gradient descent algorithm \n')
    LogisticRegressionTest(Xtrain, Ytrain, Xtest, Ytest)
```
