# -*- coding: utf-8 -*-
"""
Created on Thu May 30 07:24:58 2019

@author: prince

Topics: Multiple  Linear Regression
"""


"""
Data Set Template 

"""
#Import Libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

x= dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
x[:,3]=labelencoder_X.fit_transform(x[:,3])


onehotencoder = OneHotEncoder(categorical_features=[3])
x= onehotencoder.fit_transform(x).toarray()

# Avoid Dummy Variable Trap
x=x[:,1:]

print(x)

print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_predict = regressor.predict(x_test)


#Building the optmial model using Backward Elimination 
import statsmodels.formula.api as sm

# add in matrix x 
# axis = 1 :- Add column 0 :- add row

#X = np.append(arr = x, values = np.ones((50,1)).astype(int),axis = 1) 
X = np.append(arr = np.ones((50,1)).astype(int), values = x,axis = 1)
#all possible predictors
X_opt = X[:,[0,1,2,3,4,5]]

# Fit the full model with all possible predictors
regressor_OLS =sm.OLS(endog = y, exog = X_opt).fit()

# P Value
regressor_OLS.summary()


#Remove one value
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS =sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


#Remove one value
X_opt = X[:,[0,3,4,5]]
regressor_OLS =sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Remove one value
X_opt = X[:,[0,3,5]]
regressor_OLS =sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


#Remove one value
X_opt = X[:,[0,3]]
regressor_OLS =sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


x_train_new,x_test_new,y_train_new,y_test_new = train_test_split(X_opt,y,test_size = 0.2,random_state = 0)
regressor_new = LinearRegression()
regressor_new.fit(x_train_new,y_train_new)
y_predict_new = regressor_new.predict(x_test_new)


# Backward Elimination Function with P -value
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        print(maxVar)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x


