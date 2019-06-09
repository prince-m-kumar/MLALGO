# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:52:14 2019

@author: prince

Implementing Polynomial Regression 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

x= dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


# Fitting Linear Regression to the Data set


# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))

#Fitting Polynomial Regression to the data set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

# Visualising the Linear Regression Result 

plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')
plt.title('truth or Bluff(Linear Regression)')
plt.xlabel('position level')
plt.show()

# Visualising the Polynomial Regression Result 
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(x,y,color = 'red')
plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color = 'blue')
plt.title('truth or Bluff(Linear Regression)')
plt.xlabel('position level')
plt.show()



#Predict a new result with polynomial Regressin

