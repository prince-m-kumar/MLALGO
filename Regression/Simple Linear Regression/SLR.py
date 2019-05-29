# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:06:25 2019

@author: prince

Simple Linear Regression

"""

"""
Data Set Template 

"""
#Import Libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

x= dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

print(x)

print(y)


"""
Split Create Data SET from Training and  Test test 
"""


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

#Fitting Simple Linear Regraession to the Training Set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predict the Test Set results
y_predict = regressor.predict(x_test)

# Visualising the Training set results
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Year Of Experience')
plt.ylabel('Salary')
plt.show()


# Visualising the Test set results
plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Year Of Experience')
plt.ylabel('Salary')
plt.show()


