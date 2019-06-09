# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:48:12 2019

@author: prince
"""


#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Import data set
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


# Feature Scalling 

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x= sc_x.fit_transform(x)
y= sc_y.fit_transform(y.reshape(-1, 1))


#Fitting data to svr to dataset

from sklearn.svm import SVR
regssor = SVR(kernel = 'rbf')
regssor.fit(x,y)


#predicting a nde result
y_pred = sc_y.inverse_transform(regssor.predict(sc_x.transform(np.array([[6.5]]))))



# Visulaizing the svr results
plt.scatter(x,y,color = 'red')
plt.plot(x,regssor.predict(x),color = 'blue')
plt.title('Truth or Bluff (svr model)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()
