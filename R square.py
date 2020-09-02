# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 05:54:55 2020

@author: nil
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('linear_r_data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=3,random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

y_predict= reg.predict(X_test)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title("Linear regression")
plt.xlabel("Experience")
plt.ylabel("Salary")



plt.scatter(X_test,y_test,color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title("Linear regression")
plt.xlabel("Experience")
plt.ylabel("Salary")