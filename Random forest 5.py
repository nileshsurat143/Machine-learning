# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:59:43 2020

@author: nil
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('polynomial.csv')
X=dataset.iloc[:,0:1].values
y=dataset.iloc[:,1].values

'''from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X = sc_X.fit_transform(X)
sc_y=StandardScaler() 
y=sc_y.fit_transform(np.reshape(y,(10,1)))'''


from sklearn.ensemble  import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=10, random_state =0)
reg.fit(X,y)



plt.scatter(X,y,color='red')
plt.plot(X, reg.predict(X), color='blue')
plt.title("SVR regression") 
plt.xlabel("position of employee")
plt.ylabel("Salary")
plt.show()


x_grid = np.arange(min(X),max(X),0.3)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(X,y,color='red')
plt.plot(x_grid, reg.predict(x_grid), color='blue')
plt.title("Decision tree") 
plt.xlabel("position of employee")
plt.ylabel("Salary")
plt.show()


reg.predict(np.reshape(6.5,(1,1)))
# predict = sc_y.inverse_transform(reg.predict(sc_X.transform(np.array([[6.5]]))))


