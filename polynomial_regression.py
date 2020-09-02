# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 06:14:36 2020

@author: nil
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 05:30:54 2020

@author: nil
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('polynomial.csv')
X=dataset.iloc[:,0:1].values
y=dataset.iloc[:,1].values




from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
lin_reg =LinearRegression()
lin_reg.fit(X_poly,y)


plt.scatter(X,y,color='red')
plt.plot(X, reg.predict(X), color='blue')
plt.title("ploynomial regression") 
plt.xlabel("position of employee")
plt.ylabel("Salary")

plt.scatter(X,y,color='red')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color='blue')
plt.title("ploynomial regression")  
plt.xlabel("position of employee")
plt.ylabel("Salary")


reg.predict(6.5)

lin_reg.predict(poly_reg.fit_transform(6.5))

