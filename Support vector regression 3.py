# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:11:45 2020

@author: nil
"""

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

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X = sc_X.fit_transform(X)
sc_y=StandardScaler() 
y=sc_y.fit_transform(np.reshape(y,(10,1)))


from sklearn.svm import SVR
reg = SVR(kernel='rbf')
reg.fit(X,y)



plt.scatter(X,y,color='red')
plt.plot(X, reg.predict(X), color='blue')
plt.title("SVR regression") 
plt.xlabel("position of employee")
plt.ylabel("Salary")
plt.show()





reg.predict(np.reshape(6.5,(1,1)))
predict = sc_y.inverse_transform(reg.predict(sc_X.transform(np.array([[6.5]]))))


