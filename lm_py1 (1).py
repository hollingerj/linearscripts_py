#!/usr/bin/env python
# coding: utf-8

# Need to upload file to the jupyter environment Reading the data set

# In[2]:
import sys
import csv
if len (sys.argv) < 2:
   sys.exit(1)


argument = sys.argv[1]

import pandas as pd
data = pd.read_csv(argument)


# Creating the scatterplot, assigning variables

# In[3]:


import matplotlib.pyplot as plt
x = data['x']
y = data['y']
plt.scatter(x,y)
plt.show()
plt.savefig("lm_orig.png")

# Modeling the Data

# In[4]:


import numpy as np
arrayX = np.array(x).reshape((-1,1))
arrayY = np.array(y)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(arrayX,arrayY)
y_pred = model.predict(arrayX)
r_sq = model.score(arrayX,arrayY)


# In[5]:


plt.plot(x,y_pred)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()
plt.scatter(x,y)
plt.plot(x,y_pred)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()
plt.savefig("lm_py.png")

# In[ ]:





# In[ ]:




