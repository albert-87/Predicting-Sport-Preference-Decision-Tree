#!/usr/bin/env python
# coding: utf-8

# In[68]:


# ML model - Persisting Models 

import pandas as pd 
from sklearn.tree import DecisionTreeClassifier

# import data set
sports_data = pd.read_csv('sportsdata.csv')
X = sports_data.drop(columns = 'sport')
y = sports_data['sport']

# create model 
model = DecisionTreeClassifier()

# train it
model.fit(X.values, y.values) 

# ask it to make predictions 
predictions = model.predict([[21,1]])
predictions


# In[ ]:




