#!/usr/bin/env python
# coding: utf-8

# In[63]:


# ML measure accuracy of model - 30% data for testing model 70% for training 

import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sports_data = pd.read_csv('sportsdata.csv')

# input data
X = sports_data.drop(columns = 'sport')

# output data
y = sports_data['sport']

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.3) 

model = DecisionTreeClassifier()
model.fit(X_train.values, y_train.values) 
predictions = model.predict(X_test.values)

score = accuracy_score(y_test, predictions)
score


# In[ ]:




