# ML model - to make prediction on what type of sport someone will like based on input parameters. 

import pandas as pd 
from sklearn.tree import DecisionTreeClassifier

# import data set
sports_data = pd.read_csv('sportsdata.csv')

# input parameters
X = sports_data.drop(columns = 'sport')

# output parameter
y = sports_data['sport']

# create model 
model = DecisionTreeClassifier()

# train model
model.fit(X.values, y.values) 

# ask model to make predictions 
predictions = model.predict([[21,1]])
predictions






