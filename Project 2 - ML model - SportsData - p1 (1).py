# ML Model - this is code for creation of the model. 

import pandas as pd 
from sklearn.tree import DecisionTreeClassifier

sports_data = pd.read_csv('sportsdata.csv')

# input data
X = sports_data.drop(columns = 'sport')

# output data
y = sports_data['sport']

model = DecisionTreeClassifier()
model.fit(X.values, y.values) 
predictions = model.predict([[21,1],[42,0]])
predictions

