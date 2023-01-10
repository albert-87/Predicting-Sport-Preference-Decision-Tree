# LOAD ML MODEL FILE

# ML model - Persisting (re-using) Models 
# We will build and train our model and save to a file. 
# The next time we want to make predictions we just load the model from the file and ask it to make predictions. 
# This saved model is already trained... so we do not need to retrain it. 
# Below is the code on how to do this. 


import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
import joblib # this job lib object has methods for saving and loading models

sports_data = pd.read_csv('sportsdata.csv')
X = sports_data.drop(columns = 'sport')
y = sports_data['sport']

model = DecisionTreeClassifier()
model.fit(X.values, y.values)

# sports_predictor.joblib is the file where we want to store our trained model. Save file. 
model = joblib.load('sports_predictor.joblib') 

predictions = model.predict([[21,1]])
predictions

