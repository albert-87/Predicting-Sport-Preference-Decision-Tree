# Visualizing a Decision tree

# we will export our model in a visual format to see how the model makes predictions 

import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree # object has a method for exporting decisiion tree in a graphical format 

#import dataset
sd = pd.read_csv('sportsdata.csv')

# create input and output data set 
X = sd.drop(columns = ['sport'])
y = sd['sport']

# create model 
model = DecisionTreeClassifier()

# train model 
model.fit(X,y)

tree.export_graphviz(model, out_file = 'sports_predictor.dot', 
                     feature_names = ['age', 'gender'], 
                     class_names = sorted(y.unique()),
                     label = 'all',
                     rounded=True, 
                     filled=True)
                     
