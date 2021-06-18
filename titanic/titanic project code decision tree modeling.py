# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 19:05:49 2021

@author: franc
"""


import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#learning_data_path = 'C:\Users\franc\Downloads\titanic\train.csv'

training_data = pd.read_csv(r'C:\Users\franc\Downloads\titanic\train.csv')
print(training_data.head())

training_data= training_data.drop(['Ticket','Cabin'],axis=1)

#to drop the NaN values
training_data = training_data.dropna()


#gives you the table of index values
#print(training_data.columns)

#the target variable. whether the passenger lives or not
y = training_data.Survived

#the input columns used (the features selected)
features = ['SibSp','Parch','PassengerId']

X = training_data[features]

train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=0)

#specifying the model then fitting it
survival_model = DecisionTreeRegressor(random_state=1)
survival_model.fit(train_X,train_y)

#Make predictions then test predictions
val_predictions = survival_model.predict(val_X)
#Mean absolute error of the predictions
val_mae = mean_absolute_error(val_predictions,val_y)

candidate_max_leaf_nodes = [5,25,50,100,250,500,1000,2000]

#defining the getmae function
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

#for loop to calculate the mae for the candidate max leaf node values    
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

#store the lead nodes value with lowest cost
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
print("\n",scores)

best_tree_size = max(scores, key=scores.get)
print("\n", best_tree_size, scores[best_tree_size])

#using the lead nodes value with lowest cost which would be the first value from best_tree_size
survival_model = DecisionTreeRegressor(max_leaf_nodes = 100)
survival_model.fit(train_X,train_y)
val_predictions = survival_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions,val_y)
#so far we have made a model to make predictions. we must now test the data

dt_model_on_full_data = DecisionTreeRegressor(random_state=1)
dt_model_on_full_data.fit(train_X,train_y)

#test_data_path = 'C:\Users\franc\Downloads\titanic\test.csv'
test_data = pd.read_csv(r'C:\Users\franc\Downloads\titanic\test.csv')
test_X = test_data[features]
test_preds = dt_model_on_full_data.predict(test_X)
