# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 19:05:49 2021

@author: franc
"""


iimport pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

training_data = pd.read_csv(r'C:\Users\franc\Downloads\titanic\train.csv')
print(training_data.head())

#the first few rows of the training data
training_data.head()

#first few rows of the tes data
test_data.head()

training_data.info()

test_data.info()

#will give me an idea of the values im working with
training_data.describe()

for col in training_data.columns:
    if training_data[col].nunique() < 10:
        print( f"{col}: {training_data[col].unique()}")
        
training_data["Embarked"].value_counts() #seeing how many times each variable shows up
#I will replace the rows with NAN values to be replaced with S since it shows up so many more times
training_data["Embarked"].fillna("S", inplace=True)
training_data["Embarked"].isnull().sum() # If this is zero that means that the NAN rows were filled successfully

#im gonna graph the data in histograms
import matplotlib.pyplot as plt
training_data.hist(bins = 50, figsize = (20,20))
plt.show()


training_data.isnull().sum()
#I will drop cabin from the data
#I can fill in the median age with imputation

#I want to fill the na data in the Age column
median_age = training_data["Age"].median()
print(median_age)
training_data["Age"].fillna(median_age, inplace=True)
training_data["Age"].isnull().sum() #if this line produces a zero then we have successfully replaced the NA rows

import seaborn as sns
relationshipmatrix = sns.heatmap(training_data[["Survived","Pclass","Age","SibSp","Parch","Fare"]].corr(), annot=True)
sns.set(rc={'figure.figsize':(20,20)})

#survived and fare have the strongest relation
#Pclass and Age have a strong negative corrolation however I'm not sure this info is useful


#dropped_cols = ["PassengerId","Name","Cabin","Ticket"]
dropped_cols = ["Name","Cabin","Ticket"]
training_data = training_data.drop(dropped_cols, axis=1)
training_data.head(10)

test_data = test_data.drop(dropped_cols, axis=1)
test_data.head(10)
#columns that are not needed in the testing data are also dropped


median_test_age = test_data["Age"].median()
test_data["Age"].fillna(median_test_age, inplace = True)

median_fare = test_data["Fare"].median()
test_data["Fare"].fillna(median_fare, inplace = True)
test_data.isnull().sum() #this means that the test_data has NAN values in age
# I will try impute these age values
#I will impute after building the rest of the model
#I wonder how NAN values in the test_data will affect the prediction rate of the ML algorithm

number_of_men = training_data.loc[training_data.Sex == 'male']["Survived"]
men_alive = sum(number_of_men)/len(number_of_men)
print("proportion of men who survived is",men_alive)

number_of_women = training_data.loc[training_data.Sex == 'female']["Survived"]
women_alive = sum(number_of_women)/len(number_of_women)
print("proportion of men who survived is",women_alive)


y = training_data["Survived"] #this is the target column
X = training_data.drop("Survived", axis=1) #this is the data without the target column
X_val = test_data.drop("Embarked", axis=1) #I believe the useless columns were dropped
X.head(10) 


#I have to use OneHotEncoding on the Sex column

X_train_gender = X["Sex"]
train_array = pd.Series(X_train_gender)
arr_train = train_array.values
reshape_arr_train = arr_train.reshape((-1,1)) #This ensures that we only have two columns and many rows instead of 1 row

#print(reshape_arr_train)

X_val_gender = X_val["Sex"]
val_array = pd.Series(X_val_gender)
arr_val = val_array.values
reshape_arr_val = arr_val.reshape((-1,1)) #This insures we have two columns and many rows instead of 1 row

print(training_data.shape)
print(test_data.shape)




#X_train_gender.head()
#X_val_gender.head()


OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(reshape_arr_train))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(reshape_arr_val))

#OH_cols_train.head()

OH_cols_train.index = X.index
OH_cols_valid.index = X_val.index


num_X_train = X.drop(["Sex"], axis=1)
num_X_val = X_val.drop(["Sex"], axis=1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_val = pd.concat([num_X_val, OH_cols_valid], axis=1)


#I will drop Embarke for now till I decide how to use it to improve the model
OH_X_train = OH_X_train.drop( ["Embarked"] ,axis=1)


OH_X_train.rename(columns = {0:'Female',1:'Male'}, inplace =True )
OH_X_val.rename(columns = {0:'Female',1:'Male'}, inplace = True)

OH_X_train.head()
#In the case of the new OneHotEncoded columns 1 is male and 0 is female. We know this because we printed reshape_arr_train.

#training_data.head()


OH_X_train.head()


model = DecisionTreeClassifier(random_state=0)
model.fit(OH_X_train,y)
predictions = model.predict(OH_X_val)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
