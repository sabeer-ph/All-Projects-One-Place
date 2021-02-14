# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 06:06:51 2021

@author: sabeer Hussain

Project : A Complete Machine Learning Project Walkthrough with Python.

Below are the steps that are covered in this Machine Learning project walkthrough:

1.Importing the Data
2. Data Visualization
3. Data Cleaning and Transformation
4. Encoding the Data
5. Splitting the data into Training and Test sets
6. Fine Tuning Algorithms
7. Cross Validate with KFold
8. Prediction on the test set

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
#%matplotlib inline
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

#Data visualization is crucial to recognize the underlying patterns to properly train the machine learning model:
    

display(data_train.head(10))

data_train.columns

sns.barplot(x='Embarked',y='Survived',data=data_train,hue='Sex')
plt.show()

'''Data Cleaning and Transformation:
Now the next step is to clean and transform data according to the output that we need. Here are the steps that I will consider in this step:

1. To avoid overfitting, I’m going to group people into logical human age groups.
2. Each booth begins with a letter. I bet this letter is much larger than the number that follows, let’s cut it off.
3. The tariff is another continuous value that should be simplified.
4. Extract the information from the “Name” function. Rather than using the full name, I extracted the last name and name prefix (Mr, Mrs etc.) and then added them as characteristics.
Finally, we need to remove unnecessary features.'''

def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df
def simplify_cabins(df):
    '''get the first letter of canbin and for NAN fill 'N' '''
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0]) # get last name
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1]) # get mr,mrs,miss
    return df

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

data_train = transform_features(data_train)
data_test = transform_features(data_test)


'''Encoding Features:
The next step is to standardize the labels. The Label encoder converts each unique string into a number, 
making the data more flexible that can be used for various algorithms. 
The result is a scary array of numbers for humans, but beautiful for machines:'''

from sklearn import preprocessing

def encode_features(df_train,df_test):
    features = ['Age','Cabin','Fare','Lname','NamePrefix','Sex']
    #print(df_train.shape)
    #print(df_test.shape)
    df_combined = pd.concat([df_train[features],df_test[features]])
    #print(df_combined.shape)
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature]=le.transform(df_train[feature])
        df_test[feature]=le.transform(df_test[feature])
    #print(df_train[features].head())
    #print(df_test[features].head())
    return df_train,df_test
        

data_train,data_test = encode_features(data_train,data_test)

'''Now the next step is to divide the data into training and testing sets. 
Here I’ll be using one variable to store all the features minus the value we want to predict, 
and the other variable to store only the value we want to predict.

For this task, I’m going to randomly mix this data into four variables.
 In this case, I train 80% of the data, then I test the remaining 20%:'''

from sklearn.model_selection import train_test_split

X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)


'''Fitting and Tuning Machine Learning Algorithm:
Now is the time to determine which algorithm will provide the best model. 
In this task, I am going with the RandomForestClassifier, but you can also use any other classifier here, 
such as Support Vector Machines or Naive Bayes:'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier. 
clf = RandomForestClassifier()
print(clf)


# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

clf = grid_obj.best_estimator_ # set the best estimator back to the model

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)


predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

'''Now we need to use KFold cross-validation to validate our machine learning model.
 KFold cross-validation helps to understand that is our model good? 
 This makes it possible to verify the efficiency of the algorithm using KFold. 
 This will divide our data into 10 compartments, then run the algorithm using a different compartment as the test 
 set for each iteration:'''
 
from sklearn.model_selection import KFold
 
def run_KFold(clf):
    folds = KFold(n_splits=10)# 891 is data_train.shape
    outcomes = []
    fold = 0
    for train_index,test_index in folds.split(X_all):
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome)) 
run_KFold(clf)

'''Testing the Model:
Now we need to predict on the actual test data:'''

ids = data_test['PassengerId']
predictions = clf.predict(data_test.drop('PassengerId', axis=1))
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.head()
