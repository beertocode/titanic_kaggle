import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
import operator
from sklearn.metrics import mean_absolute_error
import re
import warnings
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
#from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
#from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats.stats import pearsonr
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
#from nameparser import HumanName 
warnings.filterwarnings("ignore")



#defining file path
path1="train.csv"
path2="test.csv"

#load the data
data1=pd.read_csv(path1)
data1=data1.dropna(axis=0)

data2=pd.read_csv(path2)


#prints first few rows
print(data1.head())
print(data2.head())

#to see no. of empty data..
print(data2.isnull().sum())
#age cabin and embarked columnd have some empty data


#inserting survived column in test data
df=pd.DataFrame(data2)
df
"""df.insert(1,'Survived','0')
df"""
print(data2.head()) #to chk the added column

#counting columns and rows
row=len(df.index)
print(row)
column=len(df.columns)
print(column)

#condition check for survival

#setting y to actual data of survival
y=data1.Survived


features=['Pclass','Age','Sex','Embarked']
train_X=data1[features]
test_X=data2[features]

#label encode the categorical values and convert them to numbers
le=LabelEncoder()
le.fit(train_X['Sex'].astype(str))
train_X['Sex']=le.transform(train_X['Sex'].astype(str))
test_X['Sex']=le.transform(test_X['Sex'].astype(str))

le.fit(train_X['Pclass'].astype(str))
train_X['Pclass']=le.transform(train_X['Pclass'].astype(str))
test_X['Pclass']=le.transform(test_X['Pclass'].astype(str))


le.fit(train_X['Embarked'].astype(str))
train_X['Embarked']=le.transform(train_X['Embarked'].astype(str))
test_X['Embarked']=le.transform(test_X['Embarked'].astype(str))

#fill missing values in test data with mean value of age
mean=sum(train_X['Age'])/(418-86)
test_X['Age'].fillna(mean,inplace=True)

#train the model using logistic regression
model=LogisticRegression()
model.fit(train_X,y)


predictions = model.predict(test_X)

predictions=(map(round,predictions))
predictions=(map(int,predictions))

submission = pd.DataFrame({'PassengerID': data2['PassengerId'],
                           'Survived': predictions
                           })
submission.to_csv('Desktop/titanic_kaggle/tit.csv', index=False)
