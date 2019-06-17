import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Train.csv', names=['id', 'gender', 'married', 'dependents', 'education', 'self_emp', 'applicant_income', 'coapplicant_income', 'amount', 'term', 'history', 'area','status'])
df.drop(columns=['id', 'gender', 'married'], inplace=True)

#Dependents preprocessing
df.dependents.replace(" ", " 0", inplace = True)
# print(df['dependents'].value_counts())

#Self_emp
df.self_emp.replace(" ", " No", inplace = True)
# print(df['self_emp'].value_counts())

df = pd.get_dummies(df, columns = ['education', 'self_emp', 'area'], drop_first = True)

s = pd.Series(df['amount'])
df['amount'] = pd.to_numeric(s, errors = 'coerce')
df['amount'].isna().sum()
df['amount'].fillna(df.amount.mean(), inplace = True)

df['term'].replace(' ', ' 360', inplace = True)
df['history'].replace(' ', ' 1', inplace = True)

df = pd.get_dummies(df, columns = ['history', 'dependents', 'term'], drop_first = True)
op = df['status']
op = pd.get_dummies(op, columns = ['status'])
op.drop(columns=[' N'], inplace = True)
df.drop(columns=['status'], inplace = True)

#s = pd.Series(df['term'])
#df['term'] = pd.to_numeric(s, errors = 'coerce')

# df['income'] = df['applicant_income']+df['coapplicant_income']
# df.drop(columns=['applicant_income', 'coapplicant_income'], inplace = True)

from sklearn import preprocessing,cross_validation
X = df.iloc[:, :].values
X=preprocessing.scale(X)
y = op.iloc[:, :].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
model = RandomForestClassifier()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
