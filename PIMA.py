# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:18:34 2018

@author: Sagar Vakil
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models = []

models.append(("Logistic Regression:",LogisticRegression()))
models.append(("Naive Bayes:",GaussianNB()))
models.append(("K-Nearest Neighbour:",KNeighborsClassifier(n_neighbors=5)))
models.append(("Decision Tree:",DecisionTreeClassifier()))
models.append(("Support Vector Machine-linear:",SVC(kernel="linear",C=0.2)))
models.append(("Random Forest:",RandomForestClassifier(n_estimators=5)))


df_pima=pd.read_csv('C:\\Users\\ishan\\Desktop\\diabetes.csv')
##print(df_pima.head(8))

print(df_pima.describe())

###PREPROCESSING

##replacing 0 to find the missing values
#print('Dataset Shape before dropna : ', df_pima.shape)
df_pima.hist(figsize=(18,12))
print(df_pima.groupby("Outcome").size())


df_pima['Pregnancies'] = df_pima['Pregnancies'].replace(0, np.nan)
df_pima['Glucose'] = df_pima['Glucose'].replace(0, np.nan)
df_pima['BloodPressure'] = df_pima['BloodPressure'].replace(0, np.nan) 
df_pima['SkinThickness'] = df_pima['SkinThickness'].replace(0, np.nan) 
df_pima['Insulin'] = df_pima['Insulin'].replace(0, np.nan)        
df_pima['BMI'] = df_pima['BMI'].replace(0, np.nan) 
df_pima['DiabetesPedigreeFunction'] = df_pima['DiabetesPedigreeFunction'].replace(0, np.nan) 
df_pima['Age'] = df_pima['Age'].replace(0, np.nan) 

print("null values count :",df_pima.isnull().sum())

#Filling the missing values
df_pima['BMI'].fillna(df_pima['BMI'].median(), inplace=True)
df_pima['Glucose'].fillna(df_pima['Glucose'].median(), inplace=True)
df_pima['BloodPressure'].fillna(df_pima['BloodPressure'].median(), inplace=True)

df_pima = df_pima.drop('SkinThickness',axis =1)
df_pima = df_pima.drop('Insulin',axis =1)
df_pima.describe()


df_pima = df_pima.dropna();
print('Dataset Shape after dropna : ', df_pima.shape)
print(df_pima.describe())

print(df_pima.groupby("Outcome").size())

 
#PREPROCESSING COMPLETED

corr = df_pima[df_pima.columns].corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
#sns.heatmap(corr, cmap=cmap, annot = True)

df_pima.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(18,12))


X_features = pd.DataFrame(data = df_pima, columns = ['Pregnancies', 'Glucose', 'BloodPressure',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
Y = df_pima.iloc[:,6]

scaler = StandardScaler(with_mean=True, with_std=True)
X_features = scaler.fit_transform(X_features)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.25, random_state=10)

print('Accuracy of the algorithms')
results = []
names = []
for name,model in models:
    kfold = KFold(n_splits=10, random_state=22)
    cv_result = cross_val_score(model,X_train,Y_train, cv = kfold,scoring = "accuracy")
    names.append(name)
    results.append(cv_result)
for i in range(len(names)):
    print(names[i],results[i].mean()*100)




#feature Selection
    
  
#1  Univariate
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
array = df_pima.values
X = array[:, 0:6]
y = array[:, 6]

print("Accuracy of the algorithms")
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)
# summarize scores
np.set_printoptions(precision=3)

print("scores")
print(fit.scores_)
features = fit.transform(X_features)
print("features")
print(features[0:5, :])


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=22)

print('Models appended...')
results = []
names = []
for name,model in models:
    kfold = KFold(n_splits=10, random_state=22)
    cv_result = cross_val_score(model,X_train,Y_train, cv = kfold,scoring = "accuracy")
    names.append(name)
    results.append(cv_result)
for i in range(len(names)):
    print(names[i],results[i].mean()*100)


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
svm = SVC()
svm.fit(X_train,Y_train)
predictions = svm.predict(X_test)
print(accuracy_score(Y_test,predictions))
print(classification_report(Y_test,predictions))


plt.figure(figsize = (18,12))
sns.countplot(df_pima['Age'])

