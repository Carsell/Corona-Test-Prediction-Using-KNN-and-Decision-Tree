#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Load the data
df = pd.read_csv('corona.csv')


# Handle missing and 'None' values
# For simplicity, we remove rows with missing values
df = df.replace('None', np.nan)
df = df.dropna()


# Handle 'other' and 'Other' in Corona column
# We can group them together
df['Corona'] = df['Corona'].replace({'other': 'Other'})


# Convert categorical variables to numerical
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == type(object):
        df[column] = le.fit_transform(df[column])
        

# Visualize the relationship between independent variables and dependent variable
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Split the data into features and target variable
X = df.drop('Corona', axis=1)
y = df['Corona']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Apply KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


# Evaluate KNN
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print('KNN accuracy:', knn_accuracy)


# Apply Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)


# Evaluate Decision Tree
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print('Decision Tree accuracy:', dt_accuracy)

