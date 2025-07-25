import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pickle

#  dataset
dataset = pd.read_csv('student.csv')
X = dataset.iloc[:, 1:-1].values  
y = dataset.iloc[:, -1].values   

# Encode categorical feature (Internship Experience)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [4])], remainder='passthrough')
X = ct.fit_transform(X)

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
y=le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Training model by Decison Tree Regressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(regressor, model_file)

with open('column_transformer.pkl', 'wb') as ct_file:
    pickle.dump(ct, ct_file)

print("Model and transformer saved successfully!")
