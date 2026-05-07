#Assignment 2. Perform the following operations using Python on the Air quality and Heart
#Diseases data sets
# a. Data cleaning
# b. Data integration
# c. Data transformation
# d. Error correcting
# e. Data model building
# Dataset ( Heart Diseases ) : https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================================
# LOAD DATASET
# =========================================

heart_data = pd.read_csv(
    r'C:\Users\CloudMorph\Downloads\heart.csv'
)

# =========================================
# DISPLAY DATA
# =========================================

print(heart_data.head())

print(heart_data.tail())

print(heart_data.shape)

# =========================================
# DATA INFORMATION
# =========================================

heart_data.info()

# Check missing values
print(heart_data.isnull().sum())

# Statistical information
print(heart_data.describe())

# Count target values
print(heart_data['target'].value_counts())

# 1 --> Defective Heart
# 0 --> Healthy Heart

# =========================================
# SPLITTING FEATURES AND TARGET
# =========================================

X = heart_data.drop(columns='target')

Y = heart_data['target']

print(X)

print(Y)

# =========================================
# SPLITTING TRAINING AND TEST DATA
# =========================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)

print(X.shape, X_train.shape, X_test.shape)

# =========================================
# MODEL TRAINING
# =========================================

# Logistic Regression Model
model = LogisticRegression(max_iter=1050)

# Train the model
model.fit(X_train, Y_train)

# =========================================
# MODEL EVALUATION
# =========================================

# Accuracy on Training Data
X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(
    X_train_prediction,
    Y_train
)

print('Accuracy on Training Data : ',
      training_data_accuracy)

# Accuracy on Test Data
X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(
    X_test_prediction,
    Y_test
)

print('Accuracy on Test Data : ',
      test_data_accuracy)

print(type(X_test))

print(X_test_prediction)

# =========================================
# BUILDING A PREDICTIVE SYSTEM
# =========================================

# Single input with 13 features
input_data = (
    52, 1, 0, 125, 212,
    0, 1, 168, 0, 1,
    2, 2, 3
)

# Convert input data to numpy array
input_data_as_numpy_array = np.array(input_data)

# Reshape array for single prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Convert into DataFrame
test_df = pd.DataFrame(
    input_data_reshaped,
    columns=X_test.columns
)

# Prediction
prediction = model.predict(test_df)

print(prediction)

# Result
if prediction[0] == 0:
    print('The Person does not have Heart Disease')
else:
    print('The Person has Heart Disease')