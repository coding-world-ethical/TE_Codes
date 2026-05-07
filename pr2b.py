import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv(
    r'C:\Users\CloudMorph\Downloads\heart.csv'
)
print(heart_data.head())
print(heart_data.tail())
print(heart_data.shape)

heart_data.info()
print(heart_data.isnull().sum())
print(heart_data.describe())
print(heart_data['target'].value_counts())

X = heart_data.drop(columns='target')
Y = heart_data['target']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression(max_iter=1050)
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(
    X_train_prediction,
    Y_train
)

print('Accuracy on Training Data : ',
      training_data_accuracy)

X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(
    X_test_prediction,
    Y_test
)

print('Accuracy on Test Data : ',
      test_data_accuracy)
print(type(X_test))
print(X_test_prediction)

input_data = (
    52, 1, 0, 125, 212,
    0, 1, 168, 0, 1,
    2, 2, 3
)

input_data_as_numpy_array = np.array(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

test_df = pd.DataFrame(
    input_data_reshaped,
    columns=X_test.columns
)

prediction = model.predict(test_df)

print(prediction)

if prediction[0] == 0:
    print('The Person does not have Heart Disease')
else:
    print('The Person has Heart Disease')