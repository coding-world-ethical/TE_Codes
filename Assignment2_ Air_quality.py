#Assignment 2. Perform the following operations using Python on the Air quality and Heart
#Diseases data sets
# a. Data cleaning
# b. Data integration
# c. Data transformation
# d. Error correcting
# e. Data model building
# Dataset (Air quality): https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv(
    r'C:\Users\CloudMorph\Downloads\AirQuality.csv',
    sep=';'
)

# Check column names and datatypes
df.info()

# Statistical information of dataset
df.describe()

# Display top 5 rows
df.head()

# Check missing values
df.isnull().sum()

# Last two columns contain all NaN values
df['Unnamed: 15'].isnull().sum()
df['Unnamed: 16'].isnull().sum()

# Drop unwanted columns
df.drop(
    ['Unnamed: 15', 'Unnamed: 16'],
    axis=1,
    inplace=True
)

# Display rows containing NaN values
df[df.isnull().any(axis=1)]

# Remove rows where all values are NaN
df.dropna(how='all', inplace=True)

# Display column names
column_list = df.columns.values.tolist()
print(column_list)

# Display unique values in each column
for column_name in column_list:
    print("\n", column_name)
    print(df[column_name].value_counts(dropna=False))

# =====================================
# ERROR CORRECTION
# =====================================

# Display dataset information
df.info()

print(df['CO(GT)'])
print(df['C6H6(GT)'])
print(df['RH'])
print(df['AH'])
print(df['T'])

# List of columns
j = 'CO(GT) C6H6(GT) T RH AH'.split()

print(j)

# Replace comma with decimal point
df.replace(
    to_replace=',',
    value='.',
    regex=True,
    inplace=True
)

# Convert columns to numeric datatype
for i in j:
    df[i] = pd.to_numeric(
        df[i],
        errors='coerce'
    )

# =====================================
# DATA TRANSFORMATION
# =====================================

df.head()

# Standardization
scaler = StandardScaler()

Numerical_col = df.select_dtypes(
    exclude=[np.object_, np.datetime64]
)

# Apply StandardScaler
for col in Numerical_col:
    df[[col]] = scaler.fit_transform(df[[col]])

df.head()

df.info()

# Convert Date column
df['Date'] = pd.to_datetime(
    df['Date'],
    dayfirst=True
)

# Convert Time column
df['Time'] = pd.to_datetime(
    df['Time'],
    format='%H.%M.%S'
).dt.time

# Final Output
print(df.head())