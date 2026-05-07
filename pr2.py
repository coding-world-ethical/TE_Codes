import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    r'C:\Users\CloudMorph\Downloads\AirQuality.csv',
    sep=';'
)

df.info()
df.describe()
df.head()
df.isnull().sum()
df['Unnamed: 15'].isnull().sum()
df['Unnamed: 16'].isnull().sum()

df.drop(
    ['Unnamed: 15', 'Unnamed: 16'],
    axis=1,
    inplace=True
)
df[df.isnull().any(axis=1)]
df.dropna(how='all', inplace=True)
column_list = df.columns.values.tolist()
print(column_list)

for column_name in column_list:
    print("\n", column_name)
    print(df[column_name].value_counts(dropna=False))

df.info()

print(df['CO(GT)'])
print(df['C6H6(GT)'])
print(df['RH'])
print(df['AH'])
print(df['T'])

j = 'CO(GT) C6H6(GT) T RH AH'.split()

print(j)

df.replace(
    to_replace=',',
    value='.',
    regex=True,
    inplace=True
)

for i in j:
    df[i] = pd.to_numeric(
        df[i],
        errors='coerce'
    )

df.head()
scaler = StandardScaler()

Numerical_col = df.select_dtypes(
    exclude=[np.object_, np.datetime64]
)

for col in Numerical_col:
    df[[col]] = scaler.fit_transform(df[[col]])

df.head()
df.info()

df['Date'] = pd.to_datetime(
    df['Date'],
    dayfirst=True
)

df['Time'] = pd.to_datetime(
    df['Time'],
    format='%H.%M.%S'
).dt.time

print(df.head())