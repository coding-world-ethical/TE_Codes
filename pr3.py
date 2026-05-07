import pandas as pd
import numpy as np
df = pd.read_csv(
    r'C:\Users\CloudMorph\Downloads\heart.csv'
)
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())
print(df.describe())

print(df.isnull().sum())
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print(df.shape)
print(df.duplicated().sum())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 6))

sns.heatmap(
    df.corr(),
    annot=True
)

plt.title(
    'Degree of Correlation of Variables in the Dataset'
)

plt.show()

sns.countplot(
    x='target',
    data=df
)

plt.xticks(
    [0, 1],
    ['Less Chance', 'More Chance']
)

plt.title('Chances of Heart Disease')
plt.show()
plt.figure(figsize=(15, 6))

print(df['target'].value_counts(normalize=True))

sns.countplot(
    x='sex',
    data=df
)

plt.title('Number of Males and Females')
plt.xticks(
    [0, 1],
    ['Females', 'Males']
)

plt.show()

sns.countplot(
    x='sex',
    data=df,
    hue='target'
)

plt.title('Chances of Heart Disease by Gender')

plt.xticks(
    [0, 1],
    ['Females', 'Males']
)

plt.legend(
    labels=['Less Chance', 'High Chance']
)

plt.show()

sns.histplot(df['cp'])

plt.xticks(
    [0, 1, 2, 3],
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ],
    rotation=70
)

plt.figure(figsize=(10, 7))

plt.show()

# Pie Chart
list1 = list(df['cp'].value_counts(normalize=True))

print(list1)

plt.pie(
    list1,
    labels=[
        "Typical Angina",
        "Non-Anginal",
        "Atypical Angina",
        "Asymptomatic"
    ],
    startangle=180,
    shadow=True,
    autopct='%1.1f%%'
)

plt.show()

sns.countplot(
    x='cp',
    hue='target',
    data=df
)

plt.title(
    'Relation between Types of Chest Pain and Chances of Heart Attack'
)

plt.xticks(
    [0, 1, 2, 3],
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ]
)

plt.legend(
    labels=['Low Chance', 'High Chance']
)

plt.show()

sns.countplot(
    x='fbs',
    hue='target',
    data=df
)

plt.legend(
    labels=['Low Chance', 'High Chance']
)

plt.show()

print(df['fbs'].value_counts(normalize=True))

g = sns.FacetGrid(
    df,
    hue="sex",
    aspect=4
)

g.map(
    sns.kdeplot,
    'trestbps',
    fill=True
)

plt.xlabel("Resting Blood Pressure")

plt.legend(
    labels=["Female", "Male"]
)

plt.show()

chol = sns.FacetGrid(
    df,
    hue="sex",
    aspect=4
)

chol.map(
    sns.kdeplot,
    'chol',
    fill=True
)

plt.xlabel("Serum Cholesterol")

plt.legend(
    labels=["Female", "Male"]
)

plt.show()