# Visualize the data using Python libraries matplotlib, seaborn by plotting the graphs for Air quality
#and Heart Diseases datasets
# Dataset( Heart Disease) : https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset


import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv(
    r'C:\Users\CloudMorph\Downloads\heart.csv'
)

# Dataset Information
print(df.shape)

print(df.columns)

print(df.dtypes)

print(df.info())

print(df.describe())

# Check Missing Values
print(df.isnull().sum())

# No null values

# Check Duplicate Values
print(df.duplicated().sum())

# Remove Duplicate Values
df.drop_duplicates(inplace=True)

# Shape after removing duplicates
print(df.shape)

# Check duplicates again
print(df.duplicated().sum())

# =========================================
# HEATMAP VISUALIZATION
# =========================================

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 6))

sns.heatmap(
    df.corr(),
    annot=True
)

# Optional heatmap styling
# sns.heatmap(
#     df.corr(),
#     annot=True,
#     linewidths=0.5,
#     fmt=".2f",
#     cmap="YlGnBu"
# )

plt.title(
    'Degree of Correlation of Variables in the Dataset'
)

plt.show()

# =========================================
# COUNTPLOT OF TARGET VARIABLE
# =========================================

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

# =========================================
# COUNTPLOT OF GENDER
# =========================================

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

# =========================================
# HEART DISEASE BY GENDER
# =========================================

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

# =========================================
# CHEST PAIN DISTRIBUTION
# =========================================

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

# =========================================
# CHEST PAIN VS TARGET
# =========================================

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

# =========================================
# FBS VS TARGET
# =========================================

sns.countplot(
    x='fbs',
    hue='target',
    data=df
)

plt.legend(
    labels=['Low Chance', 'High Chance']
)

plt.show()

# Percentage of diabetic and non-diabetic people
print(df['fbs'].value_counts(normalize=True))

# =========================================
# RESTING BLOOD PRESSURE VISUALIZATION
# =========================================

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

# =========================================
# SERUM CHOLESTEROL VISUALIZATION
# =========================================

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