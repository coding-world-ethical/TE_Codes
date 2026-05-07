#Assignment 1. Perform the following operations using Python on the Facebook metrics data sets
# a. Create data subsets
# b. Merge Data
# c. Sort Data
# d. Transposing Data
# e. Shape and reshape Data
#
# Download Dataset (Facebook metrics) : from following link
# https://www.kaggle.com/datasets/masoodanzar/facebook-metrics

import pandas as pd
import numpy as np

# Read CSV file
df = pd.read_csv(
    r'C:\Users\CloudMorph\Downloads\dataset_Facebook.csv',
    sep=';'
)

# Display first 5 rows
print("First 5 Rows:\n")
print(df.head())

# =========================================================
# a. Create Data Subsets
# =========================================================

# Create subset where Type = Photo
subset1 = df.loc[df["Type"] == "Photo"]

# Save subset to CSV
subset1.to_csv(
    r'C:\Users\CloudMorph\Downloads\photo_data.csv',
    index=False
)

print("\nPhoto subset saved successfully!")

# Create subset by selecting rows based on position
temp = df.iloc[10:25]

print("\nRows from 10 to 24:\n")
print(temp)

# Create subsets using specific columns and rows
df1 = df[['Type', 'Category', 'comment']].loc[4:17]

df2 = df[['Type', 'Category', 'Paid']].loc[24:30]

df3 = df[['Type', 'Category', 'Paid']].loc[31:35]

print("\nDF1:\n")
print(df1)

print("\nDF2:\n")
print(df2)

print("\nDF3:\n")
print(df3)

# =========================================================
# b. Merge Data
# =========================================================

merged_df = pd.concat([df1, df2, df3])

print("\nMerged Data:\n")
print(merged_df)

# =========================================================
# c. Sort Data
# =========================================================

sorted_df = df.sort_values(
    ['Category', 'Paid'],
    ascending=False
)

print("\nSorted Data:\n")
print(sorted_df.head())

# =========================================================
# d. Transpose Data
# =========================================================

transposed_df = df.transpose()

print("\nTransposed Data:\n")
print(transposed_df)

# =========================================================
# e. Shape and Reshape Data
# =========================================================

# Melt DataFrame
df_melted = df1.melt(
    id_vars=None,
    value_vars=None,
    ignore_index=False
)

print("\nMelted Data:\n")
print(df_melted)

# Pivot DataFrame
df_pivoted = df_melted.pivot(
    columns='variable',
    values='value'
)

print("\nPivoted Data:\n")
print(df_pivoted)