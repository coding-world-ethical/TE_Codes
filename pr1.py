import pandas as pd
import numpy as np
df = pd.read_csv(
    r'C:\Users\CloudMorph\Downloads\dataset_Facebook.csv',
    sep=';'
)
print("First 5 Rows:\n")
print(df.head())

subset1 = df.loc[df["Type"] == "Photo"]

subset1.to_csv(
    r'C:\Users\CloudMorph\Downloads\photo_data.csv',
    index=False
)
print("\nPhoto subset saved successfully!")
temp = df.iloc[10:25]
print("\nRows from 10 to 24:\n")
print(temp)

df1 = df[['Type', 'Category', 'comment']].loc[4:17]

df2 = df[['Type', 'Category', 'Paid']].loc[24:30]

df3 = df[['Type', 'Category', 'Paid']].loc[31:35]

print("\nDF1:\n")
print(df1)

print("\nDF2:\n")
print(df2)

print("\nDF3:\n")
print(df3)

merged_df = pd.concat([df1, df2, df3])

print("\nMerged Data:\n")
print(merged_df)

sorted_df = df.sort_values(
    ['Category', 'Paid'],
    ascending=False
)

print("\nSorted Data:\n")
print(sorted_df.head())

transposed_df = df.transpose()

print("\nTransposed Data:\n")
print(transposed_df)

df_melted = df1.melt(
    id_vars=None,
    value_vars=None,
    ignore_index=False
)

print("\nMelted Data:\n")
print(df_melted)

df_pivoted = df_melted.pivot(
    columns='variable',
    values='value'
)

print("\nPivoted Data:\n")
print(df_pivoted)