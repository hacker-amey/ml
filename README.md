import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

df = pd.read_csv("~/Downloads/adult.csv", na_values="?")
print(df.head())

df.fillna(df.mode().iloc[0], inplace=True)
print("Missing values:", df.isnull().sum().sum())

le = LabelEncoder()
for col in df.select_dtypes(include="object"):
    df[col] = le.fit_transform(df[col])
print(df.head(3))

normalized = MinMaxScaler().fit_transform(df)
print(normalized[:3])

standardized = StandardScaler().fit_transform(df)
print(standardized[:3])

print("Encoded Data:\n", df.head(3))
print("\nNormalized:\n", normalized[:3])
print("\nStandardized:\n", standardized[:3])
print("\nMissing Values:", df.isnull().sum().sum())