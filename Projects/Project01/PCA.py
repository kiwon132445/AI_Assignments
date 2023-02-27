import pandas as pd
import numpy as np

dataset = "UNSW-NB15-BALANCED-TRAIN.csv"
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# Comma delimination
df = pd.read_csv(dataset, header = 0)

# Put the original column values in a python list
original_headers = list(df.columns.values)

# Remove the non-numeric columns
df = df._get_numeric_data()

numeric_headers = list(df.columns.values)

numpy_array = df.as_matrix()

print(df)