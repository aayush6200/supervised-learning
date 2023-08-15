import numpy as np
import pandas as pd


# getting our file
r_path = './dataset/advertise_data.csv'

df = pd.read_csv(r_path)
print(df)
df = df.dropna()    # cleaning row with missing values
df = df.drop_duplicates()   # cleaning row with missing values

# creates a new dataset with two columns named TV and sales
new_df = df[['TV', 'sales']]




