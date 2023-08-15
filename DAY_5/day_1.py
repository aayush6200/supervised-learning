import pandas as pd
import numpy as np
import os
from os.path import join, dirname


# get current directory and required directories

current_dir = os.getcwd()
print(current_dir)
r_path = join(current_dir, 'housing.csv')


# reading our data from the csv file

df = pd.read_csv(r_path)
first_row = df.iloc[0]
print(first_row)


# cleaning our dataset

# dropping our rows with missing/none data
df = df.dropna()
# dropping our rows with duplicates values

df = df.drop_duplicates()

print(df.iloc[0])

# lets create a model
# train model to predict house information

# f(x)=ax_1+bx_2+cx_3+df

# getting the info
longitude, latitude, age, rooms, bedrooms, income = df['longitude'], df['latitude'], df[
    'housing_median_age'], df['total_rooms'], df['total_bedrooms'], df['median_income']


# lets convert all of these into the array

x = np.array([longitude, latitude, age, rooms, bedrooms, income])
print(x.shape)

# defining w's
w_values = x.shape[1]

# creating the numpy for w and transposing it
w = np.array([[np.random.randint(0, 7) for x in range(1, w_values+1, 1)]]).T

# defining b and alpha values
b = 2
alpha = 0.000001


def hypothesis_function(w, x, b):
    y = np.dot(x, w)
    pass


def cost_function():
    pass


def derivatives_function():
    pass


def gradient_descent():
    pass


y_pred = hypothesis_function()
