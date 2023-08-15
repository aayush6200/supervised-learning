import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''
in this program we will calculate the hypothesis function
mean square error and then calculate gradient descent for our dataset
we will choose random w and b and then try to optimize them
'''
r_path = './dataset/advertise_data.csv'

df = pd.read_csv(r_path)  # reading dataset through our file

df = df.dropna()  # removing the row with missing/invalid values
df = df.drop_duplicates()  # dropping all the duplicates row

tv_expenses = df['TV']  # dataset_with tv
sales = df['sales']  # dataset with sales


def hypothesis_function(w, x, b):
    return np.dot(w, x)+b


def cost_function(y_pred, y_train, m):
    mse = (np.sum((y_train-y_pred)**2))/(2*m)
    return mse


def derivative_function(y_train, y_pred, x, m):
    dj_dw = np.sum((y_pred-y_train)*x) / m
    dj_db = -np.sum(y_train - y_pred) / m

    return [dj_dw, dj_db]


max_iterations = 1000


def gradient_function(w_old, dj_dw, b_old, dj_db, alpha):
    new_w = w_old-(alpha*(dj_dw))
    new_b = b_old-(alpha*(dj_db))

    return [new_w, new_b]


# lets suppose w and b in order to minimize it
w, b = 2, 2
for i in range(max_iterations):
    y_pred = hypothesis_function(w, tv_expenses, b)
    mean_squared_error = cost_function(y_pred, sales, len(tv_expenses))
    derivatives = derivative_function(
        sales, y_pred, tv_expenses, len(tv_expenses))
    dj_dw, dj_db = derivatives[0], derivatives[1]

    gradient = gradient_function(w, dj_dw, b, dj_db, 0.001)
    w, b = gradient[0], gradient[1]
    print(f"The new gradients are :{gradient[0]},{gradient[1]}")
