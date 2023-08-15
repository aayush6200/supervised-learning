import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


r_path = './dataset/advertise_data.csv'

# reading the dataset
df = pd.read_csv(r_path)
# cleaning the dataset
df = df.dropna()  # cleaning the row
df = df.drop_duplicates()  # removing duplicates

# filtering the required data and the converting to arrays
tv_expenses = np.array(df['TV'])
radio_expenses = np.array(df['radio'])
newspaper_expenses = np.array(df['newspaper'])
sales = np.array(df['sales']).reshape(-1, 1)

x = np.array([tv_expenses, radio_expenses, newspaper_expenses]).T
length = len(x)

# Initialize weights
num_features = x.shape[1]

'''we will try to calculate gradient descent for multiple variables
Algorithm:
start with random w and y values
firstly use hypothesis function to get prediction f_w_b_(x)
then use f_w_b_(x) to calculate cost function
then use cost function to calculate gradient descent
'''


def hypothesis_function(w, x, b):
    return np.dot(x, w)+b


def cost_function(y_pred, y_train, m):
    cost = np.sum((y_train-y_pred)**2)/len(y_train)
    return cost


def derivatives_function(y_train, y_pred, x, m):
    # compute derivatives
    print('x shape', x.shape)

    print('y shape', (y_pred).shape)
    print('y shape', (y_train).shape)
    print('y shape', (y_pred-y_train).shape)
    dj_dw = np.dot(x.T, y_pred - y_train) / m
    dj_db = np.sum(y_pred - y_train) / m

    return dj_dw, dj_db


'''To calculate the gradient function we will try to see if w and b converges to certain degree
To check if they converges, we will try to see if the difference between prev w and prev b converges

for learning rate we will try to use an three values to determine the best value

formula for gradient descent is 
w=w-alpha*dj_dw
b=b-alpha*dj_db

'''


def gradient_function(w, b, dj_dw, dj_db, alpha):
    w_new = w-alpha*dj_dw
    b_new = b-alpha*dj_db

    return w_new, b_new


alpha_1 = 0.000001
w, b = np.array([0.1, 0.1, 0.1]).reshape(-1, 1), 2
print(w)  # shape (3,1)

iterations = 10000
total_cost = []

for i in range(iterations):
    y_pred = hypothesis_function(w, x, b)
    total_cost.append(cost_function(y_pred, sales, len(y_pred)))

    dj_dw, dj_db = derivatives_function(sales, y_pred, x, len(y_pred))
    # print('dj_dw shape', dj_dw.shape)
    w, b = gradient_function(w, b, dj_dw, dj_db, alpha_1)
    print(w)
total_cost_array = np.array(total_cost)
print(total_cost_array.shape)
print(x.shape)

plt.plot(np.array(range(iterations)), total_cost_array, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.title('Cost Function over Iterations')
plt.show()
