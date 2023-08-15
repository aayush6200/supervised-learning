import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# this file will be used to analysis the single variable linear regression(TV vs sales)
r_path = './dataset/advertise_data.csv'

df = pd.read_csv(r_path)


new_df = df[['TV', 'sales']]


# cleaning our dataset
new_df = new_df.dropna()
new_df = new_df.drop_duplicates()

tv_expense = np.array(new_df['TV'])  # creates an array of tv_expense
sales = np.array(new_df['sales'])


def hypothesis_function(x, w, b):
    y = np.dot(w, x) + b
    return y


def costs_function(y_pred, y_train, m):
    '''
    This function returns the mean square error 
    y_pred: The predicted data by our model.
    y_train: The trained data that was given to our model. Its a predetermined output
    m: The length of input which in our case is  len(y_train)/len(tv_expenses)
    '''
    return np.sum((y_pred-y_train)**2)/m  # return the mean squared error


# for sake of simplicity, lets start with w=0.005 and b=2
w = 0.07
b = 2

y_pred = hypothesis_function(tv_expense, w, b=2)

print('The predicted sales data is:', y_pred)

calculated_mse = costs_function(y_pred, sales, len(tv_expense))
print('The predicted mse is :', calculated_mse)


'''
for sake of understanding lets create a graph 
plotting the linear regression obtained from our model
'''

# plt.scatter(tv_expense, sales)
# plt.scatter(tv_expense, y_pred)

# plt.show()


w_values = []
mse_values = []
for i in np.arange(0.00001, 0.1, 0.0001):

    '''we will try to use as many value of w as possible keeping b constant
        we will store the w  and calculated mse in w_value and mse_value arrays respectively
    '''

    w = i
    y_pred = hypothesis_function(tv_expense, w, b=2)

    print('The predicted sales data is:', y_pred)

    calculated_mse = costs_function(y_pred, sales, len(tv_expense))
    w_values.append(w)
    mse_values.append(calculated_mse)
    print('The predicted mse is :', calculated_mse)


'''
lets calculate all the mse and the plot the graph for mse analysis

'''

plt.scatter(w_values, mse_values)
plt.xlabel('weight/slope(w)')
plt.ylabel('mean squared error')
plt.show()
'''
lets start with a random w and b
lets use gradient descent to calculate the optimal value of w and b
for sake of simplicity l

'''


'''


'''
