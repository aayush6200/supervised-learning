import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


r_path = './dataset/advertise_data.csv'

df = pd.read_csv(r_path)  # creates a dataset for us

# for shake of  simplicity we will create a dataset with TV expenses  and sales generated
# Tv expenses will be considered as independent and sales will be dependent

new_df = df[['TV', 'sales']]  # creates a new dataset with TV and sales column


# cleaning our dataset in two steps: row cleanup and duplicates removal

new_df = new_df.dropna()  # row cleanup removes any row with missing or no values

new_df = new_df.drop_duplicates()  # removes any duplicate rows

# lets create a scatter plot of two datasets where TV is x axis and  sales is y axis

tv_expenses = np.array(new_df['TV'])
sales = np.array(new_df['sales'])

# plt.scatter(tv_expenses, sales, label='Actual sales')


# print(new_df)


# lets predict sales using our model


# def hypothesis_function(x, w, b):
#     '''
#     The model will use the concept of linear regression
#     y_pred=wx+b
#     for sake of simplicity, bias(b)=0 and w=0.5
#     we will create hypothesis function  to calculate our model output
#     lastly we will calculate mean squared error
#     '''

#     y = np.dot(w, x)+b
#     return y

# # lets calculate the mean squared error


# def MSE_function(y_pred, y_target, m):
#     '''The model will calculate Mean squared error
#     helping us better understand the prediction accuracy of
#     our model
#     y_pred signifies our predicted values by model
#     y_target represents the initial target values during training
#     m is the length of y_pred which is similar to y_target
#     mean squared error=1/m(for loop till range(m): sum of (y_pred, y_target)^2)'''
#     Sum = 0
#     for i in range(m):
#         return np.sum((y_pred - y_target) ** 2) / m


# # y_pred = hypothesis_function(tv_expenses, w=0.09, b=2)
# # print(y_pred)
# # Arrays to store MSE and w values
# mse_values = []
# w_values = []

# for i in range(0, 5, 20):
#     val = i/10.0
#     w = val
#     y_pred = hypothesis_function(tv_expenses, w, b=2)
#     calculated_mse = MSE_function(y_pred, sales, len(y_pred))
#     w_values.append(w)
#     mse_values.append(calculated_mse)
# print(w_values)
# print(mse_values)
# for i in range(len(w_values)):
#     plt.scatter(w_values[i], mse_values[i], label='MSE')
# plt.legend()
# plt.xlabel('W')
# plt.ylabel('Jw')
# plt.show()

def hypothesis_function(x, w, b):
    y = np.dot(w, x) + b
    return y


def MSE_function(y_pred, y_target, m):
    return np.sum((y_pred - y_target) ** 2) / m


def gradient_decent(dj_dw, w, alpha):
    w_new = w-alpha*(dj_dw)

    return w_new


# Arrays to store MSE and w values
mse_values = []
w_values = []
w_grad_decent = []
mse_grad_decent = []

# Range for 'w'
for i in np.arange(0.01, 0.11, 0.01):
    w = i

    # Calculate predicted sales and MSE
    y_pred = hypothesis_function(tv_expenses, w, b=2)
    calculated_mse = MSE_function(y_pred, sales, len(tv_expenses))
    print(f'For w = {w}, the resulting MSE is: {calculated_mse:.2f}')

    # Append values to arrays
    w_values.append(w)
    mse_values.append(calculated_mse)


# Plot MSE as 'w' varies
plt.scatter(w_values, mse_values)
plt.xlabel('W')
plt.ylabel('J(w)')
plt.title('MSE vs. "w" values')
plt.legend()
plt.grid(True)
plt.show()


# lets work with gradient descent

'''lets take the w that minimizes mse for sake of simplicity'''

min_mse = min(mse_values)
min_w = min(w_values)
for i in range(len(mse_values)):
    if mse_values[i] == min_mse:
        min_w = w_values[i]
print('hello world', min_w)
# y_pred_final = hypothesis_function(tv_expenses, min_w, b=2)
# plt.scatter(tv_expenses, sales)
# plt.scatter(tv_expenses, y_pred_final)
# plt.xlabel('tv_expenses')
# plt.ylabel('model prediction')
# plt.show()

# y_pred = hypothesis_function(tv_expenses, min_w, b=2)
# dJ_dw = np.sum((y_pred-sales)*tv_expenses) / len(tv_expenses)
convergence_threshold = 1e-6
while i != 1000:
    y_pred = hypothesis_function(tv_expenses, min_w, b=2)
    dJ_dw = np.sum((y_pred-sales)*tv_expenses) / len(tv_expenses)
    new_w = gradient_decent(dJ_dw, min_w, 0.000000001)

    calculated_mse = MSE_function(y_pred, sales, len(tv_expenses))
    # Stopping criterion

    min_w = new_w
    print(new_w)
    w_grad_decent.append(min_w)
    mse_grad_decent.append(calculated_mse)
    i += 1
print(f"Final value of 'w': {min_w}")
y_pred_final = hypothesis_function(tv_expenses, min_w, b=2)
plt.scatter(tv_expenses, sales)
plt.scatter(tv_expenses, y_pred_final)
plt.xlabel('tv_expenses')
plt.ylabel('model prediction')
plt.show()

print(
    f"Final MSE: {MSE_function(hypothesis_function(tv_expenses, min_w, b=2), sales, len(tv_expenses)):.2f}")


plt.scatter(mse_grad_decent, w_grad_decent)
plt.xlabel('W')
plt.ylabel('J(w)')
plt.title('MSE vs. "w" values')
plt.legend()
plt.grid(True)
plt.show()
