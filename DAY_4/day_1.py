import numpy as np


# vector creation

a = np.zeros(4)  # prints a vector with 4 zeros
print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,))   # prints a vector with 4 zeros
print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# prints a vector with 4 random sample between 0(inclusive) and 1(exclusive)
a = np.random.random_sample(4)
print(
    f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")


# indexing and looping inside numpy arrays

a = np.arange(10)

print(a)  # prints the vector upto 10 elements starting from 0(inclusive)-9(exclusive)

print(a[2])  # prints 3rd index element starting from 0 index


print(a[-1])  # prints the last index element starting from -1


# slicing


# access 5 consecutive elements (start:stop:step)
c = a[2:7:1]
print("a[2:7:1] = ", c)

# access 3 elements separated by two
c = a[2:7:2]
print("a[2:7:2] = ", c)

# access all elements index 3 and above
c = a[3:]
print("a[3:]    = ", c)

# access all elements below index 3
c = a[:3]
print("a[:3]    = ", c)

# access all elements
c = a[:]
print("a[:]     = ", c)


# vector single wise operation


a = np.array([1, 2, 3, 4])
print(f"a             : {a}")
# negate elements of a, b=[-1,-2,-3,-4]
b = -a
print(f"b = -a        : {b}")

# sum all elements of a, returns a scalar
b = np.sum(a)
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")


# return the squares of each element of a
b = a**2
print(f"b = a**2      : {b}")


# vector vector dot product


a = np.arange(5)
b = np.arange(5, 10)

print(f"a = {a} ,b= {b}")


# using dot operations between a and b

a_dot_b = np.dot(a, b)

print(f'The dot product of a and b is :{a_dot_b}')
