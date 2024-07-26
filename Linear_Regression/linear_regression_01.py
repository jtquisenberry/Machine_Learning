# Based on https://github.com/lazyprogrammer/machine_learning_examples/tree/master/linear_regression_class

# Import modules
import numpy as np
import matplotlib.pyplot as plt

# Load the data.
# CSVs came from thelazyprogrammer.
# Setup X and Y variables as Python lists.
X = []
Y = []
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# Convert Python lists to numpy arrays.
X = np.array(X)
Y = np.array(Y)

# Plot the data
plt.scatter(X, Y)
plt.show()


'''
BEGIN code linear regression calculations.
'''

# 1. Σ (x[i])^2
denominator1 = X.dot(X)

# 2. x-bar * Σ x[i]
denominator2 = X.mean() * X.sum()

# 3. Complete denominator
denominator = denominator1 - denominator2

# 4. Σ (x[i] * y[i])
anumerator1 = X.dot(Y)

# 5. y-bar * Σ (x[i])
anumerator2 = Y.mean() * X.sum()

# 6. Complete anumerator
anumerator = anumerator1 - anumerator2

# 7. y-bar Σ (x[i]^2)
bnumerator1 = Y.mean() * X.dot(X)

# 8. x-bar Σ (x[i] * y[i])
bnumerator2 = X.mean() * X.dot(Y)

# 9. Complete bnumerator
bnumerator = bnumerator1 - bnumerator2

# 10. Calculate a
a = anumerator / denominator

# 11. Calculate b
b = bnumerator / denominator


# let's calculate the predicted Y
Yhat = a*X + b

# let's plot everything together to make sure it worked
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# determine how good the model is by computing the r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)

'''
END code linear regression calculations.
'''
