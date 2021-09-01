import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


# Define univariate Function
def poly(x):
    return x * (x - 1) * (x + 1) * (x - 4)


x = np.linspace(-2, 4.5, 100)
y = poly(x)
plt.figure()
plt.plot(x, y)
plt.show()

# univariate optimisation to find lowest value of the function
x_opt = optimize.minimize_scalar(poly)
print(x_opt)

# Output:
# fun: -24.05727870023589
# nfev: 16
# nit: 11
# success: True
# x: 3.0565452919448806

# multivariate optimisation method using a start value
x_opt = optimize.fmin(poly, -2)
print(x_opt)

# Output:
# Optimization terminated successfully.
#          Current function value: -1.766408
#          Iterations: 17
#          Function evaluations: 34
# [-0.60097656]
# => finds local minima but not global minimum

# multivariate optimisation method using a Better start value
x_opt = optimize.fmin(poly, 1)
print(x_opt)

# Output:
# Optimization terminated successfully.
#          Current function value: -24.057279
#          Iterations: 19
#          Function evaluations: 38
# [3.05654297]
# => finds correct global minimum as shown in first optimiser

