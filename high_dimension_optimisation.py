import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def wobble(x, y, c):
    """ This function defines a surface with a bump as (x, y) with centre c (2 tuple)"""
    d1 = x - c[0]
    d2 = y - c[1]

    d = d1**2 + d2**2
    z = np.cos(0.5*d)*np.exp(-0.1*d)

    return z


def my2dtest(x_vect):
    """ test function that combines 3 wobble functions together by adding them together"""
    x = x_vect[0]
    y = x_vect[1]

    c1 = (1, 2)
    c2 = (-2, 1)
    c3 = (2.5, -0.8)

    w1 = 1.0
    w2 = 0.9
    w3 = 0.6

    z = w1 * wobble(x, y, c1) + w2 * wobble(x, y, c2) + w3 * wobble(x, y, c3)

    return z


# plot surface
v = np.linspace(-5, 5, 1000)
xm, ym = np.meshgrid(v, v)
z = my2dtest([xm, ym])
#
# f = plt.figure()
# ax = f.gca(projection='3d')
# ax.plot_surface(xm, ym, z)
# plt.show()

fig, ax = plt.subplots()

c = ax.pcolormesh(xm, ym, z, cmap='RdBu')
fig.colorbar(c, ax=ax)
plt.show()


# Comparing multiple local optimisers (requires start value)
optimisers = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

# initialise problem domain mesh
resolution = 100
v = np.linspace(-5, 5, resolution)
xm, ym = np.meshgrid(v, v)

for opt_method in optimisers:
    opt_val = np.zeros(xm.shape)
    for i in range(xm.shape[0]):
        for j in range(xm.shape[1]):
            x = xm[i, j]
            y = ym[i, j]
            x_opt = optimize.minimize(my2dtest, [x, y], method=opt_method)
            opt_val[i, j] = x_opt.fun

    plt.figure()
    plt.imshow(opt_val)
    plt.title(opt_method)
    plt.show()
