import numpy as np
import scipy.sparse as spsp
import matplotlib.pyplot as plt

# INITIALIZATION

n_values = range(100,1001,200)
num_n = len(n_values);

A_densities = np.zeros((num_n, 1))
L_densities = np.zeros((num_n, 1))
U_densities = np.zeros((num_n, 1))


# COMPUTATION OF THE DENSITIES

for i in range(num_n):
    n = n_values[i];
    A = spsp.rand(n, n, 0.1)
    A_densities[i] = A.nonzero()[0].size / np.prod(A.shape)
    lu = spsp.linalg.splu(A)
    L_densities[i] = lu.L.nonzero()[0].size / np.prod(lu.L.shape)
    U_densities[i] = lu.U.nonzero()[0].size / np.prod(lu.U.shape)


# PLOTS

plt.figure(1)
plt.title('Densities')
plt.plot(n_values, A_densities, label='A')
plt.plot(n_values, L_densities, label='L')
plt.plot(n_values, U_densities, label='U')
plt.xlabel('N')
plt.ylabel('Density')
plt.legend()

plt.figure(2)
plt.title('Matrix A')
plt.spy(A, markersize=3)

plt.figure(3)
plt.title('Matrix L')
plt.spy(lu.L, markersize=3)

plt.figure(4)
plt.title('Matrix U')
plt.spy(lu.U, markersize=3)



plt.show()

