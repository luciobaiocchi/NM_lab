import numpy as np

# INITIALIZATION


A = np.array([[0.835, 0.667], [0.333, 0.266]])
b_true = np.array([[0.168], [0.067]])

l_err_meas = -0.001
u_err_meas = 0.001

err_measures = np.linspace(l_err_meas, u_err_meas, 11)

Bmeasured = b_true + err_measures

x_true = np.array([[1], [-1]])

# SOLVING THE LINEAR SYSTEM

X = np.linalg.solve(A, Bmeasured)
Rnorms = np.linalg.norm(A @ X - Bmeasured, axis=0)

# ANALYSIS & COMMENTS

print('**** We have a square linear system Ax=b s.t. A is:')
print('')
print(A)
print('')
print(' while the vector of known terms is given by the results of an experiments ****')
print('')

print('**** The instrument used for measuring b has a tolerance of +-0.001 ****')
print('**** The true value of b would be: ')
print('')
print(b_true)
print('')
print(' but the measurements obtained are the following (one per column):')
print('')
print(Bmeasured)
print('')

print('**** Solving the linear system for each measurement, we obtain the solutions')
print('')
print(X)
print('')
print(' with residuals of norms')
print('')
print(Rnorms)
print('')

print(f'**** THE REASON OF THIS BEHAVIOUR IS THAT k(A) = {np.linalg.cond(A)} >> 1 ****')








