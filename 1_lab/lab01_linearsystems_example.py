import numpy as np

# INITIALIZATION

np.random.seed(1)

n = 5
N = 10

A = np.random.rand(n, n)
b = np.expand_dims(A.sum(axis=1), axis=1)
B = b + np.random.rand(n, N)

d = np.random.rand(n)

D = np.diag(d, 0)

print('**** A random matrix for a square linear system ****')
print('')
print(A)
print('')
print('**** A vector of known terms s.t. the sol. of Ax=b is x*=(1,...,1) ****')
print('')
print(b)
print('')

print('**** A matrix of many known terms adding random values vector b ****')
print('')
print(B)
print('')

print('**** A random diagonal matrix ****')
print('')
print(D)
print('')


# SOLVING THE LINEAR SYSTEMS


x = np.linalg.solve(A, b)
X = np.linalg.solve(A, B)

xd = b / np.expand_dims(d, axis=1)


print('**** Solution for Ax=b ****')
print('')
print(x)
print(f' with residual norm: {np.linalg.norm(A @ x - b)}')
print('')
print('**** Solutions for Ax=b1, ..., Ax=bN (i.e., AX=B) ****')
print('')
print(X)
print(f' with residual norms: {np.linalg.norm(A @ X - B, axis=0)}')
print('')
print('**** Solution for Dx=b ****')
print('')
print(xd)
print(f' with residual norm: {np.linalg.norm(D @ xd - b)}')
print('')

print(f'**** THE SOLUTIONS ARE RELIABLE BECAUSE k(A) = {np.linalg.cond(A)} AND k(D) = {np.linalg.cond(D)} ARE NOT TOO LARGE ****')



