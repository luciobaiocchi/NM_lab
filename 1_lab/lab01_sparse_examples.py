import numpy as np
import scipy as sp
import scipy.sparse as spsp
import matplotlib.pyplot as plt

# "FALSE" SPARSE MATRIX

A = np.round(np.random.rand(5, 5) - 0.15) * np.random.rand(5, 5)
A = A + np.diag(np.random.rand(5), 0)
# DENSITY OF A
dnst = A.nonzero()[0].size / A.size

print(f'**** A "FALSE" SPARSE MATRIX (density={dnst})****')
print('')
print(A)


# TRANSLATE A "FALSE" SPARSE MATRIX INTO A "TRUE" SPARSE ONE

#### "TRUE" SPARSE MATRIX, OBTAINED FROM A
#
# "DICTIONARY-OF-KEYS" STORAGE
Asp = spsp.dok_array(A);
#
## OTHER STORAGE METHODS, GOOD FOR MATRIX MULTIPLICATION: 
# spsp.csc_array(A)
# spsp.csr_array(A)


print('**** SAME MATRIX, BUT "TRULY" SPARSE ****')
print('')
print(Asp)

# VISUALIZATION OF A SPARSE MATRIX

plt.figure()
plt.spy(Asp, markersize=3)


# SUBMATRICES OF SPARSE MATRICES

print('**** SUBMATRICES OF A SPARSE MATRIX AND BACK TO DENSE ****')
print('')
print('--- Asp[:, 1]')
print(Asp[:, [1]])
print('... I.E.: ')
print(Asp[:, [1]].todense())
print('')
print('--- Asp[1, :]')
print(Asp[[1], :])
print('... I.E.: ')
print(Asp[[1], :].todense())
print('')
print('--- Asp[[2, 4], [1:3]]')
print(Asp[[1, 3], 0:2])
print('... I.E.: ')
print(Asp[[1, 3], 0:2].todense())
print('')

# OPERATIONS WITH SPARSE MATRICES

Asp_square = Asp @ Asp

print('**** SPARSE with SPARSE returns SPARSE ****')
print('')
print(f'--- Asp @ Asp (density={Asp_square.size / np.prod(Asp_square.shape)}) ---')
print(' ')
print(Asp_square)

rand_scal = np.random.rand() - 0.5
Asp_scal = Asp + rand_scal * Asp
print(f'--- Asp + scalar * Asp (density={Asp_scal.size / np.prod(Asp_scal.shape)}) ---')
print(' ')
print(Asp_scal)

Asp_linsolve = spsp.linalg.spsolve(Asp.tocsr(), Asp.tocsr())
print(f'--- Solution of Asp @ X = Asp (density={Asp_linsolve.size / np.prod(Asp_linsolve.shape)}) ---')
print('')
print(Asp_linsolve)

Asp_linsolve_vec = spsp.linalg.spsolve(Asp.tocsr(), spsp.dok_array(np.expand_dims(Asp.sum(axis=1), axis=1)))
print('---** EXEPTION: spsp.linalg.spsolve(spMATRIX, spVECTOR) returns dense vectors **---')
print(f'--- Solution of Asp @ x = Asp.sum(axis=1) (density={Asp_linsolve_vec.nonzero()[0].size / Asp_linsolve_vec.size}) ---')
print('')
print(Asp_linsolve_vec)


Asp_A = Asp @ A
print('**** SPARSE with DENSE (and vice-versa) returns DENSE ****')
print(' ')
print(f'--- Asp @ A (density={Asp_A.nonzero()[0].size / Asp_A.size}) ---')
print('')
print(Asp_A)

Asp_scal_dense = Asp + rand_scal * A
print(f'--- Asp + scalar * A (density={Asp_scal_dense.nonzero()[0].size / Asp_scal_dense.size}) ---')
print('')
print(Asp_scal_dense)

Asp_linsolve_dense = spsp.linalg.spsolve(Asp.tocsr(), A)
print(f'--- Solution of Asp @ X = A (density={Asp_linsolve_dense.nonzero()[0].size / Asp_linsolve_dense.size}) ---')
print('')
print(Asp_linsolve_dense)


# INITIALIZATION/GENERATION OF SPARSE MATRICES

N = int(1e4);

print('**** INITIALIZATION/GENERATION OF SPARSE MATRICES ****')
print('')
print(f'--- All-zeros sparse matrix ({N ** 2} elements)')
print(spsp.dok_array((N, N)))
print('')
print('--- Random sparse matrix (density=0.001)')
plt.figure()
plt.spy(spsp.rand(N, N, 0.001), markersize=3)
print('')
print(f'--- Diagonal sparse matrix ({N} non-zero elements)')
# print(spsp.dia_array(([np.random.rand(N)], [0]), (N, N)))

plt.figure()
plt.spy(spsp.dia_array(([np.random.rand(N)], [0]), (N, N)), markersize=3)
print('')


plt.show()


