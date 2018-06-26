import numpy as np
from augmented_lagrangian import augmented_lagrangian, _Resid_vec
import time
from scipy.special import comb
from multivariate_monomials import generate_Amat
from numpy import linalg as LA
from BMsgd import BM_SGD

# Generate a random positive semidefinite matrix Q
def make_random_Q(n):

    mat = np.random.randn(n, n)
    mat = np.matmul(mat, np.transpose(mat))
    
    return mat

# degree: degree of p(x), must be an even number
degree = 8

# deg: maximum degree of monomials in [x]_d
deg = int(degree/2)

# numVar: number of variables
numVar = 4
# n: length of [x]_d; Also the dimention of X
n = comb(numVar + deg, deg, exact = True)
print(n)

# m: number of coefficients in p(x); number of constraints
m = comb(numVar + degree, degree, exact = True)

# r_max: Maximum number of ranks needed s.t. r(r+1)/2 <= m
# if this assumption holds.
r_max = int((np.sqrt(1+8*m) - 1)/2) + 1

# r: the rank of R with the substition X = RR^T
r = int(r_max/2)
r = numVar + 1
r = r_max

# Make constraint matrices such that <A_i,X> = p_i, i = 1,...,m
A_mats = generate_Amat(deg,numVar)
    
#coefficients
#p = np.array([10, 8, 0, -2, 1])
#p = np.array([11, 34, 51, -2, 16, -2, 1])

Q = make_random_Q(n)
#Q = np.random.randn(Q_deg + 1, Q_deg + 1)

p = np.zeros(m)
for i in range(m):
    val = np.trace(np.matmul(np.transpose(A_mats[i]),Q))
    p[i] = val
    
p = p.reshape((-1,1))

#print(A_mats)
start = time.time()

R_init = np.random.uniform(-1, 1, size=(n, r))
R = BM_SGD(A_mats, R_init, p, step = 0.0001, maxsteps = 0, precision = 0.0001)
# R = augmented_lagrangian(A_mats, p, n, m, r)
end = time.time()
Q_guess = np.matmul(R,np.transpose(R))

print('\n')
print('The degree of p(x) is', degree, 'Number of variables:', numVar) 
print('The number of constraints is', m, 'Maximum rank is', r_max,'The rank used is',r)
# print(R.shape)
# print('Q_guess is', Q_guess)
print('The maximum error in coefficients is', LA.norm(_Resid_vec(A_mats, m, p, R)))
print('Time in running augmented Lagrangian method is', end - start)





