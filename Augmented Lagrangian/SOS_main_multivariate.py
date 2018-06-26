import numpy as np
from augmented_lagrangian import augmented_lagrangian, _Resid_vec
import time
from scipy.special import comb
from multivariate_monomials import generate_Amat

# Generate a random positive semidefinite matrix Q
def make_random_Q(n):

    mat = np.random.randn(n, n)
    mat = np.matmul(mat, np.transpose(mat))
    
    return mat

# degree: degree of p(x), must be an even number
degree = 4

# deg: maximum degree of monomials in [x]_d
deg = int(degree/2)

# numVar: number of variables
numVar = 2

# n: length of [x]_d; Also the dimention of X
n = comb(numVar + deg, deg, exact = True)
print(n)

# m: number of coefficients in p(x); number of constraints
m = comb(numVar + degree, degree, exact = True)

# r: the rank of R with the substition X = RR^T
r = 3

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
R = augmented_lagrangian(A_mats, p, n, m, r)
end = time.time()
Q_guess = np.matmul(R,np.transpose(R))
print(R.shape)
print(Q_guess)
print(_Resid_vec(A_mats, m, p, R))
print(end - start)





