# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:33:08 2018

@author: jhseidman
"""

import numpy as np
from augmented_lagrangian import augmented_lagrangian, _Resid_vec
import time

def anti_off_diag(n, l):
    
    if l <= n:
        mat = np.rot90(np.diagflat(np.ones(l), n - l))
    else:
        mat = np.rot90(np.diagflat(np.ones(n - l + n), n - l))
    return mat

def make_random_Q(degree):

    mat = np.random.randn(degree + 1, degree + 1)
    mat = np.matmul(mat, np.transpose(mat))
    
    return mat

degree = 100
n = int(degree/2) + 1
print(n)
m = degree + 1
k = 3

# Make constraint matrices
A_mats = []
for i in range(m):
    mat = anti_off_diag(n, i+1)
    A_mats.append(mat)
    
#coefficients
#p = np.array([10, 8, 0, -2, 1])
#p = np.array([11, 34, 51, -2, 16, -2, 1])

Q_deg = int(degree/2)

Q = 100*make_random_Q(Q_deg)
#Q = np.random.randn(Q_deg + 1, Q_deg + 1)

p = np.zeros(degree + 1)
for i in range(degree + 1):
    val = np.trace(np.matmul(np.transpose(A_mats[i]),Q))
    p[i] = val
    
p = p.reshape((-1,1))



#print(A_mats)
start = time.time()
R = augmented_lagrangian(A_mats, p, n, m, k)
end = time.time()
Q_guess = np.matmul(R,np.transpose(R))
print(R.shape)
print(Q_guess)
print(_Resid_vec(A_mats, m, p, R))
print(end - start)
