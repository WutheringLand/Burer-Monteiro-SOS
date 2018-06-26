from __future__ import division, print_function, absolute_import
import math
import scipy.linalg
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from matplotlib import rc
rc('text', usetex=True)


def augmented_lagrangian(A_mats, p, n, m, k, plotting=False, printing=True, init=None, maxpenalty = 1e150):
    """
    Returns the resulting local minimizer R of the BM problem.
    """

    y = np.ones(m).reshape((-1, 1))
    R = np.random.uniform(-1, 1, size=(n, k))
    # Save the initial point to R_init
    R_init = R
    # Check the feasibility of the problem 
    isInfeasible = False

    penalty = 1.0
    gamma = 10.0
    eta = .25
    target = .00000001
    vec = _Resid_vec(A_mats, m, p, R)
    v = vec.reshape((1, -1)).dot(vec)
    v_best = v
    while v > target:
        Rv = _matrix_to_vector(R)
        if printing == True:
            print('Starting L-BFGS-B on augmented Lagrangian...')
        optimizer = opt.minimize(lambda R_vec: _augmented_lagrangian_func(
            R_vec, A_mats, y, p, m, n, k, penalty), Rv, jac=lambda R_vec: _jacobian(R_vec, A_mats, p, m, n, y, penalty, k), method="L-BFGS-B")
        if printing == True:
            print('Finishing L-BFGS-B on augmented Lagrangian...')
        R = _vector_to_matrix(optimizer.x, k)
        vec = _Resid_vec(A_mats, m, p, R)
        v = vec.reshape((1, -1)).dot(vec)
        if printing == True:
            print('Finish updating variables...')
        if v < eta * v_best:
            y = y - penalty * vec
            v_best = v
        else:
            penalty = gamma * penalty
        if printing == True:
            print(penalty)
        # Check penalty for infeasibility
        if penalty >= maxpenalty:
            isInfeasible = True
            print('Infeasibility detected.')
            break
    if printing == True:
        print('Augmented Lagrangian terminated.')
    return R, isInfeasible, R_init


def _generate_random_rect(n, k):
    """
    Returns a random initialization of matrix.
    """

    R = np.random.uniform(-1, 1, (n, k))
    for i in range(n):
        R[i, :] = R[i, :] / np.linalg.norm(R[i, :])
    return R


def _basis_vector(size, index):
    """
    Returns a basis vector with 1 on certain index.
    """

    vec = np.zeros(size)
    vec[index] = 1
    return vec

def _Resid_vec(A_mats, m, p, R):
    """
    Returns vector with constraint residuals
    """
    vec = np.empty(m)
    Q = np.matmul(R, np.transpose(R))
    for i in range(m):
        vec[i] = np.trace(np.matmul(np.transpose(A_mats[i]),Q)) - p[i]
    return vec.reshape((-1,1))


def _augmented_lagrangian_func(Rv, A_mats, y, p, m, n, k, sigma):
    """
    Returns the value of objective function of augmented Lagrangian.
    """

    R = _vector_to_matrix(Rv, k)
    Resid = _Resid_vec(A_mats, m, p, R)
    first_term = -np.dot(np.transpose(y), Resid)
    second_term = sigma/2.0 * np.dot(np.transpose(Resid), Resid)
    val = first_term + second_term

    return val


def _vector_to_matrix(Rv, k):
    """
    Returns a matrix from reforming a vector.
    """
    U = Rv.reshape((-1, k))
    return U


def _matrix_to_vector(R):
    """
    Returns a vector from flattening a matrix.
    """

    u = R.reshape((1, -1)).ravel()
    return u

# def _take_one_row(Z, R, l):
#     Z[l,:] = R[l,:]
#     return Z

def _vec_dot_mats(mats, vec, n, m):
    
    sum = np.zeros((n,n))
    for i in range(m):
        sum = np.add(sum, vec[i]*mats[i])
    return sum



def _jacobian(Rv, A_mats, p, m, n, y, sigma, k):
    """
    Returns the Jacobian matrix of the augmented Lagrangian problem.
    """

    R = _vector_to_matrix(Rv, k)

    first_term = np.matmul(2.0*_vec_dot_mats(y, A_mats, n, m),R)

    second_term = np.matmul(2.0*sigma*_vec_dot_mats(_Resid_vec(A_mats, m, p, R), A_mats, n, m), R)

    jacobian = np.add(first_term, second_term)

    jac_vec = _matrix_to_vector(jacobian)
    
    return jac_vec.reshape((1, -1)).ravel()

