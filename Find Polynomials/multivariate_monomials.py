import numpy as np
from scipy.special import comb

# _count_down returns the list of monomials of a given degree
# degree of monomials; pos: index of variable, initial value = 1; numVar: number of variables; degree: 
def _count_down(degree,pos,numVar):
	if pos == numVar:
		return [[degree]]
	outerList = []
	for first in range(degree, -1, -1):
		for element in _count_down(degree-first, pos+1, numVar):
			innerList = [first] + element
			outerList.append(innerList)
	return outerList

# make the list of monomials with index order
def monomialsList(degree,numVar):
	outerList = []
	for total in range(degree + 1):
		outerList = outerList + _count_down(total,1,numVar)
	return outerList

# generate A_i for to formulat SDPs 
def generate_Amat(deg,numVar):
	monomials_deg = monomialsList(deg,numVar)
	monomials_2deg = monomialsList(2*deg,numVar)

	monomials_deg_arraylist = [np.array(l) for l in monomials_deg]
	monomials_2deg_arraylist = [np.array(l) for l in monomials_2deg]

	# m : number of constraints
	m = len(monomials_2deg)
	# n: number of monomials of degree/2
	n = len(monomials_deg)

	A_mat = []
	for gamma in range(m):
		mat = np.zeros((n,n))
		for alpha in range(n):
			for beta in range(n):
				if all(monomials_deg_arraylist[alpha] + monomials_deg_arraylist[beta] == monomials_2deg_arraylist[gamma]):
					mat[alpha,beta] = 1
		A_mat.append(mat)
	return A_mat
