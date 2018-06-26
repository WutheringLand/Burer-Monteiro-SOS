import numpy as np
def BM_SGD(A_mats, R_init, p, step=0.001, maxsteps=0, precision=0.001,):
	m = len(A_mats)
	R = R_init
	R_history = [R]
	cost = 10
	Cost_history = [cost]
	order = np.random.permutation(m)
	epoch = 0

	while cost > precision:
		cost = 0
		for i in range(m):
			idx = order[i]
			# The gradient of J_i(R) is 4*(<A_i, RR^T> - b_i)*A_i*R
			error = np.trace(A_mats[idx].T.dot(R.dot(R.T))) - p[idx]
			R = R - step*4*error*A_mats[idx].dot(R)
			cost += error**2/2
		order = np.random.permutation(m)
		Cost_history.append(cost)
		R_history.append(R)
		epoch += 1

		if maxsteps:
			if epoch > maxsteps:
				break

	return R


