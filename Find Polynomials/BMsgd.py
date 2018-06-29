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
			R = R - step*2*error*A_mats[idx].dot(R)
			cost += error**2/2
		order = np.random.permutation(m)
		Cost_history.append(cost)
		R_history.append(R)
		epoch += 1

		if maxsteps:
			if epoch > maxsteps:
				break

	return R

def BM_VGD(A_mats, p, n, r, step = 1e-4, maxsteps = 0, precision = 1e-6, init = None,lb = -1, ub = 1):
	m = len(A_mats)

	if init is None:
	    R = np.random.uniform(lb, ub, size=(n, r))
	# Save the initial point to R_init
	    R_init = R
	else:
	    R = init
	    R_init = init

	R_history = [R]
	cost = 10
	Cost_history = [cost]
	epoch = 0

	descent = 10

	while descent > precision:
		cost = 0
		idx1 = A_mats[0].shape[0]
		idx2 = R.shape[1]
		grad = np.zeros((idx1,idx2))
		for i in range(m):
			# The gradient of J_i(R) is 4*(<A_i, RR^T> - b_i)*A_i*R
			error = np.trace(A_mats[i].T.dot(R.dot(R.T))) - p[i]
			grad = np.add(grad, 2*error*A_mats[i].dot(R))
			
			cost += error**2/2
		R = R - step*grad
		Cost_history.append(cost)
		descent = np.absolute(Cost_history[-1] - Cost_history[-2])
		R_history.append(R)
		epoch += 1

		if maxsteps:
			if epoch > maxsteps:
				break

	return R,cost,R_init