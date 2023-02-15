#=========================================================================
# Multiple-splitting projection test for high dimensional mean vectors
# Authors: Wanjun Liu, Xiufan Yu, and Runze Li
# Year: 2022
#=========================================================================

import numpy as np
from function import *
import sys

jobid = int(sys.argv[1])
nrep = 100
vec_seed = np.random.choice(range(9999, 9999999), size = 100000, replace = False)

cov_type = 'ar'
alt_type = 'sparse'
h_type = 0
n = 40
p = 1000

if h_type == 0:
	vecc = np.array([0.0])
if h_type == 1 and cov_type == 'ar':
	vecc = np.array([0.5])
if h_type == 1 and cov_type == 'cs':
	vecc = np.array([0.5])

intc = int(h_type)
lenc = len(vecc)
vrho = np.linspace(0.1, 0.9, 9)
lenr = len(vrho)
nsplits = 100
#lensplt = len(nsplits)
nmethod = 3

if alt_type == 'sparse':
	s = 10
	mu0 = np.append(np.ones(s), np.zeros(p-s))
if alt_type == 'dense':
	mu0 = np.ones(p)*np.sqrt(1./n)

if cov_type == 'ar':
	lambds = 2**np.linspace(-2,-6,50)
if cov_type == 'cs':
	lambds = 2**np.linspace(-1,-5,50)

mpval_r = np.zeros([lenr*nrep, nsplits])
mpval_s = np.zeros([lenr*nrep, nsplits])

for l in range(nrep):
	for i in range(lenc):
		c = vecc[i]
		for j in range(lenr):

			r = vrho[j]
			seed_id = (jobid-1)*nrep*lenc*lenr + l*lenc*lenr + i*lenr + j
			np.random.seed(vec_seed[seed_id])

			if cov_type == 'cs':
				sigma = cov_cs(p, r)
				X = get_multivariate_normal(c*mu0, sigma, n)
			if cov_type == 'ar':
				X = get_ar1X(n, p, r, c*mu0)

			pr = proj_test_ridge(X, nsplits)
			mpval_r[j*nrep+l,:] = pr

			n1 = int(n/2)
			n2 = n - n1
			ps = []
			for k in range(nsplits):
				
				idx = np.random.choice(range(n), n1, replace=False)
				X1 = X[idx]
				X2 = np.delete(X, idx, axis=0)
				out = cqp_wl_admm_bic(X1, lambds, 1./np.log(n1), rho = 10., gamma = np.sqrt(np.log(p)/(n1)))
				ws = out[0]
				y = np.dot(X2, ws)
				t = get_t_stat(y)
				pval = 2*(1. - sp_t.cdf(np.abs(t), n2-1))
				ps.append(pval)
			mpval_s[l*lenr+j,:] = np.array(ps)

np.savetxt("spval_"+str(cov_type)+"_"+str(alt_type)+"_n"+str(n)+"_p"+str(p)+"_c"+str(intc)+"_"+str(jobid), mpval_s, fmt = "%.6f")
np.savetxt("rpval_"+str(cov_type)+"_"+str(alt_type)+"_n"+str(n)+"_p"+str(p)+"_c"+str(intc)+"_"+str(jobid), mpval_r, fmt = "%.6f")




