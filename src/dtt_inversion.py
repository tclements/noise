from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
import os  
import glob
import sys

import datetime
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from scipy import sparse

# solve inversion for daily velocity change as solved by Brenguier et al., 2014
# compute MWCS for each day i against every other day j
# dV_ij = (V_j - V_i) / V_i = mwcs(ccf_i,ccf_j)
# data vector d = [dV_12,dV_13,d_V14,...,dV_n-1n]
# goal is to solve for m = [dV_1,dV_2,dV_3,...,dV_n]
# solve in naive way d = Gm, where
#
# G = [-1 1 0 0 0 ...  0 0
#      -1 0 1 0 0 ...  0 0
#      -1 0 0 1 0 ...  0 0
#       0 0 0 0 0 ... -1 1]
#
# use a bayesian least squares inversion to solve for m:
# m = (G^T Cd^-1 G + a * Cm^-1)^-1 G^T Cd^-1 d, where
# Cd is a covairance matrix that describes the Gaussian uncertainties of each dV_ij measurement
# Cm is a priori covariance matrix for model vector m
# each value of Cm is exp(-|i-j|/2*b), where i and j are days of each ccf and
# b is characteristic correlation length
# 
# Last modified by thclements@g.harvard.edu 8/1/17

def g_mat(days):
    """
    Create sparse G matrix for each day pair.
    
    Inputs
    ------
    :type days: pandas.core.frame.DataFrame 
    :param days: Datafram with day1 and day2 for each MWCS measurement
    :returns: G matrix in form
                        G = [-1 1 0 0 0 ...  0 0 
                             -1 0 1 0 0 ...  0 0 
                             -1 0 0 1 0 ...  0 0 
                              0 0 0 0 0 ... -1 1]
    """
    uniq_day = pd.Series(days[['day1','day2']].values.ravel()).unique()
    day1 = days.groupby('day1').indices
    day2 = days.groupby('day2').indices

    M = uniq_day.shape[0]
    N = len(days)
    rows = np.arange(N)
    cols = np.arange(M)
    first = np.empty(N)
    second = np.empty(N)

    for ii,day in enumerate(uniq_day):
        if ii < M - 1:
            first[day1[day]] = ii
        if ii > 0:
            second[day2[day]] = ii

    # create sparse G matrix 
    G = sparse.lil_matrix((N,M))
    G[rows,first] = -1
    G[rows,second] = 1
    return G

def model_cov(days,beta):
    """
    Create model covariance matrix for each day pair.
    
    Inputs
    ------
    :type days: np.array of datetime.datetime
    :param days: Nx2 matrix with days for each dV_ij measurement
    :type beta: float
    :param beta: Characteristic correlation length in days 
    :returns: model covariance matrix
    """
    uniq_day = pd.Series(days[['day1','day2']].values.ravel()).unique()
    day1 = days.groupby('day1').indices
    day2 = days.groupby('day2').indices
    N = uniq_day.shape[0]
    M = len(days) 
    Cm = np.zeros((N,N))

    d = {}
    for ii,day in enumerate(uniq_day):
        d[day] = ii
    ii = np.vectorize(d.get)(days['day1'].values)
    jj = np.vectorize(d.get)(days['day2'].values)
    Cm[ii,jj] = days['diff'].values
    tri_ind = np.tril_indices(N, -1)
    Cm[tri_ind] = Cm.T[tri_ind]

    Cm /= 2*beta
    Cm = np.exp(Cm)
    Cm[Cm == 1.] = 0
    return Cm
    
def data_cov(error):
    """
    Create data covariance matrix for each day pair.
    
    Inputs
    ------
    :type error: np.array of float
    :param error: Error associated with each dv/v measurement 
    :returns: data covariance matrix
    """
    return np.diag(error)


if __name__ == "__main__":

	#### INVERSION ####
	beta = 1000 
	max_days = 1000
	###################
	begin = datetime.datetime(2001,1,1)
	end = datetime.datetime(2017,9,1)
	filter_by_date = True 

	# pair = sys.argv[1]
	# read data from CSV
	# DTT = '/n/flashlfs/mdenolle/TCLEMENTS/CALI/DTT/TEST/0.1_0.3/1_DAYS/ZZ/'
	# pairs = glob.glob(os.path.join(DTT,'*.txt*'))
	# for pair in pairs[6:7]:
	CSV = '/n/holylfs/EXTERNAL_REPOS/DENOLLE/SAN_GABRIEL/DTT/0.5_2.0/30_DAYS/ZZ/CI_RIO_CI_RUS.txt'
	base = os.path.basename(CSV)

	INVERT = CSV.replace('DTT','INVERT')
	INVERT = INVERT.rstrip(base)
	if not os.path.exists(INVERT):
		os.makedirs(INVERT)
	df = pd.read_csv(CSV)
	net_sta = os.path.basename(CSV).rstrip('.txt')
	print('LOADED {}'.format(net_sta))

	# clean the dataframe 
	df['dt1'] = pd.to_datetime(df['day1'],format='%Y-%m-%d')
	df['dt2'] = pd.to_datetime(df['day2'],format='%Y-%m-%d')
	df['diff'] = df['dt2'] - df['dt1']
	df['diff'] = df['diff'] /  np.timedelta64(1, 'D')
	df = df[df['diff'] < max_days]
	df = df.dropna()
	if filter_by_date:
		df = df[(df['dt1'] > begin) & (df['dt2'] < end)]
	M = df.M.values
	EM = df.EM.values
	days = df[['day1','day2','dt1','dt2','diff']].reset_index(drop=True)

	# ind = np.where(days[:,0] > datetime.datetime(2011,1,1))[0]
	# ind1 = np.where(days[:,1] < datetime.datetime(2017,9,1))[0]
	# ind = np.intersect1d(ind,ind1)
	# days = days[ind,:]
	# EM = EM[ind]
	# M = M[ind]

	# create data matrices
	G = g_mat(days)
	Cm = model_cov(days,beta)
	Cd = EM
	Cd_inv = 1/EM
	Cd_inv[Cd_inv == np.inf] = 0 
	Cm_inv = np.linalg.inv(Cm)
	M_Cd = M * Cd_inv
	M_Cd = M_Cd.reshape(M_Cd.size,1)

	# invert for dtt
	GtCd = G.T.multiply(sparse.csr_matrix(Cd_inv))
	GtG = GtCd.dot(G).toarray()
	GtG_norm = np.linalg.norm(GtG)
	dtt_alpha = []
	norm_diffs = []
	# G = G.toarray()
	alpha = np.logspace(5,7,10)
	for ii,a in enumerate(alpha):
		alpha_Cm = a * Cm_inv
		norm_diff = GtG_norm - np.linalg.norm(alpha_Cm)
		norm_diffs.append(norm_diff)
		GtGCm = GtG + alpha_Cm
		GtGCm1 = np.linalg.inv(GtGCm)
		GtGCm1 = sparse.csr_matrix(GtGCm1)
		dtt = GtGCm1.dot(G.T.dot(M_Cd))
		dtt_alpha.append(dtt)
		print('{} alpha: {}, diff: {}, {}'.format(net_sta,a,norm_diff,ii))
	dtt_days = pd.Series(days[['dt1','dt2']].values.ravel()).unique()

	# # norms for L-curve 
	# m_norm,gmd_norm = [],[]
	# for m in dtt_alpha:
	# 	m = np.array(m)
	# 	m_norm.append(np.linalg.norm(m))
	# 	gmd = np.dot(G,m) - M
	# 	gmd_norm.append(np.linalg.norm(gmd))

	# # find maximum curvature 
	# grad = np.gradient(m_norm,gmd_norm)
	# max_grad = np.argmax(grad)

	# # df = pd.DataFrame({'days': dtt_days,'alpha':np.logspace(0,10,100) , 'dtt': dtt_alpha,
	# #                    'norm_diff': norm_diffs, 'm_norm': m_norm, 'gmd_norm': gmd_norm})
	# # df.to_csv(os.path.join(invert,base),index=False)


	# # plot 
	# fig,ax = plt.subplots(2)
	# ax[0].loglog(gmd_norm,m_norm,'ko')
	# ax[0].set_xlabel('Residual norm')
	# ax[0].set_ylabel('Solution norm')
	# ax[0].set_title('L-curve for {}'.format(net_sta))
	# ax[0].loglog(gmd_norm[max_grad],m_norm[max_grad],'yo')
	# ax[1].plot(grad)
	# plt.show()

	# plot 
	# ii = np.argmin(np.abs(norm_diffs))
	for ii,dtt in enumerate(dtt_alpha):
		fig,ax = plt.subplots(figsize=(16,5))
		ax.plot_date(dtt_days,dtt*100,'bo',alpha = 0.5)
		plt.ylabel('% dv/v')
		plt.title('dv/v for {}, alpha = {}'.format(net_sta,alpha[ii]))
		fig.autofmt_xdate()
		plt.show()
