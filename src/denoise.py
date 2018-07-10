import os
import glob
import sys

import pyasdf
import numpy as np
import scipy as sp
import obspy
import matplotlib.pyplot as plt
from noise import pws
from scipy.signal import wiener
from scipy.linalg import svd
import matplotlib.pyplot as plt
from noise import pws

def NCF_denoising(img_to_denoise,Mdate,Ntau,NSV):

	if img_to_denoise.ndim ==2:
		M,N = img_to_denoise.shape
		if NSV > np.min([M,N]):
			NSV = np.min([M,N])
		[U,S,V] = svd(img_to_denoise,full_matrices=False)
		S = sp.linalg.diagsvd(S,S.shape[0],S.shape[0])
		Xwiener = np.zeros([M,N])
		for kk in range(NSV):
			SV = np.zeros(S.shape)
			SV[kk,kk] = S[kk,kk]
			X = U@SV@V
			Xwiener += wiener(X,[Mdate,Ntau])
			
		denoised_img = wiener(Xwiener,[Mdate,Ntau])
	elif img_to_denoise.ndim ==1:
		M = img_to_denoise.shape[0]
		NSV = np.min([M,NSV])
		denoised_img = wiener(img_to_denoise,Ntau)
		temp = np.trapz(np.abs(np.mean(denoised_img) - img_to_denoise))    
		denoised_img = wiener(img_to_denoise,Ntau,np.mean(temp))

	return denoised_img

def clean_up(corr,sampling_rate,freqmin,freqmax):
	data = []
	st = obspy.Stream()
	if corr.ndim == 2:
		for ii in range(len(corr)):
			tr = obspy.Trace(data=corr[ii,:])
			tr.stats.sampling_rate = sampling_rate
			st += tr
			del tr
	else:
		tr = obspy.Trace(data=corr)
		tr.stats.sampling_rate = sampling_rate
		st += tr
		del tr

	st.detrend('constant')
	st.detrend('simple')
	percent = sampling_rate * 20 / st[0].stats.npts
	st.taper(max_percentage=percent,max_length=20.)
	st.filter('bandpass',freqmin=freqmin,freqmax=freqmax,zerophase=True)
	for tr in st:
		data.append(tr.data)
	data = np.array(data)
	if data.shape[0] == 1:
		data = data.flatten()
	return data
	
if __name__ == "__main__":
	CORR = '/n/holylfs/EXTERNAL_REPOS/DENOLLE/SAN_GABRIEL/CORR/'
	data_type = 'CrossCorrelation'
	component = 'ZZ'
	h5s = glob.glob(CORR + '*.h5')
	h5s = sorted(h5s)
	max_mad = 12 
	sampling_rate = 8.
	min_zero = 0.9
	freqmin = 2.0
	freqmax = 4.0
	Mdate = 12
	NSV = 2
	Ntau = np.ceil(np.min([1/freqmin,1/freqmax]) * sampling_rate) + 15
	# Ntau = 12
	h5s = [h5 for h5 in h5s if 'CI_MWC_CI_PDU' in h5]
	for h5 in h5s:
		with pyasdf.ASDFDataSet(h5) as ds:
			net_sta = os.path.basename(h5).rstrip('.h5')
			if 'corr' in net_sta:
				_,_,sta1,sta2 = net_sta.split('_')
				net_sta = '_'.join(['CI',sta1,'CI',sta2]) 
			corrs = ds.auxiliary_data.CrossCorrelation[net_sta][component].list()
			corr = corrs[100]
			data = ds.auxiliary_data.CrossCorrelation[net_sta][component][corr].data
			data = np.array(data)
			param = ds.auxiliary_data.CrossCorrelation[net_sta][component][corr].parameters
			lag = param['lag']
			t = np.linspace(-lag,lag,data.shape[1])
			sampling_rate = param['receiver_sampling_rate']
			data = clean_up(data,sampling_rate,freqmin,freqmax)
			new = NCF_denoising(data,np.min([Mdate,data.shape[0]]),Ntau,NSV)
			dataMEAN = np.mean(data,axis=0)
			newMEAN = np.mean(new,axis=0)
			# newMEAN = pws(new,sampling_rate=sampling_rate,pws_timegate = 0.5)
			fig, ax = plt.subplots(3, sharex=True)
			ax[0].matshow(data/data.max(),cmap='seismic',extent=[-lag,lag,data.shape[0],1],vmin=-1.0, vmax=1.0,aspect='auto')
			ax[1].matshow(new/new.max(),cmap='seismic',extent=[-lag,lag,data.shape[0],1],vmin=-1.0, vmax=1.0,aspect='auto')
			ax[2].plot(t,dataMEAN/dataMEAN.max(),'r--',label='Filtered MEAN',alpha=0.5)
			ax[2].plot(t,newMEAN/newMEAN.max(),'k',label='Denoised MEAN',alpha=0.75)
			ax[2].legend(loc='best')
			ax[0].set_title('Filterd Cross-Correlations')
			ax[1].set_title('Denoised Cross-Correlations')
			ax[0].xaxis.set_visible(False)
			ax[1].xaxis.set_visible(False)
			ax[2].xaxis.set_ticks_position('bottom')
			ax[2].set_xlabel('Lag (s)')
			plt.suptitle(net_sta)
			plt.show()