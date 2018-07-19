import time 
import os
import sys
import glob
import calendar
import itertools
from datetime import datetime

import numpy as np
import scipy
from scipy.fftpack.helper import next_fast_len
import obspy
import pyasdf
import pandas as pd
from obspy import read_inventory
from obspy.signal.invsim import cosine_taper
# from msnoise.move2obspy import whiten,myCorr

import noise
from mpi4py import MPI


"""
Run cross-correlation routine from noise module on data.

Saves cross-correlation into an HDF5 file. 

"""

def main(seed1,seed2,CORR,component,maxlag,downsamp_freq,min_dist,max_dist,
         freqmin,freqmax,XML,step=1800,cc_len =3600, method='cross_correlation',norm_type='running_mean'):

    """
    Cross-correlates noise data from ASDFDataSet data set.

    Loads ambient noise data from ASDF data set {data_set}. Cross-correlates 
    data using MSnoise myCorr correlation routine. Saves cross-correlations 
    in ASDF data set {corr_h5}. Uses parameters channel,maxlag,downsamp_freq,
    min_dist,max_dist, max_std, starttime and endtime to filter input data.

    :type data_set: str 
    :parama data_set: path/filename (/Volumes/.../../~.h5) of noise ASDF data
    :type corr_h5: str 
    :param corr_h5: path/filename (/Volumes/.../../~.h5) to save cross-correlations 
                    as ASDF data set. Must end in .h5
    :type component: str
    :param component: Specifies which channels to cross-correlate:
                      'all' for ZZ ZR ZT RZ RR RT TZ TR TT, 
                      'ZZ' for just ZZ,
    :type maxlag: int
    :param maxlag: maximum lag, in seconds, in cross-correlation
    :type downsamp_freq: float
    :param downsamp_freq: Frequency to which waveforms in stream are downsampled
    :return: Downsampled trace or stream object
    :type min_dist: float 
    :param min_dist: minimum distance between stations in km 
    :type max_dist: float 
    :param max_dist: maximum distance between stations in km 
    :type freqmin: float
    :param freqmin: minimun frequency for whitening 
    :type freqmax: float
    :param freqmax: maximum frequency for whitening 
    :type stats: list
    :param stats: list of station names to cross correlate (optional)
                  e.g. stats = ['CHN', 'CRP', 'LGB', 'PSR', 'RIO', 'RUS', 'WLT'] 
    :type step: float
    :param step: time, in seconds, between success cross-correlation windows
    :type step: float
    :param step: length of noise data window, in seconds, to cross-correlate              

    """

    source = obspy.Stream()
    receiver = obspy.Stream()

    for mseed in seed1:
        source += obspy.read(mseed, format="MSEED")
    for mseed in seed2:
        receiver += obspy.read(mseed, format="MSEED")

    source = process_raw(source, downsamp_freq)
    receiver = process_raw(receiver, downsamp_freq)
    if len(source) == 0 or len(receiver) == 0:
        raise ValueError('No traces in Stream')
    source = source.merge(method=1, fill_value=0.)[0]
    receiver = receiver.merge(method=1, fill_value=0.)[0]
    source_stats, receiver_stats = source.stats, receiver.stats

    # trim data to identical times
    t1, t2 = source.stats.starttime, source.stats.endtime
    t3, t4 = receiver.stats.starttime, receiver.stats.endtime
    t1, t3 = nearest_step(t1, t3, step)
    t2, t4 = nearest_step(t2, t4, step)
    if t3 > t1:
        t1 = t3
    if t4 < t2:
        t2 = t4
    if t1 > t2:
        raise ValueError('startime is larger than endtime')

    source = source.trim(t1, t2, pad=True, fill_value=0.)
    receiver = receiver.trim(t1, t2, pad=True, fill_value=0.)

    t_len = np.arange(0, t2 - t1 - cc_len + step, step)
    t_start = [t1 + t for t in t_len]
    t_end = [t1 + t + cc_len for t in t_len]

    # get station inventory
    if XML is not None:
        inv1 = '.'.join([source.stats.network, source.stats.station, 'xml'])
        inv2 = '.'.join([receiver.stats.network, receiver.stats.station, 'xml'])
        inv1 = read_inventory(os.path.join(XML, inv1), format="STATIONXML")
        inv2 = read_inventory(os.path.join(XML, inv2), format="STATIONXML")
        inv1 = inv1.select(channel=source.stats.channel, starttime=t1, endtime=t2)
        inv2 = inv2.select(channel=receiver.stats.channel, starttime=t1, endtime=t2)
        inv1 = noise.pole_zero(inv1)
        inv2 = noise.pole_zero(inv2)

    # window waveforms
    source_slice = obspy.Stream()
    receiver_slice = obspy.Stream()
    for win in source.slide(window_length=cc_len, step=step):
        source_slice += win
    del source
    for win in receiver.slide(window_length=cc_len, step=step):
        receiver_slice += win
    del receiver

    if len(source_slice) == 0 or len(receiver_slice) == 0:
        raise ValueError('No traces in Stream')

    # delete traces with starttimes that do not match
    to_remove = []
    for ii in range(len(source_slice)):
        t1 = source_slice[ii].stats.starttime
        t2 = receiver_slice[ii].stats.starttime
        if t1 != t2:
            to_remove.append(ii)
    if len(to_remove) > 0:
        for ii in to_remove[::-1]:
            source_slice.remove(source_slice[ii])
            receiver_slice.remove(receiver_slice[ii])

    # apply one-bit normalization and whitening 
    source_white, source_params = process_cc(source_slice, freqmin, freqmax, time_norm=True, norm_type=norm_type)    
    receiver_white, receiver_params = process_cc(receiver_slice, freqmin, freqmax, time_norm=True, norm_type=norm_type)          

    # cross-correlate using either cross-correlation, deconvolution, or cross-coherence 
    # if len(data.shape) != 3:
    #   raise ValueError('Whitened data has shape',len(data.shape))
    corr = correlate(source_white, receiver_white, maxlag * downsamp_freq, method=method)
    # try:
    #     corr = noise.remove_resp(corr, source_stats, inv1)
    # except Exception as e:
    #     print(str(e))
    # try:
    #     corr = noise.remove_resp(corr, receiver_stats, inv2)
    # except Exception as e: 
    #     print(str(e))   
    source_slice, receiver_slice = None, None

    # stack cross-correlations
    if not np.any(corr):  # nothing cross-correlated
        raise ValueError('No data cross-correlated')

    t_day = np.array([t.toordinal() for t in t_start])
    t_cc = np.vstack([t_start, t_end]).T

    return corr, t_day, t_cc, source_stats, receiver_stats, source_params, receiver_params


def cross_corr_parameters(source, receiver, start_end_t, source_params,
    receiver_params, locs, maxlag):
    """ 
    Creates parameter dict for cross-correlations and header info to ASDF.

    :type source: `~obspy.core.trace.Stats` object.
    :param source: Stats header from xcorr source station
    :type receiver: `~obspy.core.trace.Stats` object.
    :param receiver: Stats header from xcorr receiver station
    :type start_end_t: `~np.ndarray`
    :param start_end_t: starttime, endtime of cross-correlation (UTCDateTime)
    :type source_params: `~np.ndarray`
    :param source_params: max_mad,max_std,percent non-zero values of source trace
    :type receiver_params: `~np.ndarray`
    :param receiver_params: max_mad,max_std,percent non-zero values of receiver trace
    :type locs: dict
    :param locs: dict with latitude, elevation_in_m, and longitude of all stations
    :type maxlag: int
    :param maxlag: number of lag points in cross-correlation (sample points) 
    :return: Auxiliary data parameter dict
    :rtype: dict

    """

    # source and receiver locations in dict with lat, elevation_in_m, and lon
    source_loc = locs.ix[source['network'] + '.' + source['station']]
    receiver_loc = locs.ix[receiver['network'] + '.' + receiver['station']]

    # # get distance (in km), azimuth and back azimuth
    dist,azi,baz = noise.calc_distance(source_loc,receiver_loc) 

    source_mad,source_std,source_nonzero = source_params[:,0],\
                         source_params[:,1],source_params[:,2]
    receiver_mad,receiver_std,receiver_nonzero = receiver_params[:,0],\
                         receiver_params[:,1],receiver_params[:,2]
    
    starttime = start_end_t[:,0] - obspy.UTCDateTime(1970,1,1)
    starttime = starttime.astype('float')
    endtime = start_end_t[:,1] - obspy.UTCDateTime(1970,1,1)
    endtime = endtime.astype('float')
    source = stats_to_dict(source,'source')
    receiver = stats_to_dict(receiver,'receiver')
    # fill Correlation attribDict 
    parameters = {'source_mad':source_mad,
            'source_std':source_std,
            'source_nonzero':source_nonzero,
            'receiver_mad':receiver_mad,
            'receiver_std':receiver_std,
            'receiver_nonzero':receiver_nonzero,
            'dist':dist,
            'azi':azi,
            'baz':baz,
            'lag':maxlag,
            'starttime':starttime,
            'endtime':endtime}
    parameters.update(source)
    parameters.update(receiver)
    return parameters   

def stats_to_dict(stats,stat_type):
    """

    Converts obspy.core.trace.Stats object to dict

    :type stats: `~obspy.core.trace.Stats` object.
    :type source: str
    :param source: 'source' or 'receiver'
    """
    stat_dict = {'{}_network'.format(stat_type):stats['network'],
                 '{}_station'.format(stat_type):stats['station'],
                 '{}_channel'.format(stat_type):stats['channel'],
                 '{}_delta'.format(stat_type):stats['delta'],
                 '{}_npts'.format(stat_type):stats['npts'],
                 '{}_sampling_rate'.format(stat_type):stats['sampling_rate']}
    return stat_dict            

def process_raw(st,downsamp_freq):
    """
    
    Pre-process month-long stream of data. 
    Checks:
        - sample rate is matching 
        - downsamples data 
        - checks for gaps in data 
        - Trims data to first and last day of month 
        - phase-shifts data to begin at 00:00:00.0
        - chunks data into 86,400 second traces
        - removes instrument response (pole-zero)
    """

    day = 86400   # numbe of seconds in a day
    if len(st) > 100:
        raise ValueError('Too many traces in Stream')
    st = noise.check_sample(st)

    # check for traces with only zeros
    for tr in st:
        if tr.data.max() == 0:
            st.remove(tr)
    if len(st) == 0:
        raise ValueError('No traces in Stream')

    # for tr in st:
    #   tr.data = tr.data.astype(np.float)
    st = noise.downsample(st,downsamp_freq) 
    st = noise.remove_small_traces(st)
    if len(st) == 0:
        raise ValueError('No traces in Stream')

    # check gaps
    if len(noise.getGaps(st)) > 0:
        max_gap = 10
        only_too_long=False
        while noise.getGaps(st) and not only_too_long:
            too_long = 0
            gaps = noise.getGaps(st)
            for gap in gaps:
                if int(gap[-1]) <= max_gap:
                    st[gap[0]] = st[gap[0]].__add__(st[gap[1]], method=0, fill_value="interpolate")
                    st.remove(st[gap[1]])
                    break
                else:
                    too_long += 1
            if too_long == len(gaps):
                only_too_long = True

    st.merge(method=0, fill_value=np.int32(0))

    # phase shift data 
    for tr in st:
        tr = noise.check_and_phase_shift(tr)    
        if tr.data.dtype != 'float64':
            tr.data = tr.data.astype(np.float64)
    return st

def clean_up(corr,sampling_rate,freqmin,freqmax):
    if corr.ndim == 2:
        axis = 1
    else:
        axis = 0
    corr = scipy.signal.detrend(corr,axis=axis,type='constant')
    corr = scipy.signal.detrend(corr,axis=axis,type='linear')
    percent = sampling_rate * 20 / corr.shape[axis]
    taper = scipy.signal.tukey(corr.shape[axis],percent)
    corr *= taper
    corr = bandpass(corr,freqmin,freqmax,sampling_rate,zerophase=True)
    return corr

def process_cc(stream,freqmin,freqmax,percent=0.05,max_len=20.,time_norm=True,norm_type='running_mean',Nfft=None):
    """

    Pre-process for cross-correlation. 

    Checks ambient noise for earthquakesa and data gaps. 
    Performs one-bit normalization and spectral whitening.
    """
    N = len(stream)
    trace_mad = np.zeros(N)
    trace_std = np.zeros(N)
    nonzero = np.zeros(N)
    stream.detrend(type='constant')
    stream.detrend(type='linear')
    stream.taper(max_percentage=percent,max_length=max_len)
    stream.filter('bandpass',freqmin=freqmin,freqmax=freqmax,zerophase=True)
    stream.detrend(type='constant')
    scopy = stream.copy()
    scopy = scopy.merge(method=1)[0]
    all_mad = noise.mad(scopy.data)
    all_std = np.std(scopy.data)
    del scopy 
    npts = []
    for ii,trace in enumerate(stream):
        # check for earthquakes and spurious amplitudes
        trace_mad[ii] = np.max(np.abs(trace.data))/all_mad
        trace_std[ii] = np.max(np.abs(trace.data))/all_std

        # check if data has zeros/gaps
        nonzero[ii] = np.count_nonzero(trace.data)/trace.stats.npts
        npts.append(trace.stats.npts)

    # mask high amplitude phases, then whiten data
    Nt = np.max(npts)
    data = np.zeros([N,Nt])
    for ii,trace in enumerate(stream):
        data[ii,0:npts[ii]] = trace.data
    
    if data.ndim == 1:
        axis = 0
    elif data.ndim == 2:
        axis = 1

    FFTWhite = whiten(data,trace.stats.delta,freqmin,freqmax)

    if time_norm:
        if Nfft is None:
            Nfft = next_fast_len(int(FFTWhite.shape[axis]))
        white = np.real(scipy.fftpack.ifft(FFTWhite, Nfft,axis=axis)) / Nt
        Nt = FFTWhite.shape[axis]
        white = np.concatenate((white[:,-(Nt // 2) + 1:], white[:,:(Nt // 2) + 1]),axis=axis)
        if norm_type == 'one_bit': 
            white = np.sign(white)
        elif norm_type == 'running_mean':
            white = noise.running_abs_mean(white,int(1 / freqmin / 2))
        FFTWhite = scipy.fftpack.fft(white, Nfft,axis=axis)
        FFTWhite[:,-(Nfft // 2) + 1:] = FFTWhite[:,1:(Nfft // 2)].conjugate()[::-1]

    return FFTWhite,np.vstack([trace_mad,trace_std,nonzero]).T

def mseed_data(mseed_dir,starttime = None,endtime = None):
    """
    
    Return sorted list of all available mseed files in dir.

    :type mseed_dir: `str` 
    :param mseed_dir: mseed in chan.loc.start.end.mseed format
                      e.g. BHZ.00.20170113T000000Z.20170114T000000Z.mseed
    :type starttime: `~obspy.core.utcdatetime.UTCDateTime` object.
    :param starttime: Start time of data to cross-correlate
    :type endtime: `~obspy.core.utcdatetime.UTCDateTime` object.
    :param endtime: End time of data to cross-correlate

    """
    mseed = glob.glob(os.path.join(mseed_dir,'*.mseed'))
    file_list = [os.path.basename(m) for m in mseed]
    msplit = np.array([(f.split('.')) for f in file_list])
    chan = msplit[:,0]
    loc = msplit[:,1]
    start = msplit[:,2]
    end = msplit[:,3]
    ind = np.argsort(start)
    start = start[ind]
    end = end[ind]
    mseed = np.array(mseed)[ind]
    start = np.array([obspy.UTCDateTime(t) for t in start])
    end = np.array([obspy.UTCDateTime(t) for t in end])
    if starttime is not None and endtime is not None:
        ind = np.where((start >= starttime) & (end <= endtime))[0]
        mseed,start,end = mseed[ind],start[ind],end[ind]
    elif starttime is not None:
        ind = np.where(start >= starttime)[0]
        mseed,start,end = mseed[ind],start[ind],end[ind]
    elif endtime is not None:
        ind = np.where(end <= endtime)[0]
        mseed,start,end = mseed[ind],start[ind],end[ind]
    return mseed,start,end 

def correlate(fft1,fft2, maxlag, Nfft=None, method='cross_correlation'):
    """This function takes ndimensional *data* array, computes the cross-correlation in the frequency domain
    and returns the cross-correlation function between [-*maxlag*:*maxlag*].

    :type fft1: :class:`numpy.ndarray`
    :param fft1: This array contains the fft of each timeseries to be cross-correlated.
    :type maxlag: int
    :param maxlag: This number defines the number of samples (N=2*maxlag + 1) of the CCF that will be returned.

    :rtype: :class:`numpy.ndarray`
    :returns: The cross-correlation function between [-maxlag:maxlag]
    """
    # Speed up FFT by padding to optimal size for FFTPACK

    if fft1.ndim == 1:
        axis = 0
    elif fft1.ndim == 2:
        axis = 1

    if Nfft is None:
        Nfft = next_fast_len(int(fft1.shape[axis]))

    maxlag = np.round(maxlag)

    Nt = fft1.shape[axis]

    corr = np.conj(fft1) * fft2
    if method == 'deconv':
        corr /= noise.smooth(np.abs(fft1),half_win=5) ** 2
    elif method == 'coherence':
        corr /= noise.smooth(np.abs(fft1),half_win=5)
        corr /= noise.smooth(np.abs(fft2),half_win=5)
    corr = np.real(scipy.fftpack.ifft(corr, Nfft,axis=axis)) / Nt
    corr = np.concatenate((corr[:,-Nt//2 + 1:], corr[:,:Nt//2 + 1]),axis=axis)

    if maxlag != Nt:
        tcorr = np.arange(-Nt//2 + 1, Nt//2)
        dN = np.where(np.abs(tcorr) <= maxlag)[0]
        corr = corr[:,dN]

    return corr


def whiten(data, delta, freqmin, freqmax,Nfft=None):
    """This function takes 1-dimensional *data* timeseries array,
    goes to frequency domain using fft, whitens the amplitude of the spectrum
    in frequency domain between *freqmin* and *freqmax*
    and returns the whitened fft.

    :type data: :class:`numpy.ndarray`
    :param data: Contains the 1D time series to whiten
    :type Nfft: int
    :param Nfft: The number of points to compute the FFT
    :type delta: float
    :param delta: The sampling frequency of the `data`
    :type freqmin: float
    :param freqmin: The lower frequency bound
    :type freqmax: float
    :param freqmax: The upper frequency bound

    :rtype: :class:`numpy.ndarray`
    :returns: The FFT of the input trace, whitened between the frequency bounds
    """

    # Speed up FFT by padding to optimal size for FFTPACK
    if data.ndim == 1:
        axis = 0
    elif data.ndim == 2:
        axis = 1

    if Nfft is None:
        Nfft = next_fast_len(int(data.shape[axis]))

    Napod = 100
    Nfft = int(Nfft)
    freqVec = scipy.fftpack.fftfreq(Nfft, d=delta)[:Nfft // 2]

    J = np.where((freqVec >= freqmin) & (freqVec <= freqmax))[0]
    low = J[0] - Napod
    if low <= 0:
        low = 1

    left = J[0]
    right = J[-1]
    high = J[-1] + Napod
    if high > Nfft / 2:
        high = int(Nfft // 2)

    FFTRawSign = scipy.fftpack.fft(data, Nfft,axis=axis)

    # Left tapering:
    if axis == 1:
        FFTRawSign[:,0:low] *= 0
        FFTRawSign[:,low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:,low:left]))
        # Pass band:
        FFTRawSign[:,left:right] = np.exp(1j * np.angle(FFTRawSign[:,left:right]))
        # Right tapering:
        FFTRawSign[:,right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:,right:high]))
        FFTRawSign[:,high:Nfft + 1] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[:,-(Nfft // 2) + 1:] = FFTRawSign[:,1:(Nfft // 2)].conjugate()[::-1]
    else:
        FFTRawSign[0:low] *= 0
        FFTRawSign[low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[low:left]))
        # Pass band:
        FFTRawSign[left:right] = np.exp(1j * np.angle(FFTRawSign[left:right]))
        # Right tapering:
        FFTRawSign[right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[right:high]))
        FFTRawSign[high:Nfft + 1] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[-(Nfft // 2) + 1:] = FFTRawSign[1:(Nfft // 2)].conjugate()[::-1]

    return FFTRawSign

def nearest_step(t1,t2,step):
    step_min = step / 60
    if t1 == t2:
        return t1,t2

    day1,hour1,minute1,second1 = t1.day,t1.hour,t1.minute,t1.second
    day2,hour2,minute2,second2 = t2.day,t2.hour,t2.minute,t2.second

    start1 = obspy.UTCDateTime(t1.year,t1.month,t1.day)
    start2 = obspy.UTCDateTime(t2.year,t2.month,t2.day)

    t1s = np.array([start1 + s for s in range(0,86400+step,step)])
    t2s = np.array([start2 + s for s in range(0,86400+step,step)])
    t1diff = [t - t1 for t in t1s]
    t2diff = [t - t2 for t in t2s]
    ind1 = np.argmin(np.abs(t1diff))
    ind2 = np.argmin(np.abs(t2diff))
    t1 = t1s[ind1]
    t2 = t2s[ind2]

    return t1,t2

def filter_dist(pairs,locs,min_dist,max_dist):
    """

    Filter station pairs by distance

    """
    new_pairs = []
    for pair in pairs:
        netsta1 = '.'.join(pair[0].split('/')[-3:-1])
        netsta2 = '.'.join(pair[1].split('/')[-3:-1])

        dist,azi,baz = noise.calc_distance(locs.loc[netsta1],locs.loc[netsta2])

        if (dist > min_dist) and (dist < max_dist):
            new_pairs.append(pair)

    return new_pairs


if __name__ == "__main__":
    
    downsamp_freq = 20.  # frequency to downsample to 
    maxlag = 150.      # in seconds 
    min_dist = 0.       # in km     
    max_dist = 50.    # in km     
    freqmin = 0.05      # minimum frequency for whitening 
    freqmax = 4.0       # maximum frequency for whitening 
    cc_len = 1*3600     # seconds for each cross-correlation window 
    step = 1800         # seconds to step between each cross-correlation window 
    starttime = obspy.UTCDateTime(2000,1,1)
    endtime = obspy.UTCDateTime()
    component = 'ZZ' 
    # MSEED = '/n/holylfs/EXTERNAL_REPOS/DENOLLE/CALI/DATA/MSEED/*/*/**Z'  
    # CORR = '/n/holylfs/EXTERNAL_REPOS/DENOLLE/CALI/CORR/' 
    MSEED = '/n/holylfs/EXTERNAL_REPOS/DENOLLE/SAN_GABRIEL/DATA/MSEED/*/*/*HZ'
    CORR = '/n/holylfs/EXTERNAL_REPOS/DENOLLE/SAN_GABRIEL/CORR_REVISE/' 
    data_type = 'CrossCorrelation'
    locations = '/n/holylfs/EXTERNAL_REPOS/DENOLLE/SAN_GABRIEL/DATA/locations.txt'
    XML = '/n/holylfs/EXTERNAL_REPOS/DENOLLE/SAN_GABRIEL/DATA/XML'
    XML = None
    locs = pd.read_csv(locations,index_col='station')
    days_per_rank = 1  # number of days to process on each rank at a time 
    compdirs = glob.glob(MSEED)
    compdirs = sorted(compdirs)
    compdirs = [c for c in compdirs if c.split('/')[-2] not in ['LGB', 'RHC2', 'CPP', 'CRP', 'CHN', 'BFS', 'PASA']]
    # # compdirs = [c for c in compdirs if 'AZ' not in c]
    # compdirs = [c for c in compdirs if '/TO/' in c]
    pairs = list(itertools.combinations(compdirs, 2))
    # pairs = filter_dist(pairs,locs,min_dist,max_dist)

    ####### MPI ####### 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    ###################
    
    for pair in pairs:
        if rank == 0:

            # get all common days between stations 
            compdir1,compdir2 = pair[0],pair[1]
            mseed1,start1,end1 = mseed_data(compdir1,starttime=starttime,endtime=endtime)
            mseed2,start2,end2 = mseed_data(compdir2,starttime=starttime,endtime=endtime)

            # check for matching days
            tord1 = np.array([t.toordinal() for t in start1])
            tord2 = np.array([t.toordinal() for t in start2])
            ind1 = np.where(np.in1d(tord1,tord2))[0]
            ind2 = np.where(np.in1d(tord2,tord1))[0]

            # if len(ind1) == 0 or len(ind2) == 0:
                # continue

            # create pyasdf database 
            path1 = os.path.normpath(compdir1).split(os.sep)
            path2 = os.path.normpath(compdir2).split(os.sep)
            net1,sta1,chan1 = path1[-3],path1[-2],path1[-1]
            net2,sta2,chan2 = path2[-3],path2[-2],path2[-1]
            net_sta = '_'.join((net1,sta1,net2,sta2))

            print('Cross-correlating:',net_sta,obspy.UTCDateTime())
            corr_h5 = os.path.join(CORR,net_sta + '.h5')
            if not os.path.isfile(corr_h5):
                with pyasdf.ASDFDataSet(corr_h5,compression=None,mpi=False) as ds:
                    pass # create pyasdf file 
            else:
                print(net_sta,'Already exists',obspy.UTCDateTime())
                # continue

            mseed1 = mseed1[ind1]
            mseed2 = mseed2[ind2]
            splits = int(np.ceil(len(mseed1) / days_per_rank))
            split1 = np.array_split(mseed1,splits)
            split2 = np.array_split(mseed2,splits)

            # get locations of each station
            locs = pd.DataFrame.from_csv(locations,index_col = 'station')
        else:
            splits,split1,split2,net_sta = [None for _ in range(4)]

        splits = comm.bcast(splits,root=0)
        split1 = comm.bcast(split1,root=0)
        split2 = comm.bcast(split2,root=0)
        net_sta = comm.bcast(net_sta,root=0)
        extra = splits % size

        for ii in range(rank,splits + size - extra,size):

            time.sleep(rank/1000)
            # check for uneven work 
            if ii < splits:
                seed1 = split1[ii]
                seed2 = split2[ii]
                # start,end = os.path.basename(seed1[0]),os.path.basename(seed1[-1])
                # start = start.split('.')[2]
                # end = end.split('.')[2]
                # print('Cross-correlating {}: {}-{} on rank {}'.format(net_sta,start,end,rank))
                try:
                    corr,t_day,t_cc,source_stats,receiver_stats,source_params,receiver_params = main(seed1,\
                        seed2,CORR,component,maxlag,downsamp_freq,min_dist,max_dist, \
                        freqmin,freqmax,XML,step=step,cc_len = cc_len)  
                except Exception as e:
                    print(e)
                    corr,t_day,t_cc,source_stats,receiver_stats,source_params,receiver_params = [None for _ in range(7)]
            else:
                corr,t_day,t_cc,source_stats,receiver_stats,source_params,receiver_params = [None for _ in range(7)]

            corrs = comm.gather(corr,root=0)
            t_day = comm.gather(t_day,root=0)
            t_cc = comm.gather(t_cc,root=0)

            source_stats = comm.gather(source_stats,root=0)
            receiver_stats = comm.gather(receiver_stats,root=0)
            source_params = comm.gather(source_params,root=0)
            receiver_params = comm.gather(receiver_params,root=0)

            if rank == 0:
                with pyasdf.ASDFDataSet(corr_h5,mpi=False) as corr_ds:
                    for jj,_ in enumerate(t_day):
                        day_corrs = corrs[jj]
                        if day_corrs is None:
                            continue
                        day_t = t_cc[jj]
                        day_s_params = source_params[jj]
                        day_r_params = receiver_params[jj]
                        parameters = cross_corr_parameters(source_stats[jj],receiver_stats[jj],
                                     day_t,day_s_params,day_r_params,locs,maxlag)
                        t = day_t[0,0]
                        savedate = '_'.join((str(t.year),
                                             str(t.month),
                                             str(t.day)))
                        savedate = datetime.strptime(savedate,'%Y_%m_%d')
                        savedate = datetime.strftime(savedate,'%Y_%m_%d')
                        corr_sr = '_'.join(['corr',net_sta,savedate])
                        path = os.path.join(net_sta,component,corr_sr)

                        # Add auxiliary data of type CrossCorrelation
                        corr_ds.add_auxiliary_data(data=day_corrs,
                                                       data_type=data_type, 
                                                       path=path,
                                                       parameters=parameters)
                del corrs,t_day,t_cc,source_stats,receiver_stats,source_params,receiver_params

    comm.barrier()

    if rank == 0:
        sys.exit()
    # # MPI.Finalize()
