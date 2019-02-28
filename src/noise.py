from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import range
from builtins import str
from builtins import round
from builtins import int
from future import standard_library
standard_library.install_aliases()
import __future__
import copy
import glob
import os

import numpy as np 
from scipy.fftpack import fft,ifft,next_fast_len
import scipy.signal 
from scipy.signal import hilbert
from scipy.ndimage import map_coordinates

import obspy 
from obspy.signal.filter import bandpass
from obspy.core import AttribDict
import pyasdf


def snr(data,sampling_rate):
    """
    Signal to noise ratio of N cross-correlations.

    Follows method of Clarke et. al, 2011. Measures SNR at each point.

    """	
    data = np.array(data)
    N,t = data.shape
    data_mean = np.mean(data,axis=0)

    # calculate noise and envelope functions
    sigma = np.mean(data**2,axis=0) - (data_mean)**2
    sigma = np.sqrt(sigma/(N-1.))
    s = np.abs(data_mean + 1j*scipy.signal.hilbert(data_mean))

    # smooth with 10 second sliding cosine window 
    # half window length is 5s, i.e. 5 * sampling rate
    sigma = smooth(sigma,half_win=int(sampling_rate*5))
    s = smooth(s,half_win=int(sampling_rate*5))


    return np.real(s/sigma)


def smooth(x, window='boxcar', half_win=3):
    """ some window smoothing from MSnoise MWCS """
    window_len = 2*half_win+1
    # extending the data at beginning and at the end
    # to apply the window at the borders
    if window == "boxcar":
        w = scipy.signal.boxcar(window_len).astype(x.dtype)
    else:
        w = scipy.signal.hanning(window_len).astype(x.dtype)

    if x.ndim ==1:
        s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
        y = np.convolve(w/w.sum(), s, mode='valid')
        y = y[half_win:len(y)-half_win]
    elif x.ndim == 2:
        y = np.zeros(x.shape,x.dtype)
        for ii,row in enumerate(x):
            s = np.r_[row[window_len-1:0:-1], row, row[-1:-window_len:-1]]
            tmp = np.convolve(w/w.sum(), s, mode='valid')
            y[ii,:] = tmp[half_win:len(tmp)-half_win]
    return y


def nextpow2(x):
    """
    Returns the next power of 2 of x.

    :type x: int 
    :returns: the next power of 2 of x

    """

    return np.ceil(np.log2(np.abs(x))) 	


def runningMean(x, N):
    """
    Returns array x smoothed by running mean of N points.

    :type x:`~numpy.ndarray` 
    :type N: int
    :param N: Number of points to smooth over 
    :returns: Array x, smoothed by running mean of N points
    
    """
    return np.convolve(x, np.ones((N,))/N)[(N-1):]	


def running_abs_mean(x, N):
    """
    Returns array x smoothed by absolute running mean of N points.

    From Bensen et al., 2007

    :type x:`~numpy.ndarray` 
    :type N: int
    :param N: Number of points to smooth over 
    :returns: Array x, smoothed by running absolute mean of N points
    
    """
    ndim = x.ndim 
    if ndim == 1:
        weights = np.convolve(np.abs(x), np.ones((N, )) / N)[(N - 1):]
        x = x / weights 
    elif ndim == 2:
        for ii in range(x.shape[0]):
            weights = np.convolve(np.abs(x[ii, :]), np.ones((N, )) / N)[(N - 1):]
            x[ii, :] = x[ii, :] / weights
    return x

def abs_max(arr):
    """
    Returns array divided by its absolute maximum value.

    :type arr:`~numpy.ndarray` 
    :returns: Array divided by its absolute maximum value
    
    """
    
    return (arr.T / np.abs(arr.max(axis=-1))).T


def pws(arr,power=2.,sampling_rate=20.,pws_timegate = 5.):
    """
    Performs phase-weighted stack on array of time series. 

    Follows methods of Schimmel and Paulssen, 1997. 
    If s(t) is time series data (seismogram, or cross-correlation),
    S(t) = s(t) + i*H(s(t)), where H(s(t)) is Hilbert transform of s(t)
    S(t) = s(t) + i*H(s(t)) = A(t)*exp(i*phi(t)), where
    A(t) is envelope of s(t) and phi(t) is phase of s(t)
    Phase-weighted stack, g(t), is then:
    g(t) = 1/N sum j = 1:N s_j(t) * | 1/N sum k = 1:N exp[i * phi_k(t)]|^v
    where N is number of traces used, v is sharpness of phase-weighted stack

    :type arr: numpy.ndarray
    :param arr: N length array of time series data 
    :type power: float
    :param power: exponent for phase stack
    :type sampling_rate: float 
    :param sampling_rate: sampling rate of time series 
    :type pws_timegate: float 
    :param pws_timegate: number of seconds to smooth phase stack
    :Returns: Phase weighted stack of time series data
    :rtype: numpy.ndarray  
    """

    if arr.ndim == 1:
        return arr
    N,M = arr.shape
    analytic = arr + 1j * hilbert(arr,axis=1, N=next_fast_len(M))[:,:M]
    phase = np.angle(analytic)
    phase_stack = np.mean(np.exp(1j*phase),axis=0)/N
    phase_stack = np.abs(phase_stack)**2

    # smoothing 
    timegate_samples = int(pws_timegate * sampling_rate)
    phase_stack = runningMean(phase_stack,timegate_samples)
    weighted = np.multiply(arr,phase_stack)
    return np.mean(weighted,axis=0)/N

def dtw(x,r, g=1.05):
    """ Dynamic Time Warping Algorithm

    Inputs:
    x:     target vector
    r:     vector to be warped

    Outputs:
    D: Distance matrix
    Dist:  unnormalized distance between t and r
    w:     warping path
    
    originally written in MATLAB by Peter Huybers 
    """

    x = norm(x)
    r = norm(r)

    N = len(x)
    M = len(r)

    d = (np.tile(x,(M,1)).T - np.tile(r,(N,1)))**2
    d[0,:] *= 0.25
    d[:,-1] *= 0.25

    D=np.zeros(d.shape)
    D[0,0] = d[0,0]

    for ii in range(1,N):
        D[ii,0] = d[ii,0] + D[ii - 1,0]     

    for ii in range(1,M):
        D[0,ii] = d[0,ii] + D[0,ii-1]

    for ii in range(1,N):
        for jj in range(1,M):
            D[ii,jj] = d[ii,jj] + np.min([g * D[ii - 1, jj], D[ii - 1, jj - 1], g * D[ii, jj - 1]]) 

    dist = D[-1,-1]
    ii,jj,kk = N - 1, M - 1, 1
    w = []
    w.append([ii, jj])
    while (ii + jj) != 0:
        if ii == 0:
            jj -= 1
        elif jj == 0:
            ii -= 1 
        else:
            ind = np.argmin([D[ii - 1, jj], D[ii, jj - 1], D[ii - 1, jj - 1]])
            if ind == 0:
                ii -= 1
            elif ind == 1:
                jj -= 1
            else:
                ii -= 1
                jj -= 1
        kk += 1
        w.append([ii, jj])

    w = np.array(w)
    
    return D,dist,w 


def norm(arr):
    """ Demean and normalize a given input to unit std. """
    if arr.ndim == 1:
        arr -= arr.mean()
        arr /= arr.std()
    else:
        arr -= arr.mean(axis=-1, keepdims=True)
        arr = (arr.T / arr.std(axis=-1)).T
    return arr


def clean_up(corr, sampling_rate, freqmin, freqmax):
    if corr.ndim == 2:
        axis = 1
    else:
        axis = 0
    corr = scipy.signal.detrend(corr, axis=axis, type='constant')
    corr = scipy.signal.detrend(corr, axis=axis, type='linear')
    percent = np.min([sampling_rate * 20 / corr.shape[axis],0.05])
    taper = scipy.signal.tukey(corr.shape[axis], percent)
    corr *= taper
    corr = bandpass(corr, freqmin, freqmax, sampling_rate, zerophase=True)
    return corr

def stretch_mat_creation(refcc, str_range=0.01, nstr=1001):
    """ Matrix of stretched instance of a reference trace.

    From the MIIC Development Team (eraldo.pomponi@uni-leipzig.de)
    The reference trace is stretched using a cubic spline interpolation
    algorithm form ``-str_range`` to ``str_range`` (in %) for totally
    ``nstr`` steps.
    The output of this function is a matrix containing the stretched version
    of the reference trace (one each row) (``strrefmat``) and the corresponding
    stretching amount (`strvec```).
    :type refcc: :class:`~numpy.ndarray`
    :param refcc: 1d ndarray. The reference trace that will be stretched
    :type str_range: float
    :param str_range: Amount, in percent, of the desired stretching (one side)
    :type nstr: int
    :param nstr: Number of stretching steps (one side)
    :rtype: :class:`~numpy.ndarray` and float
    :return: **strrefmat**: 2d ndarray of stretched version of the reference trace.
    :rtype: float
    :return: **strvec**: List of float, stretch amount for each row of ``strrefmat``
    """

    n = len(refcc)
    samples_idx = np.arange(-n // 2 + 1, n // 2 + 1)
    strvec = np.linspace(1 - str_range, 1 + str_range, nstr)
    str_timemat = samples_idx / strvec[:,None] + n // 2
    tiled_ref = np.tile(refcc,(nstr,1))
    coord = np.vstack([(np.ones(tiled_ref.shape) * np.arange(tiled_ref.shape[0])[:,None]).flatten(),str_timemat.flatten()])
    strrefmat = map_coordinates(tiled_ref, coord)
    strrefmat = np.flipud(strrefmat.reshape(tiled_ref.shape))
    return strrefmat, strvec


def stretch(data,ref,str_range=0.05,nstr=1001):
    """
    Stretching technique for dt/t. 

    :type data: :class:`~numpy.ndarray`
    :param data: 2d ndarray. Cross-correlation measurements.
    :type ref: :class:`~numpy.ndarray`
    :param ref: 1d ndarray. The reference trace that will be stretched
    :type str_range: float
    :param str_range: Amount, in percent, of the desired stretching (one side)
    :type nstr: int
    :param nstr: Number of stretching steps (one side)
    :rtype: :class:`~numpy.ndarray` 
    :return: **alldeltas**: dt/t for each cross-correlation
    :rtype: :class:`~numpy.ndarray`
    :return: **allcoefs**: Maximum correlation coefficient for each 
         cross-correlation against the reference trace
    :rtype: :class:`~numpy.ndarray`
    :return: **allerrs**: Error for each dt/t measurement

    """
    ref_stretched, deltas = stretch_mat_creation(ref,str_range=str_range,nstr=nstr)
    M,N = data.shape

    alldeltas = np.empty(M,dtype=float)
    allcoefs = np.empty(M,dtype=float)
    allerrs = np.empty(M,dtype=float)
    x = np.arange(nstr)

    for ii in np.arange(M).tolist():
        coeffs = vcorrcoef(ref_stretched,data[ii,:])
        coeffs_shift = coeffs + np.abs(np.min([0,np.min(coeffs)]))
        fw = FWHM(x,coeffs_shift)
        alldeltas[ii] = deltas[np.argmax(coeffs)]
        allcoefs[ii] = np.max(coeffs)
        allerrs[ii] = fw/2
    
    return alldeltas, allcoefs, allerrs

def FWHM(x,y):
    """

    Fast, naive calculation of full-width at half maximum. 

    """
    half_max = np.max(y) / 2.
    left_idx = np.where(y - half_max > 0)[0][0]
    right_idx = np.where(y - half_max > 0)[0][-1]
    return x[right_idx] - x[left_idx]


def pole_zero(inv): 
    """

    Return only pole and zeros of instrument response

    """
    for ii,chan in enumerate(inv[0][0]):
        stages = chan.response.response_stages
        new_stages = []
        for stage in stages:
            if type(stage) == obspy.core.inventory.response.PolesZerosResponseStage:
                new_stages.append(stage)
            elif type(stage) == obspy.core.inventory.response.CoefficientsTypeResponseStage:
                new_stages.append(stage)


        inv[0][0][ii].response.response_stages = new_stages

    return inv


def remove_small_traces(stream,min_length = 100.):
    """
    Removes small traces from stream
    min_length = 20 s

    """	
    if len(stream.get_gaps()) == 0:
        return stream

    for tr in stream:
        if tr.stats.npts < 4 * min_length*tr.stats.sampling_rate:
            stream.remove(tr)
    return stream	


def check_and_phase_shift(trace):
    # print trace
    taper_length = 20.0
    # if trace.stats.npts < 4 * taper_length*trace.stats.sampling_rate:
    # 	trace.data = np.zeros(trace.stats.npts)
    # 	return trace

    dt = np.mod(trace.stats.starttime.datetime.microsecond*1.0e-6,
                trace.stats.delta)
    if (trace.stats.delta - dt) <= np.finfo(float).eps:
        dt = 0
    if dt != 0:
        if dt <= (trace.stats.delta / 2.):
            dt = -dt
        # direction = "left"
        else:
            dt = (trace.stats.delta - dt)
        # direction = "right"
        trace.detrend(type="demean")
        trace.detrend(type="simple")
        taper_1s = taper_length * float(trace.stats.sampling_rate) / trace.stats.npts
        trace.taper(taper_1s)

        n = int(2**nextpow2(len(trace.data)))
        FFTdata = scipy.fftpack.fft(trace.data, n=n)
        fftfreq = scipy.fftpack.fftfreq(n, d=trace.stats.delta)
        FFTdata = FFTdata * np.exp(1j * 2. * np.pi * fftfreq * dt)
        trace.data = np.real(scipy.fftpack.ifft(FFTdata, n=n)[:len(trace.data)])
        trace.stats.starttime += dt
        return trace
    else:
        return trace


def match_trace(trace,stream):
    """
    Matches trace in stream that begin at the same time UTC. 

    Removes matched trace from stream for faster matching.
    
    :type trace:`~obspy.core.trace.Trace` object. 
    :param trace: Day-long trace 
    :type stream:`~obspy.core.stream.Stream` object. 
    :param stream: Stream containing one or more day-long trace 
    :Returns: trace from stream object that has same starting time 
    :rtype:`~obspy.core.trace.Trace` object. 
     """
    
    # max time difference between starting sample 0 minutes
    max_time = trace.stats.delta 

    matched_trace = False

    for ii,tr in enumerate(stream):
        if np.abs(tr.stats.starttime - trace.stats.starttime) <= max_time and \
        len(tr.data) == len(trace.data):
            matched_trace = tr
            stream.pop(ii)
            break

    return matched_trace,stream		
    

def check_sample(stream):
    """
    Returns sampling rate of traces in stream.

    :type stream:`~obspy.core.stream.Stream` object. 
    :param stream: Stream containing one or more day-long trace 
    :return: List of sampling rates in stream

    """
    if type(stream) == obspy.core.trace.Trace:
        return stream
    else:
        freqs = []	
        for tr in stream:
            freqs.append(tr.stats.sampling_rate)

    freq = max(set(freqs),key=freqs.count)
    for tr in stream:
        if tr.stats.sampling_rate != freq:
            stream.remove(tr)

    return stream	


def check_length(stream):
    """
    Forces all traces to have same number of samples.

    Traces must be one day long.
    :type stream:`~obspy.core.stream.Stream` object. 
    :param stream: Stream containing one or more day-long trace 
    :return: Stream of similar length traces 

    """
    pts = 24*3600*stream[0].stats.sampling_rate
    npts = []
    for trace in stream:
        npts.append(trace.stats.npts)
    npts = np.array(npts)
    if len(npts) == 0:
        return stream	
    index = np.where(npts != pts)
    index = list(index[0])[::-1]

    # remove short traces
    for trace in index:
        stream.pop(trace)

    return stream				


def downsample(stream,freq):
    """ 
    Downsamples stream to specified samplerate.

    Uses Obspy.core.trace.decimate if mod(sampling_rate) == 0. 
    :type stream:`~obspy.core.trace.Stream` or `~obspy.core.trace.Trace` object.
    :type freq: float
    :param freq: Frequency to which waveforms in stream are downsampled
    :return: Downsampled trace or stream object
    :rtype: `~obspy.core.trace.Trace` or `~obspy.core.trace.Trace` object.
    """
    
    # get sampling rate 
    if type(stream) == obspy.core.stream.Stream:
        sampling_rate = stream[0].stats.sampling_rate
    elif type(stream) == obspy.core.trace.Trace:
        sampling_rate = stream.stats.sampling_rate

    if sampling_rate == freq:
        pass
    else:
        stream.interpolate(freq,method="weighted_average_slopes")	

    return stream


def remove_resp(arr,stats,inv):
    """
    Removes instrument response of cross-correlation

    :type arr: numpy.ndarray 
    :type stats: `~obspy.core.trace.Stats` object.
    :type inv: `~obspy.core.inventory.inventory.Inventory`
    :param inv: StationXML file containing response information
    :returns: cross-correlation with response removed
    """	
    
    def arr_to_trace(arr,stats):
        tr = obspy.Trace(arr)
        tr.stats = stats
        tr.stats.npts = len(tr.data)
        return tr

    # prefilter and remove response
    
    st = obspy.Stream()
    if len(arr.shape) == 2:
        for row in arr:
            tr = arr_to_trace(row,stats)
            st += tr
    else:
        tr = arr_to_trace(arr,stats)
        st += tr
    min_freq = 1/tr.stats.npts*tr.stats.sampling_rate
    min_freq = np.max([min_freq,0.005])
    pre_filt = [min_freq,min_freq*1.5, 0.9*tr.stats.sampling_rate, 0.95*tr.stats.sampling_rate]
    st.attach_response(inv)
    st.remove_response(output="VEL",pre_filt=pre_filt) 

    if len(st) > 1: 
        data = []
        for tr in st:
            data.append(tr.data)
        data = np.array(data)
    else:
        data = st[0].data
    return data			


def preprocess(trace,percent=0.01,max_len=20.):   
    """
    Removes linear trend and mean, normalizes and tapers Obspy trace. 
    
    :type trace:`~obspy.core.trace.Trace` object.   
    :type: percent: float, optional
    :param percent: percent window on each end of trace to taper
    :return: Processed trace 
    """
    trace.detrend(type='constant')
    trace.detrend(type='simple')
    percent = trace.stats.sampling_rate * 20 / trace.stats.npts
    trace.taper(max_percentage=percent,max_length=max_len) 	

    return trace


def mad(arr):
    """ 
    Median Absolute Deviation: MAD = median(|Xi- median(X)|)
    :type arr: numpy.ndarray
    :param arr: seismic trace data array 
    :return: Median Absolute Deviation of data
    """
    if not np.ma.is_masked(arr):
        med = np.median(arr)
        data = np.median(np.abs(arr - med))
    else:
        med = np.ma.median(arr)
        data = np.ma.median(np.ma.abs(arr-med))
    return data	
    

def calc_distance(sta1,sta2):
    """ 
    Calcs distance in km, azimuth and back-azimuth between sta1, sta2. 

    Uses obspy.geodetics.base.gps2dist_azimuth for distance calculation. 
    :type sta1: dict
    :param sta1: dict with latitude, elevation_in_m, and longitude of station 1
    :type sta2: dict
    :param sta2: dict with latitude, elevation_in_m, and longitude of station 2
    :return: distance in km, azimuth sta1 -> sta2, and back azimuth sta2 -> sta1
    :rtype: float

    """

    # get coordinates 
    lon1 = sta1['longitude']
    lat1 = sta1['latitude']
    lon2 = sta2['longitude']
    lat2 = sta2['latitude']

    # calculate distance and return 
    dist,azi,baz = obspy.geodetics.base.gps2dist_azimuth(lat1,lon1,lat2,lon2)
    dist /= 1000.
    return dist,azi,baz


def getGaps(stream, min_gap=None, max_gap=None):
    # Create shallow copy of the traces to be able to sort them later on.
    copied_traces = copy.copy(stream.traces)
    stream.sort()
    gap_list = []
    for _i in range(len(stream.traces) - 1):
        # skip traces with different network, station, location or channel
        if stream.traces[_i].id != stream.traces[_i + 1].id:
            continue
        # different sampling rates should always result in a gap or overlap
        if stream.traces[_i].stats.delta == stream.traces[_i + 1].stats.delta:
            flag = True
        else:
            flag = False
        stats = stream.traces[_i].stats
        stime = stats['endtime']
        etime = stream.traces[_i + 1].stats['starttime']
        delta = etime.timestamp - stime.timestamp
        # Check that any overlap is not larger than the trace coverage
        if delta < 0:
            temp = stream.traces[_i + 1].stats['endtime'].timestamp - \
                etime.timestamp
            if (delta * -1) > temp:
                delta = -1 * temp
        # Check gap/overlap criteria
        if min_gap and delta < min_gap:
            continue
        if max_gap and delta > max_gap:
            continue
        # Number of missing samples
        nsamples = int(round(np.abs(delta) * stats['sampling_rate']))
        # skip if is equal to delta (1 / sampling rate)
        if flag and nsamples == 1:
            continue
        elif delta > 0:
            nsamples -= 1
        else:
            nsamples += 1
        gap_list.append([_i, _i+1,
                        stats['network'], stats['station'],
                        stats['location'], stats['channel'],
                        stime, etime, delta, nsamples])
    # Set the original traces to not alter the stream object.
    stream.traces = copied_traces
    return gap_list		


def cross_corr_parameters(source,receiver,num_corr,locs,maxlag):
    """ 
    Creates parameter dict for cross-correlations and header info to ASDF.  

    :type source: `~obspy.core.trace.Stats` object.
    :param source: Stats header from xcorr source station
    :type receiver: `~obspy.core.trace.Stats` object.
    :param receiver: Stats header from xcorr receiver station
    :type num_corr: int
    :param num_corr: number of cross-correlation functions in stack
    :type locs: dict
    :param locs: dict with latitude, elevation_in_m, and longitude of all stations
    :type maxlag: int
    :param maxlag: number of lag points in cross-correlation (sample points) 
    :return: Auxiliary data parameter dict
    :rtype: dict

    """

    # source and receiver locations in dict with lat, elevation_in_m, and lon
    source_loc = locs[source.network + '.' + source.station]
    receiver_loc = locs[receiver.network + '.' + receiver.station]

    # get distance (in km), azimuth and back azimuth
    dist,azi,baz = calc_distance(source_loc,receiver_loc)	

    # stack duration is end time of stack - start time of stack
    stack_duration = source.endtime - source.starttime
    
    # fill Correlation attribDict 
    parameters = {
            'source':str(source.station), 
            'source_net':str(source.network),
            'receiver':str(receiver.station),
            'receiver_net':str(receiver.network),
            'comp':source.channel[-1] + receiver.channel[-1],
            'sampling_rate':source.sampling_rate,
            'ccf_windows':num_corr,
            'stack_duration':stack_duration,
            'start_year':source.starttime.year,
            'start_month':source.starttime.month,
            'start_day':source.starttime.day,
            'start_hour':source.starttime.hour,
            'start_minute':source.starttime.minute,
            'start_second':source.starttime.second,
            'start_microsecond':source.starttime.microsecond,
            'end_year':source.endtime.year,
            'end_month':source.endtime.month,
            'end_day':source.endtime.day,
            'end_hour':source.endtime.hour,
            'end_minute':source.endtime.minute,
            'end_second':source.endtime.second,
            'end_microsecond':source.endtime.microsecond,
            'source_lon':source_loc['longitude'],
            'source_lat':source_loc['latitude'],
            'receiver_lon':receiver_loc['longitude'],
            'receiver_lat':receiver_loc['latitude'],
            'dist':dist,
            'azi':azi,
            'baz':baz,
            'lag':maxlag}
    
    return parameters


def stack_parameters(params):
    """
    Creates parameter dict for monthly stack.

    :type params: list (of dicts)
    :param params: List of dicts, created by cross_corr_parameters, for daily cross-correlations

    """

    month = params[0]
    for day in params[1:]:
        month['ccf_windows'].append(day['ccf_windows'])
        month['start_day'].append(day['start_day'])
        month['start_month'].append(day['start_month'])
        month['end_day'].append(day['end_day'])
    month['end_year'].append(day['end_year'])	
    month['end_hour'].append(day['end_hour'])	
    month['end_minute'].append(day['end_minute'])
    month['end_second'].append(day['end_second'])
    month['end_microsecond'].append(day['end_microsecond'])
    return month


def load_corr(corr_h5,comp):
    """
    Load correlations into numpy array. Prepares for input into MWCS.

    :type h5: str 
    :param h5: path/filename (/Volumes/.../../~.h5) to save cross-correlations as ASDF data set. Must end in .h5
    :param comp: Components used in cross-correlation, e.g. 'ZZ', 'RT', 'TT'
    """
    
    # query dataset 
    net_sta = os.path.basename(corr_h5).replace('.h5','')
    
    with pyasdf.ASDFDataSet(corr_h5,mpi=False) as ds:
        corrs = ds.auxiliary_data.CrossCorrelation[net_sta][comp].list()

        # data to return 
        all_param = []
        all_data = []

        for corr in corrs:
            data = ds.auxiliary_data.CrossCorrelation[net_sta][comp][corr].data
            param = ds.auxiliary_data.CrossCorrelation[net_sta][comp][corr].parameters
            all_data.append(np.array(data))
            all_param.append(param)
        all_data = np.array(all_data)
        all_data = np.vstack(all_data)
        days = [c[-10:].replace('_','/') for c in corrs]

    return all_data, all_param, days, net_sta 	


def load_ref(ref_h5,comp):
    """
    Load references  into numpy array. Prepares for input into MWCS.

    :type h5: str 
    :param h5: path/filename (/Volumes/.../../~.h5) to save cross-correlations as ASDF data set. Must end in .h5
    :param comp: Components used in cross-correlation, e.g. 'ZZ', 'RT', 'TT'
    """

    # query dataset 
    net_sta = os.path.basename(ref_h5).replace('.h5','')
    
    with pyasdf.ASDFDataSet(ref_h5,mpi=False) as ds:
        ref_list = ds.auxiliary_data.Reference[net_sta][comp].list()
        ref = [r for r in ref_list if 'ALL' in r][0]
        data = ds.auxiliary_data.Reference[net_sta][comp][ref].data
        param = ds.auxiliary_data.Reference[net_sta][comp][ref].parameters
        ref = np.array(data)

    return ref,param


def vcorrcoef(X,y):
    """
    Vectorized Cross-correlation coefficient in the time domain

    :type X: `~numpy.ndarray`
    :param X: Matrix containing time series in each row (ndim==2)
    :type X: `~numpy.ndarray`
    :param X: time series array (ndim==1)
    
    :rtype:  `~numpy.ndarray`
    :return: **cc** array of cross-correlation coefficient between rows of X and y
    """
    
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    cc_num = np.sum((X-Xm)*(y-ym),axis=1)
    cc_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
    cc = cc_num/cc_den
    return cc


def spect(tr,fmin = 0.1,fmax = None,wlen=10,title=None):
    import matplotlib as plt
    if fmax is None:
        fmax = tr.stats.sampling_rate/2
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2]) #[left bottom width height]
    ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1)
    ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])

    #make time vector
    t = np.arange(tr.stats.npts) / tr.stats.sampling_rate

    #plot waveform (top subfigure)    
    ax1.plot(t, tr.data, 'k')

    #plot spectrogram (bottom subfigure)
    tr2 = tr.copy()
    fig = tr2.spectrogram(per_lap=0.9,wlen=wlen,show=False, axes=ax2)
    mappable = ax2.images[0]
    plt.colorbar(mappable=mappable, cax=ax3)
    ax2.set_ylim(fmin, fmax)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Frequency [Hz]')
    if title:
        plt.suptitle(title)
    else:
        plt.suptitle('{}.{}.{} {}'.format(tr.stats.network,tr.stats.station,
                  tr.stats.channel,tr.stats.starttime))
    plt.show()


if __name__ == "__main__":
    pass
