import os
import glob
import sys
import datetime

import pyasdf
import numpy as np
import pandas as pd 
import scipy.signal
from mpi4py import MPI
from noise import pws, clean_up, runningMean


"""

Merge station-pair cross-correlation HDF5 for each year into one file

"""


def main(corr_h5, freqmin, freqmax, hours=24, minutes=0, step=0, max_STD=50, stack_method='pws'):
    """

    Adds yearly station pair HDF5 file to station pair hdf5 for all years.

    :type corr_h5: str
    :param corr_h5: path to HDF5 file e.g. '/n/regal/denolle_lab/tclements/San_Gabriel/Correlation/corr_1997_MWC_PAS.h5'
        """
    data_type = 'CrossCorrelation'

    # get name of ALL correlation file
    corr_dir, h5 = os.path.split(corr_h5)
    corr_dir = corr_dir.replace('CORR', 'CORR_ALL')
    freq_str = '_'.join([str(freqmin), str(freqmax)])
    corr_dir = os.path.join(corr_dir, freq_str)
    # if not os.path.isdir(corr_dir):
    #   os.makedirs(corr_dir)
    corr_all_h5 = os.path.join(corr_dir, h5)

    # load year correlation file - do nothing if file is empty
    with pyasdf.ASDFDataSet(corr_h5, mpi=False) as corr_ds:
        net_sta = corr_ds.auxiliary_data[data_type].list()[0]
        comps = corr_ds.auxiliary_data[data_type][net_sta].list()

        # load/create all correlation HDF5 file
        with pyasdf.ASDFDataSet(corr_all_h5, compression=None, mpi=False) as all_ds:

            for comp in comps:
                corr_list = corr_ds.auxiliary_data[data_type][net_sta][comp].list()

                # run through daily correlations
                for corr in corr_list:

                    # load correlation
                    params = corr_ds.auxiliary_data[data_type][net_sta][comp][corr].parameters
                    ccf = corr_ds.auxiliary_data[data_type][net_sta][comp][corr].data
                    ccf = np.array(ccf)
                    sampling_rate = params['source_sampling_rate']
                    source_std = params['source_std']
                    receiver_std = params['receiver_std']
                    rec_ind = np.where(receiver_std < max_STD)[0]
                    sou_ind = np.where(source_std < max_STD)[0]
                    best = np.intersect1d(rec_ind, sou_ind)
                    if len(best) == 0:
                        print('{} All windows above {} STD'.format(corr, max_STD))
                        continue
                    if ccf.ndim == 2:  
                        ccf = ccf[best, :]

                    # subset by time 
                    starttime = params['starttime']
                    starttime = pd.to_datetime([datetime.datetime.utcfromtimestamp(s) for s in starttime])
                    endtime = params['endtime']
                    endtime = pd.to_datetime([datetime.datetime.utcfromtimestamp(s) for s in endtime])
                    s = starttime[np.argmin(starttime)]
                    e = endtime[np.argmax(endtime)]
                    intervals = np.vstack([
                        pd.date_range(s,e - datetime.timedelta(hours=hours,minutes=minutes),freq='{}h{}min'.format(0,step)),
                        pd.date_range(s+datetime.timedelta(hours=hours,minutes=minutes),e,freq='{}h{}min'.format(0,step))]).T
                    starttime, endtime = starttime[best], endtime[best]

                    # filter correlations and subset by time
                    try:
                        corrs = clean_up(ccf, sampling_rate, freqmin, freqmax)
                        # corrs = (corrs.T / np.abs(corrs.max(axis=1))).T
                        data = []
                        keep = []

                        # stack over intervals
                        for ii,interval in enumerate(intervals):
                            ind = np.where((starttime >= interval[0]) & (endtime <= interval[1]))[0]
                            if len(ind) > 0:
                                keep.append(ii)

                                # stack correlations
                                if stack_method == 'pws':
                                    interval_stack = pws(corrs[ind,:],sampling_rate=sampling_rate,pws_timegate=0.1)
                                else:
                                    interval_stack = np.mean(corrs[ind, :], axis=0)

                                data.append(interval_stack)
                        keep = np.hstack(keep) 
                        data = np.array(data)
                        intervals = intervals[keep, :]
                        params['hours_start'] = intervals[:, 0].astype('datetime64[s]').astype('int')
                        params['hours_end'] = intervals[:, 1].astype('datetime64[s]').astype('int')
                        params['hours'] = hours
                        params['minutes'] = minutes
                        path = os.path.join(net_sta, comp, corr)
                        all_ds.add_auxiliary_data(data=data,
                                                data_type=data_type,
                                                path=path,
                                                parameters=params)
                    except Exception as e:
                        print('Could not add {}'.format(corr))
                        print(e)

    return


if __name__ == '__main__':
    pass
