import os 
import glob

import numpy as np
import pyasdf
from mpi4py import MPI
from noise import clean_up, pws


def main(corr_h5, REF, stack_method='pws'):
    """
    Stack cross-correlations to make stable reference. 
    
    Creates reference correlation for station-pairs. 
    Stacks cross-correlations by year, and all cross-correlations. 
    Takes weighted mean of each day by number of hourly cross-correlations in each daily stack.

    :type corr_h5: str 
    :param corr_h5: path/filename (/Volumes/.../../~.h5) to access cross-correlations 
                    as ASDF data set. Must end in .h5
    :type REF: str 
    :param REF: path/filename to reference correlation directory 

    Last modified by thclements@g.harvard.edu 11/2/16

    """

    data_type = 'CrossCorrelation'
    # load year correlation file - do nothing if file is empty
    try:
        corr_ds = pyasdf.ASDFDataSet(corr_h5, mpi=False)
        net_sta = corr_ds.auxiliary_data[data_type].list()[0]
    except Exception as e:
        return

    # Load ref HDF5
    ref = os.path.basename(corr_h5)

    if not os.path.exists(REF):
        os.makedirs(REF)
    ref_h5 = os.path.join(REF, ref) 

    if not os.path.isfile(ref_h5):
        ref_ds = pyasdf.ASDFDataSet(ref_h5, compression=None, mpi=False)
    else:
        ref_ds = pyasdf.ASDFDataSet(ref_h5,mpi=False)
    comps = corr_ds.auxiliary_data[data_type][net_sta].list()

    # run through the correlations 
    for comp in comps:
        try:
            corrs = corr_ds.auxiliary_data[data_type][net_sta][comp].list()
        except Exception as e:
            print(e)
            continue

        # load data 
        all_corr = []
        for corr in corrs:
            params = corr_ds.auxiliary_data[data_type][net_sta][comp][corr].parameters
            ccf = corr_ds.auxiliary_data[data_type][net_sta][comp][corr].data
            ccf = np.array(ccf)
            sampling_rate = params['source_sampling_rate']
            all_corr.append(ccf)

        # add reference correlation for all years   
        all_corr = np.vstack(all_corr)
        if stack_method == 'pws':
            ref = pws(all_corr,sampling_rate=sampling_rate,pws_timegate=0.1)
        else:
            ref = np.nanmean(all_corr,axis=0)
        # create ref
        ref_str = '_'.join(['ref',net_sta,'ALL'])
        path = os.path.join(net_sta,comp,ref_str)
        parameters = {'lag':params['lag'],
                      'sampling_rate':params['source_sampling_rate'],
                      'dist':params['dist']}

        # Add auxiliary data of type Reference
        ref_ds.add_auxiliary_data(data=ref,
                                  data_type='Reference', 
                                  path=path,
                                  parameters=parameters)
    return
    
if __name__ == '__main__':
    pass
