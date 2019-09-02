# - coding: utf-8 -*-
import numpy as np
import clutterCancellation as CC
"""

                             Python based Advanced Passive Radar Library (pyAPRiL)

                                               Channel preparation


     Description:
     ------------

        This file is intended to collect and wrap all the passive radar channel preparation related functions.
        These functions isolate the reference and surveillance channels and perform the necessary filtering steps
        before the detection. The main aim of this functions set is to collect all the required tools to produce the
        reference and surveillance signal processing channels from the raw digitalized and downloaded antenna channels.

        The reference channel preparation related functions are the followings:
          - func: isolate_channels(iq_samples_matrix, ref_position): Use this function if the reference signal is
                  received on a dedicated antenna and receiver chain.
          - func: beamform_reference(iq_signal_matrix, method , params .... ): Creates the reference channel via
                  beam space processing from the available antenna channels matrix.
                  (Only in later versions !)
          - func: regenerate_reference(reference_channel, params ....): Regenerates the reference signal using the
                  available information from its modulation properties. (DVB-T, ..)
                  (Only in later versions !)
          - func: beamform_surveillance(iq_signal_matrix, method , params .... ): Creates the surveillance channel via
                  beam space processing from the available antenna channels matrix.
                  (Only in later versions !)
          - func: time_domain_filter_surveillance(reference_ch, surveillance_ch, ...) Filters the surveillance channel
                  in time domain using available reference channel.
                  (Only in later versions !)
          - func: prefilter_surveillance(reference_ch, surveillance_ch, method, ...) Filters all the surveillance
                  channels before the beamforming is performed. 
                  (Only in later versions !)

     Demonstrations:
     ---------------
        The underlying functions perform demonstration for the previously described channel preparing operations.

     Notes:
     ------------

        TODO: Write demonstration functions
    
     Dependencies:
     -------------
         - pyARGUS library is required to run these filtering functions
         - Most of the algorithms are implemented in the file named "clutterCancellation"

     Features:
     ------------

     Project: pyAPRiL

     Authors: Tamás Pető

     License: GNU General Public License v3 (GPLv3)

     Changelog :
         - Ver 1.0000    : Initial version (2017 04 11)
         - Ver 1.0010    : Surveillance beamformer function (2017 05 13)
         - Ver 1.0011    : Error corr. in ref and surv BF (2017 05 15)
         - Ver 1.0020    : Implementation of time domain filters(2017 06 01)
         - Ver 1.0030    : Implementation of pre-filtering techniques (2017 06)


     Version format: Ver X.ABCD
                         A - Complete code reorganization
                         B - Processing stage modification or implementation
                         C - Algorithm modification or implementation
                         D - Error correction, Status display modification

 """


def isolate_channels(iq_samples_matrix, ref_position):
    """
        Description:
        -----------
        This function assumes that the surveillance and reference channels are digitalized and downloaded together
        in the digital data acquisition stage, however the reference signal is received on a dedicated receiver and
        antenna channel. At the output of the function the isolated reference signal and the rearranged matrix of
        the surveillance channels will appear.

        Parameters:
        -----------

            :param: iq_samples_matrix:  This matrix contains the signal array download from
                               the receiver hardware, which contains all the antenna channels. M is the count of the
                               surveillance channels, while N is the number of samples in a channel.
            :param: ref_position: Index of the reference channel is the iq_sample matrix.
            :type iq_samples_matrix:  M+1 x N complex numpy array
            :type ref_position: int

        Return values:
        --------------
            :return: reference_channel:  Isolated reference channel
            :return: surveillance_channels: Isolated surveillance channels
            :rtype: sureveillance channel: (N by 1 numpy array)
            :rtype: reference_channel: (N by M numpy array)
    """
     # --- Input check ---
    N = np.size(iq_samples_matrix, 1)
    Mp1 = np.size(iq_samples_matrix, 0)
    
    if Mp1 > N:
        print("WARNING: The number of antenna channels is greater than the number of samples in a channel. M+1 > N")
        print("You may flipped the input matrix")
        print("N:", N)
        print("M+1:", Mp1)
    
    if ref_position >= Mp1:
        print("ERROR: Invalid reference channel index. The requested index is greater than the "
              "number of channels.\nValid indexes are:  0..%d" % (Mp1-1))
        print("Requested ch index: ", ref_position)
        print("No valid outputs are generated!")
        return None, None

    # --- Process ---
    reference_channel = np.zeros(N, dtype=complex)
    reference_channel[:] = iq_samples_matrix[ref_position, :]
    
    surveillance_channels = np.zeros((Mp1-1, N), dtype=complex)
    surveillance_channels[:] = np.delete(iq_samples_matrix, ref_position, 0)[:]
    
    return reference_channel, surveillance_channels

def time_domain_filter_surveillance(ref_ch, surv_ch, method, **kwargs):
    """
    Description:
    -----------
        Filters the surveillance channel in the time and Doppler domain. The implemented filters are based on the
        optimum Wiener filter but differs in the implementation and subspace selection techniques. The implemented
        filters are the followings:

        -SMI : Sample Matrix Inversion. Estimates the R auto-correlation matrix and r cross-correlation
                      vector with averaging.
                
        -LMS        : Least Mean Square method.
        
        -NLMS       : Normalized Least Mean Square method is the modification of the LMS method in a sense that the
                      input vector (reference signal) is normalized with it's Euclidean norm.
                 
        -block LMS  : Batched version of the LMS algorithm
        -block NLMS : Batched version of the NLMS algorithm
        -RLS        : Recursive Least Square method uses the "exponential forgetting factor - lambda" to estimate the
                      time-varying cross-correlation matrix. Beside that the algorithm utilises the
                      Sherman-Morrison formula to calculate to inverse of a modified matrix when the modification is only
                      the addition of a dyad.
        -ECA        : Extensive Cancellation Algorithm is the the extension of the standard time domain Wiener filter
                      The subspace of the reference signal is extended into the Doppler domain.
        -ECA-B      : Batched version of the ECA algorithm
        -ECA-B&S    : Batched and Sequenced version of the ECA filter
        -ECA-S      : Sliding window version of ECA algorithm
        -SMI_MRE    : Minimum Redundancy Estimation with Sample Matrix Inversion technique. Only the first column of the full auto-correlation
                      matrix is calculated, the remaining elements are completed using its Hermitian and Toeplitz property.
        -ECA-S MRE  : Minimum Redundancy Estimation for the batched version of the ECA algorithm
        
        w_init parameter can be used to implement weight inheritance across consecutive processing in iterative
        algorithms.

    Parameters:
    -----------
        :param sur_ch : Contains the complex signal samples of the surveillance channel                
        :param ref_ch : Complex signal samples of the reference channel
        :param method : Selects the method to use.
        
        :type surv_ch : (N x 1 complex numpy array)
        :type ref_ch  : (N x 1 complex numpy array)
        :type method  : string
            
     **kwargs
        The required time domain filter parameters for each filter are received 
        through the **kwargs interface
        
        Valid keys are the followings:
        
        :key K      : Time domain filter dimension
        :key D      : Maximum Doppler extension measured in Doppler bins
        :key T      : Number of batches for ECA-B or batch length for block LMS
        :key Na     : Sliding window size, measured in samples
        :key imp    : Implementation type
        :key mu     : Step size parameter for the iterative algorithms - LMS or NLMS
        :key lamb   : Forgetting factor for the RLS algoritm
        :key ui     : Update interval for the iterative algorithms - LMS, NLMS, RLS
        :key w_init : Initialization vector for the iterative algorithms - LMS, NLMS, RLS, (default: None)

        :type K      : int
        :type D      : int
        :type T      : int
        :type Na     : int
        :type mu     : float
        :type lambd  : float
        :type ui     : int
        :type imp    : string
        :type w_init : (K x 1 complex numpy array)

    Return values:
    --------------
        :return: filtered_surveillance_channel
        :return: w: Time domain filter coefficients
        :return error_array: Instantaneous error from the iterative filters
        :rtype: N x 1 complex numpy array, D
        :rtype: w: Complex numpy matrix, size is dependent on the selected method
        :rtype: error_array: 1D complex numpy array

        :return None,None : In case of calculation failure
    """

    # --input check--
    if ref_ch is None or surv_ch is None:
        print("ERROR: An input channel is None objects")
        print("ERROR: No output is generated")
        return None
    # kwargs processing
    K = kwargs.get('K') # Time domain filter dimension
    D = kwargs.get('D') # Doppler domain filter dimension
    #T = 1, D=0, mu=1, lambd=1, ui=1, Na=0, imp="fast", w_init=None
    # --init--
    w = None
    error_array = None

    # --calculation--
    if method == "SMI":
        pass
    elif method == "LMS":
        pass
    elif method == "NLMS":        
        pass
    elif method == "RLS":
        pass
    elif method == "block LMS":
        pass
    elif method == "block NLMS":
        pass

    elif method == "SMI-MRE":
        if K is not None:
            filtered_surv_ch, w = CC.Wiener_SMI_MRE(ref_ch, surv_ch, K)
        else:
            print("ERROR: Insufficient filter parameters")
            return None

    elif method == "ECA":
        if not (K is None or D is None):            
            subspace_list = CC.gen_subspace_indexes(K, D)
            filtered_surv_ch = CC.ECAS(ref_ch, surv_ch, subspace_list, T=1, Na=0)
        else:
            print("ERROR: Insufficient filter parameters")
            return None

    elif method == "ECA-S":
        pass
    elif method == "S ECA-S":
        pass
    else:
        print("ERROR: The specified method is not recognized")
        print("ERROR: No output is generated")
        return None

    return filtered_surv_ch
