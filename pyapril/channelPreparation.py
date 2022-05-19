# - coding: utf-8 -*-
import numpy as np
from pyapril import clutterCancellation as CC
try:
    from pyargus import beamform as bf         
    has_pyargus = True
except ImportError as e:
    print("WARNING: PyArgus is not installed, beamspace processing methods will not work")
    has_pyargus = False
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

     Version history :
         - Ver 1.0.0     : Initial version (2017 04 11)
         - Ver 1.1.0     : Surveillance beamformer function (2017 05 13)
         - Ver 1.1.1     : Error corr. in ref and surv BF (2017 05 15)
         - Ver 1.2.0     : Implementation of time domain filters(2017 06 01)
         - Ver 1.3.0     : Implementation of pre-filtering techniques (2017 06)
         - Ver 1.3.1     : Rebase beamformer wrapper function (2020 01 11)



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

        -SMI        : Sample Matrix Inversion. Estimates the R auto-correlation matrix and r cross-correlation
                      vector with the sample average technique.
                
        -LMS        : Least Mean Square method.
        
        -NLMS       : Normalized Least Mean Square method is the modification of the LMS method in a sense that the
                      input vector (reference signal) is normalized with it's Euclidean norm.
                 
        -block LMS  : Batched version of the LMS algorithm
        -block NLMS : Batched version of the NLMS algorithm
        -RLS        : Recursive Least Square method uses the "exponential forgetting factor - lambda" to estimate the
                      time-varying cross-correlation matrix. Beside that the algorithm utilises the
                      Sherman-Morrison formula to calculate the inverse of a modified matrix when the modification is only
                      an addition of a dyad.
        -ECA        : Extensive Cancellation Algorithm is the the extension of the standard time domain Wiener filter
                      in which the subspace of the reference signal is extended into the Doppler domain.
        -ECA-B      : Batched version of the ECA algorithm
        -ECA-B&S    : Batched and Sequenced version of the ECA filter
        -ECA-S      : Sliding window version of ECA algorithm
        -SMI-MRE    : Minimum Redundancy Estimation with Sample Matrix Inversion. In this technique only the first column of the 
                      full auto-correlation matrix is calculated (with sample average) and the remaining elements are completed 
                      by utilizing the Hermitian and Toeplitz properties of the matrix.
        -ECA-S MRE  : Minimum Redundancy Estimation for the batched version of the ECA algorithm
        
        w_init parameter can be used to implement weight inheritance across the consecutive processing blocks when iterative
        algorithms are used.

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

def beamform_surveillance(ant_chs, alignment_vector, method, **kwargs):
    """
        Description:
        -----------
            The underlying function produces the signal of the surveillance channel 
            from the antenna channels with the application of beamspace processing
            techniques. Using the method string the applied technique can be selected.

            Available methods: (For detailed descriptions please check the pyargus library)
                - Explicit: Coefficient vector is specified to the function.
                
                - Beam steering: Simply steers the direction of the main beam with
                                 compensating the progressive phase shifts on the
                                 antenna elements.
                
                - MSIR : Maximum signal to interference ratio. Fixed beamforming method. 
                         The direction of the main beam an the interferences must be 
                         specified. If the specified constraints are less than the 
                         antenna channels (degrees of freedom) Godara's method is used.
                - DMSIR : DoA estimation aided Max SIR beamformer. This method tries to 
                          automatically estimate the direction of illuminators and places
                          nulls in the radiation pattern into these directions.
                - MVDR : Minimum Variance Distortionless Response. Adaptive optimum 
                         beamforming.
                         
                - Principal eigenvalue: Subspace based technique. To reject the 
                         high power clutter components, it projects the surveillance
                         channels to a subspace ortogonal to the principal eigenvalues.

        Implementation notes:
        ---------------------
            This function can handle only 1D antenna array processing techniques.
            Coefficent values are normalized prior to the beamspace processing.

        Parameters:
        -----------

            :param ant_chs         : Multichannal complex signal array of the antenna channels
            :param alignment_vector: Contains the positions of the antenna elements in the antenna system. 
                                     The distances are interpreted in wavelength. e.g.: If d=lambda/2
                                     [ 0.5 0.75 1.25 ]  --> x<---->x<-->x<---->x
                                                              d     d/2   d
            :param method          : Beamspace processing method. Valid values are the followings: 
                                     "Explicit","MSIR","MVDR","Principal eigenvalue"
            
            :type ant_chs          : (N x M) complex numpy array, where M is the number of antenna channels
                                     and N is the sample size
            :type alignment_vector : 1 dimensional float array
            :type method           : string
                
        **kwargs
            The required space domain filter (beamformer) parameters for each filter are received 
            through the **kwargs interface
            
            Valid keys are the followings:
            :key explicit_w         : This coefficient vector is used when the explicit beamformer method is specified. -Explicit
            :key MSIR_constraints   : First row contains the angles and the second row contains the constraint values. - MSIR
            :key direction          : Main beam direction for the adaptive optimum beamformer. - MVDR
                                      The dimension is deg, thus the valid range for float values is: 0 < .. < 180.
            :key peigs              : Sets the subspace dimension of the principal signal components. - Principal eigenvalue
            :key Rnn                : Spatial correclation matrix of the noise plus interferences - MVDR

            :type explicit_w        : (M x 1) complex numpy array
            :type MSIR_constraints  : (2 x [M-1]) numpy array
            :type direction         : float
            :type peigs             : int
            :type Rnn               : (M x M) complex numpy array
            

        Return values:
        --------------
            :return reference_channel : (N by 1 complex numpy array) Reference channel obtained with beamspace processing.
            :return coefficient_vector: (M by 1 complex numpy array) calculated and used coefficient vector
            :return None, None : In case of calculation failure
    """
    # TODO: Implement and wrap reference orthogonal beamformer
    if not has_pyargus:
        print("ERROR: PyArgus is not installed!")
        print("No valid outputs are generated!")
        return None, None

    # kwargs processing
    explicit_w = kwargs.get('explicit_w') 
    MSIR_constraints = kwargs.get('MSIR_constraints') 
    direction = kwargs.get('direction') 
    peigs = kwargs.get('peigs') 
    Rnn = kwargs.get('Rnn') 
    
    # --- Input check ---
    N = np.size(ant_chs, 0)
    M = np.size(ant_chs, 1)
    
    if M > N:
        print("WARNING: The number of antenna channels is greater than the number of samples in a channel. M > N")
        print("You may flipped the input matrix")
        print("No valid outputs are generated!")
        print("N:", N)
        print("M:", M)
        return None, None
    if M != np.size(alignment_vector):
        print("ERROR: Mismatch in the input signal channel numbers and the antenna alignment vector size")
        print("No valid outputs are generated!")
        return None, None

    if method == "Max SIR" and MSIR_constraints is None:
        print("ERROR: Constraints are not specified for the Max SIR method")
        print("No valid outputs are generated!")
        return None, None

    if method == "Explicit" and explicit_w is None:
        print("ERROR: Coefficient vector is not specified, but the beamforming method is set to explicit")
        print("No valid outputs are generated!")
        return None, None

    if method == "Explicit" and (np.size(explicit_w, 0) != M) is None:
        print("ERROR: The number of antenna elements and the size of the given coefficient vector does not match")
        print("No valid outputs are generated!")
        return None, None

    if method == "MVDR" and direction is None:
        print("ERROR: Direction of the desired signal is not specified for the adaptive optimum beamformer (MVDR)")
        print("No valid outputs are generated!")
        return None, None

    if method == "Principal eigenvalue" and peigs is None:
        print("ERROR: Principal eigenvalue count is not specified for the subspace based beamformer")
        print("No valid outputs are generated!")
        return None, None

    if direction > 180 or direction < 0:
        print("ERROR: Direction of the desired signal is not in the range of 0..180 deg")
        print("No valid outputs are generated!")
        return None, None

    # -- Coefficient calculation --
    if method == 'Max SIR':
        if np.size(MSIR_constraints, 0) == M:
            w = bf.fixed_max_SIR_beamform(MSIR_constraints[0, :], 
                                          MSIR_constraints[1, :], 
                                          alignment_vector)
        else:  # Apply Godara's method
            w = bf.Goadar_max_sir_beamform(MSIR_constraints[0, :], 
                                           MSIR_constraints[1, :], 
                                           alignment_vector)

    elif method == 'DOA aided Max SIR':
        print("ERROR: This method is currently not implemented")
        print("No valid outputs are generated!")
        return None, None
        #w = CC.maxSIR_DOA(surv_chs=ant_chs, array_alignment=alignment_vector, 
        #                  doa_method="MEM", target_DOA=direction)

    elif method == "MVDR":
        if Rnn is None:
            R = bf.estimate_corr_matrix(ant_chs, imp="fast")
        else:
            R = Rnn
        # Create array response vector for the desired signal angle - ULA assumpation
        aS = np.exp(alignment_vector * 1j * 2 * np.pi * np.cos(np.deg2rad(direction)))
        aS = np.matrix(aS).reshape(M, 1)
        
        # Calculate optimal weight coefficient vector
        w = bf.optimal_Wiener_beamform(R, aS)

    elif method == "Principal eigenvalue":

        R = bf.estimate_corr_matrix(ant_chs, imp="fast")

        # Create array response vector for the desired signal angle
        aS = np.exp(alignment_vector * 1j * 2 * np.pi * np.cos(np.deg2rad(direction)))
        aS = np.matrix(aS).reshape(M, 1)

        # Calculate optimal weight coefficient vector
        w = bf.peigen_bemform(R, aS, peigs)
    
    elif method == "Beam steering":
        w = np.exp(alignment_vector * 1j * 2 * np.pi * np.cos(np.deg2rad(direction)))

    elif method == "Explicit":
        w = explicit_w

    norm_factor = np.sqrt(M) / np.sum(np.abs(w))
    w = w * norm_factor

    # -- Beamspace processing --
    surv_ch = np.inner(np.transpose(np.conjugate(w)), ant_chs)[:]

    return surv_ch, w

