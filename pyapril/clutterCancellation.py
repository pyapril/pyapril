# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg as LA
"""
                             Python based Advanced Passive Radar Library (pyAPRiL)

                                           Channel Preparation Module
     Description:
     ------------

        List of implemented algorithms:

            - Wiener filter with Sample Matrix Inversion - Wiener_SMI     
            - Wiener filter with minimum redundance estiomation - Wiener_SSMI
            - Extensive Cancellation Algorithm ECA
            - Batched version of the Extensive Cancellation Algorithm - ECA-B 
            - Least Mean Squares method - LMS                                 (Only in later versions !)            
            - Sliding window Extensive Cancellation Algorithm - ECA-S         
            - Least Mean Square algorithm - LMS                               (Only in later versions !)
            - Normalized Least Mean Square algorithm - NLMS                   (Only in later versions !)
            - Recursive Least Squares - RLS                                   (Only in later versions !)
            - Block Least Mean Square algorithm - Block LMS                   (Only in later versions !)
            - Block Normalized Least Mean Square algorithm - Block NLMS       (Only in later versions !)
            - Space- Time Adaptive Cancellation - STAC                        (Only in later versions !)
            - Fixed maximum SIR beamformer with DOA estimation - MSIR-DOA     (Only in later versions !)
     Notes:
     ------------
        

     Features:
     ------------

        Project: pyAPRiL
        Author: Dávid Huber, Tamás Pető
        License: GNU General Public License v3 (GPLv3)

        Version history :
            - Ver 1.0.0      : Initial version (2017 04 27)
            - Ver 1.1.1      : Calc correction (2017 04 30)
            - Ver 1.2.0      : FFT based calc including pruned correlation func (2017 05 01)
            - Ver 1.3.0      : Wiener filter with SMI technique (2017 05 16)
            - Ver 1.4.0      : Least Mean Square filter (2017 05 16)
            - Ver 1.5.0      : Extensive Cancellation Algorithm with sliding window (2017 05 23)
            - Ver 1.6.0      : Iterative algorithms, LMS, NLMS, BLMS, BNLMS (2017 06 01)
            - Ver 1.7.0      : Maximum SIR beamformer with DOA estimation (2017 08 05)
            - Ver 1.7.1      : Better comments (2018 10 24)   

"""

def Wiener_SMI(ref_ch, surv_ch, K, imp="direct_for"):
    """
    Description:
    ------------
        
        This function operates in the time domain. 
        It performs Wiener filtering on the surveillance channel using the given reference channel. 
        The signal subspace is spanned from the time delayed replicas of the reference signal in a dimension of "K".
        Sample Matrix Inversion (SMI) technique use the sample average of the instantaneous values of the R
        autocorrelation matrix and r cross correlation vector to estimate the expected values.        

    Implementation notes:
    ---------------------
        Two different implementation exist for this function. One of them use a for loop to iterate through the
        signal samples while the other use a direct vector product from numpy. The latter consumes more memory
        but much faster for large arrays. The implementation can be selected using the "imp" function parameter.
        Set imp="direct_for" to use the memory efficient implementation with a for loop or set to "direct_matrix" in order to use
        the faster direct matrix product implementation.

    Parameters:
    -----------

    :param ref_ch: Signal vector of the reference channel
    :param surv_ch: Signal vector of the surveillance channel
    :param K: Filter dimension
    :param imp: Implementation method "direct_for" or "direct_matrx", default:"direct_for"

    :type ref_ch: N by 1 complex numpy array
    :type surv_ch: N by 1 complex numpy array
    :type K: int
    :type imp: string

    Return values:
    --------------

    :return filt_ch: Filtered surveillance channel
    :return w: Time domain filter coefficient vector
    :rtype N by 1 complex numpy array
    :rtype K by 1 complex numpy array

    :return None: General failure

    """
    
    # -- input check --
    if imp != "direct_matrix" and imp != "direct_for":
        print("WARNING: Unidentified implementation type")
        print("WARNING: direct_matrix method is used")
        imp = "direct_matrix"
        
    # -- calculation --
    N = ref_ch.size  # Get the number of samples of the coherent processing interval
    R = np.zeros((K, K), dtype=complex)  # Autocorrelation matrix allocation
    r = np.zeros((1, K), dtype=complex)  # cross correlation vector allocation
    filt_ch = np.zeros(N, dtype=complex)  # Filtered surveillance channel allocation

    if imp == "direct_matrix":
        # Prepare signal delay matrix for fast TD filtering
        X = np.zeros((N, K), dtype=complex)  # K dimensional signal subspace matrix
        X[:, 0] = ref_ch
        for i in np.arange(1, K):
            X[:, i] = np.pad(ref_ch, (i, 0), mode='constant')[:-i]
        X = X.T
        
        # Estimate the autocorrelation matrix and the cross-correlation vector
        R = np.dot(X, X.conj().T) 
        r = np.dot(X, surv_ch.conj())
        
        # Calculate coefficient vector
        w = np.dot(np.linalg.inv(R), np.transpose(r))
        
        # Perform filtering
        filt_ch = surv_ch - np.dot(X.T, w.conj().T)

    elif imp == "direct_for":
        # Estimate the autocorrelation matrix and the cross-correlation vector
        for i in np.arange(K, N):
            R += np.outer(ref_ch[i - K:i][::-1], np.conjugate(ref_ch[i - K:i][::-1]))
            r += (ref_ch[i - K:i][::-1]) * np.conjugate(surv_ch[i - 1])
        
        # Normalization with the sample size
        R = np.divide(R, N - K)
        r = np.divide(r, N - K)
        
        # Calculate coefficient vector
        w = np.dot(np.linalg.inv(R), np.transpose(r))
        
        # Prepare signal vectors for filtering
        filt_ch = np.zeros((N+1), dtype=complex)  # Filtered surveillance channel
        ext_ref = np.concatenate((np.zeros(K, dtype=complex), ref_ch), axis=0)  # extend signal with zeros
        x_n = np.zeros(K, dtype=complex)  # The last K samples of the reference signal

        # Filtering
        for n in np.arange(0, N + 1, 1):
            x_n[:] = ext_ref[n:n + K]  # Select last "l" element in the reference channel
            x_n = x_n[::-1]  # To preserve FIR filter format
            filt_ch[n] = np.dot(np.transpose(w.conj()), x_n)[0]  # Calculate filtered output
        filt_ch = filt_ch[1:N + 1]
        filt_ch = surv_ch - filt_ch

    return filt_ch, w


def pruned_correlation(y, x, clen):
    """
        Description:
        -----------
        Calculates the part of the correlation function of arrays with same size
        The total length of the cross-correlation function is 2*N-1, but this
        function calculates the values of the cross-correlation between [N-1 : N+clen-1]

        Parameters:
        -----------
        :param x : input array
        :param y : input array
        :param clen: correlation length
        
        :type x: 1 x N complex numpy array
        :type y: 1 x N complex numpy array
        :type clen: int

        Return values:
        --------------
        :return corr : part of the cross-correlation function
        :rtype  corr : 1 x clen complex numpy array
        
        :return None : inconsistent array size
    """
    
    # --input check--
    N = x.shape[0]
    if N != y.shape[0] or clen > (N + 1) / 2:
        print('ERROR:not shapeable')
        return None
    
    # --calculation--
    # set up input matrices pad zeros if not multiply of the correlation length
    cols = clen - 1
    rows = np.int32(N / (cols)) + 1
    zeropads = cols * rows - N
    x = np.concatenate((x, np.zeros(zeropads, dtype=complex)))
    y = np.concatenate((y, np.zeros(zeropads, dtype=complex)))

    # shaping inputs into matrices
    xp = np.reshape(x, (rows, cols))
    yp = np.reshape(y, (rows, cols))

    # padding matrices for FFT
    ypp = np.vstack([yp[1:, :], np.zeros(cols, dtype=complex)])
    yp = np.concatenate([yp, ypp], axis=1)
    xp = np.concatenate([xp, np.zeros((rows, cols), dtype=complex)], axis=1)

    # execute FFT on the matrices
    xpw = np.fft.fft(xp, axis=1)
    bpw = np.fft.fft(yp, axis=1)

    # magic formula which describes the unified equation of the universe
    corr_batches = np.fliplr(np.fft.fftshift(np.fft.ifft(np.multiply(xpw, bpw.conj()), axis=1)).conj()[:, 0:clen])

    # sum each value in a column of the batched correlation matrix
    return np.sum(corr_batches, axis=0)


def shift(x, i):
    """
        Description:
        -----------
        Similar to np.roll function, but not circularly shift values
        Example:
        x = |x0|x1|...|xN-1|
        y = shift(x,2)
        x --> y: |0|0|x0|x1|...|xN-3|

        Parameters:
        -----------
        :param:x : input array on which the roll will be performed
        :param i : delay value [sample]
        
        :type i :int
        :type x: N x 1 complex numpy array
        Return values:
        --------------
        :return shifted : shifted version of x
        :rtype shifted: N x 1 complex numpy array

    """
    N = x.shape[0]
    if np.abs(i) >= N:
        return np.zeros(N)
    if i == 0:
        return x
    shifted = np.roll(x, i)
    if i < 0:
        shifted[np.mod(N + i, N):] = np.zeros(np.abs(i))
    if i > 0:
        shifted[0:i] = np.zeros(np.abs(i))
    return shifted


def Wiener_SMI_MRE(ref_ch, surv_ch, K):
    """
        Description:
        ------------
            Performs Wiener filtering with applying the Minimum Redundance Estimation (MRE) technique. 
            When using MRE, the autocorrelation matrix is not fully estimated, but only the first column.
            With this modification the required calculations can be reduced from KxK to K element.
            
        Parameters:
        -----------
            :param K      : Filter tap number
            :param ref_ch : Reference signal array
            :param surv_ch: Surveillance signal array
            
            :type K      : int
            :type ref_ch : 1 x N complex numpy array
            :type surv_ch: 1 x N complex numpy array

        Return values:
        --------------
            :return filt: Filtered surveillance channel
            :rtype filt: 1 x N complex numpy array
            
            :return None: Input parameters are not consistent
    """
    # --- Input check ---
    if ref_ch.shape[0] != surv_ch.shape[0]:
        print("ERROR: No output is generated")
        return None

    N = ref_ch.shape[0]  # Number of time samples
    R = np.zeros((K, K), dtype=complex)  # Autocorrelation mtx.

    R[:, 0] = pruned_correlation(ref_ch, ref_ch, K)  # Calc. first column of the autocorr. matrix
    # ---t_R = 0.6sec running time, t_r is the same,K = 2048
    r = pruned_correlation(surv_ch, ref_ch, K)  # Cross-correlation vector

    # Complete the R matrix based on its Hermitian and Toeplitz property
    for k in range(1, K):
        R[:, k] = shift(R[:, 0], k)

    R += np.transpose(np.conjugate(R))
    R *= (np.ones(K) - np.eye(K) * 0.5)

    w = np.dot(LA.inv(R), r)  # weight vector
    # inverse and dot product run time : 1.1s for 2048*2048 matrix

    # output convolution lasts 1.61 sec K = 2048
    return surv_ch - np.convolve(ref_ch, w, mode='full')[0:N], w  # subtract the zero doppler clutter

def ECAS(ref_ch, surv_ch, subspace_list, T, Na):
    """

    Description:
    ------------
        Performs filtering on the surveillance channel in the time domain. The ECA-S algorithm is a modification
        of the ECA-B alg. where the filtering and the coefficient estimation is made on different block sizes.
        The window used for the coefficient estimation is the symmetrical extension of the window that is used for the
        filtering. This extension can be controlled via the Na parameter. Na is the one sided window extension
        measured in signal samples.
    Parameters:
    -----------
        :param ref_ch        : Signal component to suppress (reference channel)
        :param surv_ch       : Channel to filter (surveillance channel)
        :param subspace_list : Contains the time delay and Doppler shift which 
                               is used to span the signal subspace
        :param T             : Number of batches
        :param Na            : One sided window extension measured in samples

        :type ref_ch        : N by 1 complex numpy array
        :type surv_ch       : N by 1 complex numpy array
        :type subspace_list : 2D python list
        :type:T             : int
        :type Na            : int

    Return values:
    --------------
        :return filt_ch: Filtered surveillance channel
        :return w: Time domain filter coefficient vector
        
        :rtype N by 1 complex numpy array
        :rtype K by 1 complex numpy array

        :return None: General failure
    """
    # TODO: If the batch size is not multiple of the CPI the signals should be extend with zeros
    N = ref_ch.size
    batch_size = N // T

    # -- input check --
    if N % T:
        print("ERROR: Improper batch size, filtering is bypassed")
        print("ERROR: No output is generated")
        return None, None

    for k in subspace_list:
        if k[1] > batch_size:
            print("ERROR: Time delay is greater than the size of the batch, "
                  "this is not handled properly. Time delay is bypassed")
            print("ERROR: No output is generated")
            return None, None
        if k[1] > (batch_size + Na):
            print("ERROR: Time delay is greater than the size of the batch plus the sliding window, "
                  "this is not handled properly. Time delay is bypassed")
            print("ERROR: No output is generated")
            return None, None
    # -- calculation --
    KD = len(subspace_list)  # Signal subspace dimension

    extended_ref_ch = np.zeros(N + 2*Na, dtype=complex)
    extended_ref_ch[Na:Na + N] = ref_ch[:]
    extended_surv_ch = np.zeros(N + 2*Na, dtype=complex)
    extended_surv_ch[Na:Na + N] = surv_ch[:]

    filtered_surv_ch = np.zeros(N, dtype=complex)

    for l in np.arange(0, N, batch_size):  # Runs on batches
        surv_signal_batch = surv_ch[l: l + batch_size]  # Filtering will be performed on this block
        surv_signal_batch_ta = extended_surv_ch[l:l + batch_size + 2 * Na]  # "w" will be estimated on this block
        x = np.zeros((batch_size, KD), dtype=complex)  # Signal subspace for filtering
        x_ta = np.zeros((batch_size + 2*Na, KD), dtype=complex)  # Signal subspace for the estimation of the "w"

        # Generate matrices for filtering and estimation from the current batch
        subspace_vector_index = -1  # Used to index the subspace matrix (X) inside the for loops
        for k in subspace_list:  # Select current subspace vector parameter
            #print("Progress: %d %%" %(((subspace_vector_index + 2) / KD) * ((l + batch_size) / N) * 100))
            subspace_vector_index += 1

            doppler_shift_array_ta = np.exp(np.arange(l - Na, l + batch_size + Na, 1) * 1j * 2 * np.pi * k[0] / (2*N))
            doppler_shift_array = np.exp(np.arange(l, l + batch_size, 1) * 1j * 2 * np.pi * k[0] / (2*N))
            if not l:  # For the first batch zero values must be inserted (l=0)
                x[:, subspace_vector_index] = shift(ref_ch[l:l + batch_size], k[1])
                x_ta[:, subspace_vector_index] = shift(extended_ref_ch[l:l + batch_size + 2 * Na], k[1])
            else:
                x[:, subspace_vector_index] = ref_ch[l - k[1]:l + batch_size - k[1]]
                x_ta[:, subspace_vector_index] = extended_ref_ch[l - k[1]:l + 2 * Na + batch_size - k[1]]

            x[:, subspace_vector_index] *= doppler_shift_array  # Doppler shift
            x_ta[:, subspace_vector_index] *= doppler_shift_array_ta  # Doppler shift

        # Orthogonal subspace projection
        x = np.matrix(x)  # N x KD matrix
        x_ta = np.matrix(x_ta)  # N x KD matrix
        surv_signal = np.matrix(surv_signal_batch).getT()  # column vector
        surv_signal_ta = np.matrix(surv_signal_batch_ta).getT()  # column vector
        corr_matrix = x_ta.getH() * x_ta  # KD x KD
        corr_vector = x_ta.getH() * surv_signal_ta  # KD x 1
        weight_vector = corr_matrix.getI() * corr_vector  # coefficient vector
        filtered_surv_ch[l:l + batch_size] = np.array(surv_signal - x * weight_vector)[:, 0]

    return filtered_surv_ch

def gen_subspace_indexes(K, D):
    """
    Description:
    -----------
    Generate a signal subspace parameter list for the ECA algorithm variants.

    Parameters:
    -----------
        :param K: Number of considered time delay
        :param D: Maximum Doppler frequency value

        :type: K: int
        :type: D: int
    Return values:
    --------------
        :return: Generated subspace list
        :rtype: 2D python list
    """
    rd_index_list = []  # Used to store range-Doppler indexes on which the filtering have to performed
    # Create initial filter list
    for i in np.arange(-D, D + 1, 1):  # Doppler shifts
        for j in np.arange(0, K):  # Runs on signal delays
            rd_index_list.append([i , j])

    return rd_index_list

def temp_xcorr_vect_estimate(ref_signal, surv_signal, K, implementation="direct_for"):
    """
    Description:
    -----------
        Using this function the time domain cross-correlation vector can be estimated using different estimation methods.        
    
    Implementation notes:
    ---------------------
        implementation types:
            - direct_for    : Use a for loop to iterate through the signal samples
            - direct_matrix : Use large memory blocks to perform the required matrix operations
            - block_fft     : Partitionate the signal vector into smaller fractions, then calculates the cross-correlation
                             using FFT on each of them. 
        default: direct_for
    Parameters:
    -----------
        :param ref_signal: Signal vector of the reference channel
        :param surv_signal: Signal vector of the surveillance channel
        :param K: dimension
        :param implementation: "slow", "fast", "ultra_fast"
                
        :type ref_signal: N by 1 complex numpy array
        :type surv_signal: N by 1 complex numpy array
        :type K: int
        :type implementation: string
    
    Return values:
    --------------
        :return: cross correlation vector
        :rtype  (K x K complex numpy array)
        
        :return: None - Error, check input parameters!
    """
    # -- input check and prepare --
    if ref_signal.shape[0] != surv_signal.shape[0]:
        print("ERROR: The shape of the reference signal and surveillance signal does not match")
        print("ERROR: No output is generated")
        return None

    N = ref_signal.shape[0]  # Number of time samples
    r = np.zeros(K, dtype=complex)  # Cross-correlation vector allocation

    # -- calculation --
    if implementation == "direct_for":        
        for i in np.arange(K, N):
            r += (ref_signal[i - K:i][::-1]) * np.conjugate(surv_signal[i - 1])

    elif implementation == "direct_matrix":
        X = np.zeros((N, K), dtype=complex)  # Subspace matrix allocation
        # Fill up subspace matrix
        for k in range(K):
            X[:, k] = shift(ref_signal, k)
        
        r = np.dot(surv_signal.conj(), X)

    elif implementation == "block_fft":
        r = pruned_correlation(surv_signal, ref_signal, K)
    else:
        print("ERROR: Implementation type is not recognized:", implementation)
        print("ERROR: No output is generated!")
        return None

    r = np.divide(r, N)  # normalization
    return r


def temp_corr_mtx_estimate(ref_signal, K, implementation="direct_for"):
    """
    Description:
    -----------
        Using this function the time domain auto-correlation matrix can be estimated using different estimation methods.

    Implementation notes:
    ---------------------
            - direct_for    : Use a for loop to iterate through the signal samples
            - direct_matrix : Use large memory blocks to perform the required matrix operations
            - block_fft_mre : Partitionate the signal vector into smaller fractions, then calculates the cross-correlation
                              using FFT on each of them. With the MRE technique only K element is calculated.
        default: direct_for

     Parameters:
    -----------
        :param ref_signal: Signal vector of the reference channel        
        :param K: dimension
        :param implementation: "slow", "fast", "ultra_fast"
                
        :type ref_signal: N by 1 complex numpy array        
        :type K: int
        :type implementation: string
    
    Return values:
    --------------
        :return Estimated correlation matrix
        :rtype  (K x K complex numpy array)
        
        :return: None - Error, check input parameters!
    
    """
    # --- Input check and prepare ---
    N = ref_signal.shape[0]  # Number of time samples
    R = np.zeros((K, K), dtype=complex)  # Allocation
        
    if implementation == "direct_for":
        for i in np.arange(K, N):
            R += np.outer(ref_signal[i - K:i][::-1], np.conjugate(ref_signal[i - K:i][::-1]))

    elif implementation == "direct_matrix":
        X = np.zeros((N, K), dtype=complex) # Subspace matrix allocation
        # Fill up subspace matrix
        for k in range(K):  
            X[:, k] = shift(ref_signal, k)                   
        R = np.dot(X.conj().T, X)

    elif implementation == "block_fft":       
        R[:, 0] = pruned_correlation(ref_signal, ref_signal, K)  # Calc. first column of the autocorr. matrix       
    
        # Complete the R matrix based on its Hermitian and Toeplitz property
        for k in range(1, K):
            R[:, k] = shift(R[:, 0], k)
    
        R += np.transpose(np.conjugate(R))
        R *= (np.ones(K) - np.eye(K) * 0.5)

    else:
        print("ERROR: Not recognized implementation type")
        print("ERROR: No output is generated")
        return None
    
    R = np.divide(R, N)  # normalization
    return R


#----------------------------------------------#
#              UTILITY FUNCTIONS               #
#----------------------------------------------#

def change_to_log_scale(input_array, dynamic_range, data_type="power", normalize=False):

        if data_type == "power":
            multiplier = 10
        elif data_type == "voltage":
            multiplier = 20
        else:
            print("Unidentified dataType, handled as power")
            multiplier = 10
        array_abs = np.abs(input_array)
        if normalize:
            array_abs = np.divide(array_abs, np.max(array_abs))
        array_log = multiplier * np.log10(array_abs)

        maximum = np.max(array_log)
        for i in np.nditer(array_log, op_flags=['readwrite']):
            if i < maximum - dynamic_range:
                i[...] = maximum - dynamic_range

        return array_log
