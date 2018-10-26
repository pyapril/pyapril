# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import fft
from scipy import signal
try:
    import pyfftw        
    has_pyfftw = True
except ImportError as e:
    has_pyfftw = False
 
"""

                          Python based Advanced Passive Radar Library (pyAPRiL)

                                            detector


     Description:
     ------------
         This file contains the implementation of the cross-correlation based detectors.
         All functions realize the same detector but implemented using different signal
         processing schemes. 
         
         The implementations are listed bellow:
             - Direct, time domain implementation
             - Frequency domain implementation
             - Overlap and save method

         These functions are capable of sending Qt signals to perform progress display in Graphical User Interfaces.
         Check the APRIL package documentation for further information on the usage of this functionality.
        
     Demonstrations:
     ---------------
        
     Notes:
     ------------

        TODO: Write demonstration functions
        TODO: Use pyFFTW in the "cc_detector_ons" implementation also

     Dependencies:
     -------------
        - pyFFTw library is required for methods implemented in the frequency domain
        - The scipy package is also required for applying window functions.

     Features:
     ------------

     Project: pyAPRiL

     Authors: Tamás Pető, Dávid Huber

     License: GNU General Public License v3 (GPLv3)

     Changelog :
         - Ver 1.0000    : Initial version


     Version format: Ver X.ABCD
                         A - Complete code reorganization
                         B - Processing stage modification or implementation
                         C - Algorithm modification or implementation
                         D - Error correction, Status display modification

 """


def windowing(surv_ch, window_function):
        """
        Description:
        ------------

            Applies the specified window function on the surveillance channel. Using the appropriate window function
            the energy leakage of the reflected signal component in the Doppler domain can be avoided.
            To select the desired window function chose one from the followings:

                - "Rectangular"
                - "Flat top"
                - "Hamming"
                - "Hann"
                - "Tukey"
                - "Blackman"
                - "Blackman-Harris"

        Parameters:
        -----------
            :param: surv_ch: Surveillance channel
            :param: window_function: (string) Name of the applied window function
        Return values:
        --------------
        """
        if window_function == "Rectangular":
            window = np.ones(surv_ch.size)
        elif window_function == "Flat top":
            window = signal.flattop(surv_ch.size)
        elif window_function == "Hamming":
            window = signal.hamming(surv_ch.size)
        elif window_function == "Hann":
            window = signal.hann(surv_ch.size)
        elif window_function == "Tukey":
            window = signal.tukey(surv_ch.size)
        elif window_function == "Blackman":
            window = signal.blackman(surv_ch.size)
        elif window_function == "Blackman-Harris":
            window = signal.blackmanharris(surv_ch.size)
        else:
            print("Window function is not identified")
        return surv_ch * window


def cc_detector_td(ref_ch, surv_ch, fs, fD_max, fD_s, r_max, verbose=0, Qt_obj=None):
    """
        Description:
        ------------

            If Qt_obj is not specified no progress update messages will be sent. The verbose flag has no effect
            on the GUI progress update messages.
        
        Parameters:
        -----------
        
            :param: ref_ch: Reference channel
            :param: surv_ch: Surveillance channel
            :param: fs: Sampling frequency in Hz
            :param: fD_max: Maximum Doppler frequency
            :param: fD_s: Doppler frequency step size
            :param: r_max: Maximum range
            :param: verbose: Verbose mode, 0 means disabled
            :param: Qt_obj: This object is used to transfer progress update to a Qt based GUI
        Return values:
        --------------
    """
    N = np.size(ref_ch)
    # Allocating space for the Doppler shifted copy of the reference signal        
    ref_Doppler_shift = np.zeros(N, dtype=complex)
    Doppler_freqs = np.arange(-fD_max, fD_max + fD_s, fD_s)
    
    # Allocate range-Doppler maxtrix
    rd_matrix = np.zeros((Doppler_freqs.size, r_max), dtype=complex)
    
    for fD_index in range(Doppler_freqs.size):

        # Progress display update
        if verbose:
            print("Doppler frequency :", Doppler_freqs[fD_index])
            progress = int(100*fD_index / Doppler_freqs.size)
            print("Progress %d " % progress)
        if Qt_obj is not None:
            progress = int(100 * fD_index / Doppler_freqs.size)
            Qt_obj.progress_update.emit(progress)

        # Doppler shift
        for i in range(N):
            ref_Doppler_shift[i] = ref_ch[i] * np.exp(i * 1j * 2 * np.pi * Doppler_freqs[fD_index] / fs)
        
        # Range loop 
        for tau in range(r_max):
            rd_matrix[fD_index, tau] = np.dot(np.conjugate(shift(ref_Doppler_shift, tau)), surv_ch)
            
    return rd_matrix


def cc_detector_fd(ref_ch, surv_ch, fs, fD_max, r_max, verbose=0, Qt_obj=None):
    """
        Description:
        ------------
        
        Implementation notes:
        ---------------------
        
            The current implementation use 4 threads to calculate the inverse Fourier transform. Modify this parameter
            in case it does not fit to your hardware configuration.

        Parameters:
        -----------

            :param: ref_ch: Reference channel
            :param: surv_ch: Surveillance channel
            :param: fs: Sampling frequency in Hz
            :param: fD_max: Maximum Doppler frequency
            :param: r_max: Maximum range
            :param: verbose: Verbose mode, 0 means disabled
        Return values:
        --------------
    """

    # --> Set processing parameters
    N = np.size(ref_ch)  # Calculate the number of samples
    fD_step = fs / (2 * N)  # Doppler frequency step size (with zero padding)

    iFFT_in = pyfftw.empty_aligned(2 * N, dtype='complex128')
    iFFT_out = pyfftw.empty_aligned(2 * N, dtype='complex128')
    iFFT_object = pyfftw.FFTW(iFFT_in, iFFT_out, threads=4, direction="FFTW_BACKWARD", flags=("FFTW_MEASURE",))

    # --> CAF calculation
    # Number of Doppler frequencies
    Doppler_freqs_size = np.int16(fD_max / fD_step)

    # Allocate range-Doppler maxtrix
    rd_matrix = np.zeros((int(2 * Doppler_freqs_size + 1), int(r_max)), dtype=complex)

    # Zero padding
    surv_ch_ext = np.hstack((np.zeros(N), surv_ch))  # extend with zeros at the beginning of the time series
    ref_ch_ext = np.hstack((ref_ch, np.zeros(N)))  # extend with zeros at the end of the time series

    # Calculate the Fourier transforms of the channels
    surv_ch_ext_w = fft(surv_ch_ext)
    ref_ch_ext_w = fft(ref_ch_ext)

    for fd in np.arange(-Doppler_freqs_size, Doppler_freqs_size + 1, 1):

        # Progress display update
        if verbose:
            progress = (fd + Doppler_freqs_size) * 50 / Doppler_freqs_size
            # Emit progress here for GUI update
            print("Current Doppler frequency:  %.2f" % (fd * fD_step))
            print("Progress %.1f " % progress)
        if Qt_obj is not None:
            progress = (fd + Doppler_freqs_size) * 50 / Doppler_freqs_size
            Qt_obj.progress_update.emit(progress)

        # Perform Doppler shift with circular shift
        ref_ch_ext_w_shift = np.roll(ref_ch_ext_w, fd)

        # Cross correlation function in frequency domain
        iFFT_in[:] = surv_ch_ext_w * np.conjugate(ref_ch_ext_w_shift)

        # Calculate correlation function with inverse transformation
        iFFT_object()

        # Slice usefull range
        rd_matrix[int(fd + Doppler_freqs_size), :] = iFFT_out[N:N + int(r_max)]

    return rd_matrix


def cc_detector_ons(ref_ch, surv_ch, fs, fD_max, r_max, verbose=0, Qt_obj=None):
    """

    Parameters:
    -----------
        :param N: Range resolution - N must be a divisor of the input length
        :param F: Doppler resolution, F has a theoretical limit. If you break the limit, the output may repeat
                    itself and get wrong results. F should be less than length/N otherwise use other method!
    Return values:
    --------------
        :return None: Improper input parameters    
    
    """
    # --> Input check
    N = np.size(ref_ch)  # Calculate the number of samples
    if N % r_max !=0:
        print("ERROR: signal batch can not be partitioned with the specified range sample length")
        return None
    # --> Set processing parameters
    fD_step = fs / (2 * N)  # Doppler frequency step size (with zero padding)    
    Doppler_freqs_size = int(fD_max / fD_step)        
    no_sub_tasks = N // r_max    
    
    # Allocate range-Doppler maxtrix
    mx = np.zeros((2*Doppler_freqs_size+1, r_max),dtype = complex)

    
    ref_ch_align = np.reshape(ref_ch, (no_sub_tasks, r_max))  # shaping reference signal array into a matrix
    surv_ch_align = np.reshape(surv_ch,(no_sub_tasks, r_max))  # shaping surveillance signal array into a matrix
    surv_ch_align = np.vstack([surv_ch_align, np.zeros(r_max, dtype=complex)])  # padding one row of zeros into the surv matrix        
    ref_ch_align = np.concatenate([ref_ch_align, np.zeros((no_sub_tasks, r_max),dtype = complex)],axis = 1)  # shaping
    surv_ch_align = np.concatenate([surv_ch_align[0 : no_sub_tasks,:], surv_ch_align[1 : no_sub_tasks +1, :]], axis = 1)   
  
    # row wise fft on both channels
    ref_fft = np.fft.fft(ref_ch_align, axis = 1)
    surv_fft = np.fft.fft(surv_ch_align, axis = 1)  
    
    # correlation in frequency domain 
    corr = np.multiply(surv_fft, ref_fft.conj())                                           
    corr = np.fft.ifft(corr,axis = 1)
    
    # Doppler FFT
    corr = np.concatenate([corr, np.zeros((no_sub_tasks, 2*r_max), dtype = complex)],axis = 0) # Prepare with zero padding
    corr = np.fft.fft(corr,axis = 0)  # column wise fft
    
    # crop and fft shift
    mx[ 0 : Doppler_freqs_size, 0 : r_max] = corr[2*no_sub_tasks - Doppler_freqs_size : 2*no_sub_tasks, 0 : r_max] 
    mx[Doppler_freqs_size : 2 * Doppler_freqs_size+1, 0 : r_max] = corr[ 0 : Doppler_freqs_size+1 , 0 : r_max]
    
    return mx

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
