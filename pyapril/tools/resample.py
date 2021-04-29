# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt # For demonstrations
import scipy.signal as signal

def resample (resample_ratio, signal):
    """
        Resamples the input signal with the given resample ratio. If the 
        input signal has "N" samples, the output signal will have "N*R"
        samples.
        
    Parameters:
    -----------
    
        - resample_ratio : rational number
        - signal : (complex 1D numpy array)
    
    Return values:
    --------------
    
        - resampled_signal : (complex 1D numpy array)                      
    """
    
    N =  np.size(signal)
    N_new = int(N * resample_ratio)
    
    resampled_signal = np.zeros(N_new, dtype=complex)
    
    for i in range(N_new):        
        j = (i+1) * (1/resample_ratio)#  original signal index        
        if j > int(j):
            j = int(j + 1)
        else:
            j = int(j)
        
        #print("i: %d j: %d"%(i+1,j))
        resampled_signal[i] = signal[j-1]
        
    return resampled_signal

def resample_n_filt (n, d , no_taps, x):
    """
        Resamples the input signal with the given resample ratio. If the 
        input signal has "N" samples, the output signal will have "int(N*R)"
        samples.
        
        The resampling ratio is R = n/d
        
        
    Implementation notes:
    ---------------------
        The resampling is performed in three stage. The signal is first 
        upsampled with "n" then a low pass filter is applied on the uppsampled
        signal to filter aliasing frequencies during the decimation stage. The
        low pass filter tap size can be specified using the "no_taps" input 
        parameter. After the signal has been filtered it is downsampled 
        (decimated) with "d". 
        
        The resampling process may consume a lot of memory as the upsampled 
        signal is stored before the filtering and the decimation.
        
        The anti-alliasing low pass filter uses "Hann" window
        
        The up and downsampling is done with zero order hold, thus the signal
        samples are repeated (up-sampling) and discarded (down-sampling).

    
    Parameters:
    -----------
    
        - n : (int) resample ratio nominator
        - d : (int) resample ratio denominator
        - tap_size: (int) low pass filter tap number
        - x : (complex 1D numpy array) input signal
    
    Return values:
    --------------
    
        - x_resampled : (complex 1D numpy array)                      
    """
    
    N =  np.size(x)    
    
    # Upsample    
    x_upsampled = np.zeros(N*n, dtype=complex)
    for i in range(N):        
        x_upsampled[i*n:(i+1)*n] = x[i]
        
    # Filter
    cut_off = 1/np.max((n,d))
    filter_coeffs = signal.firwin(no_taps, cut_off, window="hann")   
    x_filtered = np.convolve (filter_coeffs, x_upsampled, mode = "same")
    
    
    # Downsample
    x_resampled = np.zeros(int(N*n/d), dtype=complex)                 
    for k in np.arange(0, int(N*n/d), 1):
        x_resampled[k] = x_filtered[int(k*d)]    
    return x_resampled 


"""
*******************************************************************************
                           D E M O N S T R A T I O N
*******************************************************************************
"""
def demo():

    f  = 10 *10**6 
    fs = 60 *10**6
    N  = 600
    t = np.arange(N)
    n = 2
    d = 1
    R = n/d
    
    x = np.sin(2*np.pi*f/fs *t )
    
    #plt.plot(x) 
    
    xw = np.fft.fft(x)
    
    freq = fs/N * np.arange(int(-N/2),int(N/2),1) / 10**6
    xw_rev = np.concatenate((xw[int(N/2):N] , xw[0:int(N/2)]))
    plt.plot(freq,20*np.log10(np.abs(xw_rev)))
    
    # Resampling
    #fs_in = 60*10**6
    #fs_out = 90 *
    
    x_resampled=resample(resample_ratio=R, signal = x)   
    
    
    xw2 = 20*np.log10(np.abs(np.fft.fft(x_resampled)))
    freq2 = fs/N * np.arange(int(-N*R/2),int(N*R/2),1) / 10**6
    xw2_rev = np.concatenate((xw2[int(N*R/2):int(N*R)] , xw2[0:int(N*R/2)]))
    plt.plot(freq2, xw2_rev)
      
    plt.axvline(x=fs/2/10**6, linewidth=2, color="red")
    plt.axvline(x=-fs/2/10**6, linewidth=2, color="red")
    
    x_resampled=resample_n_filt(n, d, 100, x)
    
    xw2 = 20*np.log10(np.abs(np.fft.fft(x_resampled)))
    freq2 = fs/N * np.arange(int(-N*R/2),int(N*R/2),1) / 10**6
    xw2_rev = np.concatenate((xw2[int(N*R/2):int(N*R)] , xw2[0:int(N*R/2)]))
    plt.plot(freq2, xw2_rev)
    
    #plt.ylim((-10 ,40))
    
#demo()  
#plt.figure(3)
#plt.plot(x)
#plt.plot(x2)
