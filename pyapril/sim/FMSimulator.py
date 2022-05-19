import numpy as np
import math
import sys
from pyapril.tools import resample

from pydub import AudioSegment
from pydub.utils import mediainfo

"""
                          Python based Advanced Passive Radar Library (pyAPRiL)

                                            FM IoO Simulator
"""

def generate_fm_signal(s_mod, fs): 
    """
        Generates FM signal using the give modulation signal
        
        Parameters:
        -----------
        :param: s_mod: Modulating signal
        :param:    fs: Sampling frequency of the modulating signal [Hz]
        
        :type: s_mod: One dimensional numpy array
        :type:    fs: float
        
        Retrun values:
        --------------
        :return: s : Generated FM signal
        :return: fs: Sampling frequency of the modulated signal
        
        :rtype:   s: One dimensional numpy array
        :rtype:  fs: float 
    
    """
    
    # Internal processing parameters
    fd             = 75 # frequency deviation [kHz] -> Same as in FM broadcast

    # Generate FM signal
    s_mod = s_mod/np.max(np.abs(s_mod)) # Normalize
    k_fm = fd*10**3/np.max(s_mod)    
    s = np.sin(2*np.pi*k_fm/fs*np.cumsum(s_mod)) # Modulate
    return s, fs

def generate_fm_signal_from_sound(sound_fname, T_sim, offset): 
    """
    Generated FM signal from sound file
    
    The offset parameter is a float number between 0 and 1. It
    specifyies the start position of the time window(T_sim) 
    that will be used from the sound file for the FM signal 
    preparation.
    
    Parameters:
    ----------
    :param: sound_fname: Name of sound sound file to be imported 
    :param:       T_sim: Duration of the reqested simulated signal [s]
    :param:      offset: Offset position of the processed window
    
    :type: sound_fname: string
    :type:      T_sim : float
    :type:      offset: float (0..1)
    
    Retrun values:
    --------------
    :return: s : Generated FM signal
    :return: fs: Sampling frequency of the modulated signal
    
    :rtype:   s: One dimensional numpy array
    :rtype:  fs: float 
    
    """
    # Internal processing parameters
    resample_ratio = 5 
    fir_tap_size   = 10 # Resample filter tap size
    fd             = 75 # frequency deviation [kHz] -> Same as in FM broadcast
    
    # Import sound file
    sound = AudioSegment.from_mp3(sound_fname)
    sound_samples = np.array(sound.get_array_of_samples())
    
    info     = mediainfo(sound_fname)
    fs = int(info['sample_rate'])
    
    # Resample    
    offset = int(offset*np.size(sound_samples))
    N_raw = math.ceil(T_sim/(1/fs)) # Number of samples needed from the modulating signal
    sound_samples = sound_samples[offset:offset+N_raw] # Cut out useful portion
    s_mod = resample.resample_n_filt(resample_ratio,1,fir_tap_size,sound_samples)
    fs = resample_ratio * fs 
  
    # Generate FM signal
    s_mod = s_mod/np.max(np.abs(s_mod)) # Normalize
    k_fm = fd*10**3/np.max(s_mod)
    s = np.sin(2*np.pi*k_fm/fs*np.cumsum(s_mod)) # Modulate
    return s, fs