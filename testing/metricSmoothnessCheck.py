# -*- coding: utf-8 -*-
import sys
import os
currentPath = os.path.dirname(os.path.realpath(__file__))
april_path = os.path.join(os.path.dirname(currentPath), "pyapril")
sys.path.insert(0, april_path)

import numpy as np
from matplotlib import pyplot as plt    
import metricExtract as me

#
#  Script used to check the smoothness of the extracted metric array, with 
#  increasing number of records
#

# Load target track file
track = np.load("VEGAM20190729FOXC0S0FM_SurvP5_track.npy")[:,0:2].astype(dtype=int)

# Centralizing Doppler coordinates
for p in range(track.shape[0]):
    if track[p, 1] > -1:    
        track[p, 1] -= (525-1)//2

win=[5,5,3,3]
win_pos=[50,200]
win_pos[1] -= (525-1)//2
filename_temp = "_raw_iq/VEGAM20190729FOXC0S0_"
statistics = np.zeros((10, 24))  # Number of metrics, number of time records
for offset in range(24):
    print("Offset: {:d}".format(offset))
    metric_array = me.eval_clutter_filter_on_track_raw(iq_fname_temp=filename_temp,
                                                    start_ind=750, 
                                                    stop_ind=750+offset+2,
                                                    filter_method="SMI-MRE",
                                                    target_rds=track,
                                                    K=64,
                                                    win=win,
                                                    win_pos=win_pos,
                                                    rd_size=[64, 300],
                                                    rd_windowing="Hann",
                                                    max_clutter_delay=64)
    
    
    for m in range(metric_array.shape[0]-1):                
        statistics [m, offset] = 20*np.log10(np.median(10**(metric_array[m+1, :]/20)))        
      
for m in range(metric_array.shape[0]-1):
    plt.plot(np.arange(2,2+24),statistics[m,:] - np.max(statistics[m,:]))
    plt.legend(['CA','R_nf','Mu imp','Delta imp','Alpha imp','L','Rdpi','Rzdc','P','D'])
    #plt.legend(['CA','R_nf','Rdpi','Rzdc','P','D']) # Metrics without target coord knowledge
    plt.xlabel("Number of records")
    plt.ylabel("Normalized median [dB]")
    plt.grid(True)
    
