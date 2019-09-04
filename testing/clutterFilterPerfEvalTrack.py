# -*- coding: utf-8 -*-
import sys
import os
currentPath = os.path.dirname(os.path.realpath(__file__))
april_path = os.path.join(os.path.dirname(currentPath), "pyapril")
sys.path.insert(0, april_path)

import numpy as np
import metricExtract as me

#
#  Clutter filter performance evaluation on full track from raw data
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

metric_array = me.eval_clutter_filter_on_track_raw(iq_fname_temp=filename_temp,
                                                start_ind=750, 
                                                stop_ind=776,
                                                filter_method="SMI-MRE",
                                                target_rds=track,
                                                K=64,
                                                win=win,
                                                win_pos=win_pos,
                                                rd_windowing="Hann",
                                                max_clutter_delay=64)


from matplotlib import pyplot as plt          
plt.plot(metric_array[0,:], metric_array[1,:]) # CA 
plt.plot(metric_array[0,:], metric_array[2,:]) # Rnf
plt.plot(metric_array[0,:], metric_array[3,:]) # Mu_imp
plt.plot(metric_array[0,:], metric_array[4,:]) # Delta_imp
plt.plot(metric_array[0,:], metric_array[5,:]) # Alpha_imp
plt.plot(metric_array[0,:], metric_array[6,:]) # L
plt.plot(metric_array[0,:], metric_array[7,:]) # R_dpi
plt.plot(metric_array[0,:], metric_array[8,:]) # R_zdc
plt.plot(metric_array[0,:], metric_array[9,:]) # P
plt.plot(metric_array[0,:], metric_array[10,:]) # D
plt.legend(['CA','R_nf','Mu imp','Delta imp','Alpha imp','L','Rdpi','Rzdc','P','D'])
plt.grid(True)





