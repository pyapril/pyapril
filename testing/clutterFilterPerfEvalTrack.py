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

"""
# This script is used to convert the track data
track = np.load("VEGAM20180313C1S0FM_track_tr0.npy")[:,0:2].astype(dtype=int)
# Centralizing Doppler coordinates
for p in range(track.shape[0]):
    if track[p, 1] > -1:    
        #track[p, 1] -= (525-1)//2
        track[p, 1] -= (131-1)//2
"""

# Load target track file
#track = np.load("VEGAM20190729FOXC0S0FM_SurvP5_track.npy")[:,0:2].astype(dtype=int)
track = np.load("VEGAM20180313C1S0FM_track_tr0.npy")

win=[5,5,3,3]
win_pos=[50,-45]
#filename_temp = "_raw_iq/VEGAM20190729FOXC0S0_"
filename_temp = "_raw_iq/VEGAM20180313HR2C1S0FM/VEGAM20180313HR2U0C1S0FM_"

#surv_ch_ind, ref_ch_ind
metric_array = me.eval_clutter_filter_on_track_raw(iq_fname_temp=filename_temp,
                                                start_ind=77, 
                                                stop_ind=220,
                                                filter_method="SMI-MRE",
                                                ref_ch_ind=0,
                                                surv_ch_ind=1,
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





