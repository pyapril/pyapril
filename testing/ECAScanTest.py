# -*- coding: utf-8 -*-
import sys
import os
currentPath = os.path.dirname(os.path.realpath(__file__))
april_path = os.path.join(os.path.dirname(currentPath), "pyapril")
sys.path.insert(0, april_path)

import numpy as np
from matplotlib import pyplot as plt    
import metricExtract as me

"""
#
#  Clutter filter parameter scan
#  Test: ECA scan

# Load target track file
#track = np.load("VEGAM20190729FOXC0S0FM_SurvP5_track.npy")[:,0:2].astype(dtype=int)
track = np.load("VEGAM20180313C1S0FM_track_tr0.npy")

win=[5,5,3,3]
win_pos=[50,-45]
filename_temp = "_raw_iq/VEGAM20190729FOXC0S0_"
filename_temp = "_raw_iq/VEGAM20180313HR2C1S0FM/VEGAM20180313HR2U0C1S0FM_"
scan_res = me.scan_ECA(statistic='avg',
                      time_range=[2,64,1],
                      doppler_range=[0,10,1],
                      iq_fname_temp=filename_temp,
                      start_ind=77, 
                      stop_ind=107,
                      filter_method="ECA",
                      target_rds=track, 
                      win=win,
                      win_pos=win_pos,
                      rd_size=[128, 300],
                      rd_windowing="Hann",
                      max_clutter_delay=128)
"""
eca_scan_fig, eca_scan_axes = plt.subplots()
eca_scan_plot = eca_scan_axes.imshow(scan_res[0:32,:,9], interpolation='sinc', cmap='nipy_spectral', origin='lower', aspect='auto')
eca_scan_fig.colorbar(eca_scan_plot)
plt.xlabel('Doppler domain [bin]')
plt.ylabel('Time domain [tap]')
plt.title('ECA - Parameter Scan, metric: D')
plt.xlim([0,9])
#plt.legend(['CA','R_nf','Mu imp','Delta imp','Alpha imp','L','Rdpi','Rzdc','P','D'])
    
