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
#  Clutter filter parameter scan
#  Test: Wiener-SMI-MRE dimension

# Load target track file
#track = np.load("VEGAM20190729FOXC0S0FM_SurvP5_track.npy")[:,0:2].astype(dtype=int)
track = np.load("VEGAM20180313C1S0FM_track_tr0.npy")


win=[5,5,3,3]
win_pos=[50,-45]

#filename_temp = "_raw_iq/VEGAM20190729FOXC0S0_"
filename_temp = "_raw_iq/VEGAM20180313HR2C1S0FM/VEGAM20180313HR2U0C1S0FM_"
scan_res = me.time_domain_dim_stability(statistic='avg',
                                      dim_list=np.arange(2,64,1).tolist(),                                      
                                      iq_fname_temp=filename_temp,
                                      start_ind=77, 
                                      stop_ind=80,
                                      filter_method="SMI-MRE",
                                      iqrec_win_size=2,
                                      target_rds=track, 
                                      win=win,
                                      win_pos=win_pos,
                                      rd_size=[128, 300],
                                      rd_windowing="Hann",
                                      max_clutter_delay=128)

