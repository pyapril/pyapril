# -*- coding: utf-8 -*-
import sys
import os
currentPath = os.path.dirname(os.path.realpath(__file__))
april_path = os.path.join(os.path.dirname(currentPath), "pyapril")
sys.path.insert(0, april_path)

import numpy as np
import metricExtract as me

#
#  Single file clutter filter performance evaluation on raw data
#

# TODO: Clean up and document this testing script

iq = np.load('_raw_iq/VEGAM20190729FOXC0S0_755.npy')
target_rd=[9,410]
win=[6,6,3,3]
win_pos=[100,200]

target_rd[1] -=262 # Doppler cell centralization
win_pos[1] -= (525-1)//2 # Doppler cell centralization


ref_ch = iq[3,:]
surv_ch = iq [6,:]
metrics = me.eval_clutter_filter_raw(ref_ch, 
                                  surv_ch,                                  
                                  "SMI-MRE",
                                  K=64,
                                  target_rd= target_rd, 
                                  win=win, 
                                  win_pos=win_pos)
for (metric, value) in metrics.items():
        print("{:s} : {:.2f} dB".format(metric,value))





