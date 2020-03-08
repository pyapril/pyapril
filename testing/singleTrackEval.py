# -*- coding: utf-8 -*-
import sys
import os
currentPath = os.path.dirname(os.path.realpath(__file__))
april_path = os.path.join(os.path.dirname(currentPath), "pyapril")
sys.path.insert(0, april_path)

import numpy as np
import matplotlib.pyplot as plt
import metricExtract as me

# TODO: Clean up and document this testing script

# Load target track file
track = np.load("VEGAM20190729FOXC0S0FM_SurvP5_track.npy")[:,0:2].astype(dtype=int)
rdat = np.load("VEGAM20190729FOXC0S0FM_SurvP5_RDAT.npy")    
# Centralizing Doppler coordinates
for p in range(track.shape[0]):
    if track[p, 1] > -1:    
        track[p, 1] -= (525-1)//2

win=[5,5,3,3]
win_pos=[100,200]
win_pos[1] -= (525-1)//2


metric_array = me.eval_metrics_on_track(track, rdat[:,0,:,:], win=win, win_pos=win_pos)
valid_cpi_indices = metric_array[:,0]
deltas = metric_array[:,1]
alphas = metric_array[:,2]
#mus = = metric_array[:, 3]
plt.plot(valid_cpi_indices, alphas)
plt.plot(valid_cpi_indices, deltas)
plt.legend(["Alpha", "Delta"])
plt.grid(True)
print("Average Delta metric: {:.2f}".format(np.average(deltas)))
print("Average Alpha metric: {:.2f}".format(np.average(alphas)))



