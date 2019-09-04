# -*- coding: utf-8 -*-

import sys
import os
currentPath = os.path.dirname(os.path.realpath(__file__))
april_path = os.path.join(os.path.dirname(currentPath), "pyapril")
sys.path.insert(0, april_path)

from RDTools import export_rd_matrix_img
import detector
import channelPreparation as chprep
import numpy as np
import metricExtract as me


# Single-Matrix clutter filter performance evaluation test
# Target 1 [9,410]
# Target 2 [106, 389]

iq = np.load('_raw_iq/VEGAM20190729FOXC0S0_755.npy')
ref_ch = iq[3,:]
surv_ch = iq [6,:]
surv_ch_filt = chprep.time_domain_filter_surveillance(ref_ch, surv_ch, "SMI-MRE", K=64)
rd_matrix = detector.cc_detector_ons(ref_ch, surv_ch, 200*10**3, 100, 128, verbose=0, Qt_obj=None)
rd_matrix_filt = detector.cc_detector_ons(ref_ch, surv_ch_filt, 200*10**3, 100, 128, verbose=0, Qt_obj=None)
#export_rd_matrix_img(fname='rd_test.png', rd_matrix=rd_matrix, max_Doppler=100, dyn_range=20, dpi=800)


win=[6,6,3,3]
win_pos=[100,200]
target_rd=[9,410]
# Centralizing Doppler coordinates
target_rd[1] -= (525-1)//2
win_pos[1] -= (525-1)//2

metrics = me.eval_clutter_filt_perf(surv_ch=surv_ch, surv_ch_filt=surv_ch_filt,
                                 rd_matrix=rd_matrix, rd_matrix_filt=rd_matrix_filt,
                                 target_rd=target_rd, win=win, win_pos=win_pos)
for (metric, value) in metrics.items():
    print("{:s} : {:.2f} dB".format(metric,value))



