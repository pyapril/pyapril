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

iq = np.load('_raw_iq/VEGAM20190729FOXC0S0_755.npy')
ref_ch = iq[3,:]
surv_ch = iq [6,:]
surv_ch = chprep.time_domain_filter_surveillance(ref_ch, surv_ch, "SMI-MRE", K=64)
rd_matrix = detector.cc_detector_ons(ref_ch, surv_ch, 200*10**3, 100, 128, verbose=0, Qt_obj=None)
#export_rd_matrix_img(fname='rd_test2.png', rd_matrix=rd_matrix, max_Doppler=100, dyn_range=20, dpi=800)


target_rd=[9,410]
win=[5,5,3,3]
win_pos=[100,200]
# Centralizing Doppler coordinates
target_rd[1] -= (525-1)//2
win_pos[1] -= (525-1)//2

nf    = me.extract_noise_floor(rd_matrix, win, win_pos)
delta = me.extract_delta(rd_matrix, target_rd, win)
mu    = me.extract_mu(rd_matrix, target_rd, surv_ch)
print("Noise floor: {:.2f} dB".format(nf))
print("Delta: {:.2f} dB".format(delta))
print("Mu: {:.2f} dB".format(mu))

