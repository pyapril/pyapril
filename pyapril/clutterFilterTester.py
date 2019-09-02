# -*- coding: utf-8 -*-
from RDTools import export_rd_matrix_img
import detector
import channelPreparation as chprep
import numpy as np

iq = np.load('VEGAM20190729FOXC0S0_755.npy')
ref_ch = iq[3,:]
surv_ch = iq [6,:]
surv_ch = chprep.time_domain_filter_surveillance(ref_ch, surv_ch, "ECA", K=64, D=2)
rd_matrix = detector.cc_detector_ons(ref_ch, surv_ch, 200*10**3, 100, 128, verbose=0, Qt_obj=None)
export_rd_matrix_img(fname='rd_test.png', rd_matrix=rd_matrix, max_Doppler=100, dyn_range=40, dpi=800)

