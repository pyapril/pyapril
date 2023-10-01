# -*- coding: utf-8 -*-
import numpy as np
from pyapril.targetLocalization import localize_target_ms_sx

"""
Tests the multistatic target localization

Project: pyAPRIL
Authors: Tamás Pető
License: GNU General Public License v3 (GPLv3)
"""    
def calculate_bistatic_range(ioo_coords, radar_coords, target_coords):
    # Baseline distance
    L = np.sqrt(np.sum(np.abs(radar_coords-ioo_coords)**2)) 
    
     # Target to IoO distance
    Rt = np.sqrt(np.sum(np.abs(target_coords-ioo_coords)**2)) 
    
    # Target to radar distance
    Rr = np.sqrt(np.sum(np.abs(target_coords-radar_coords)**2)) 
    
    # Bistatic distance
    Rb = Rt+Rr-L

    return Rb

"""
PARAMETERS
"""
ioo_coords = np.array([[0.0, -100.0, -500.0],
                      [500.0, 500.0, 2000.0],
                      [-2000.0, 2000.0, 0.0]])


target_coords = np.array([0.0, 1000.0, 0.0])

"""
TEST
"""

rb_vec =  np.array([calculate_bistatic_range(ioo_coords[i,:], np.array([0,0,0]), target_coords) for i in range(ioo_coords.shape[0])])
x1,x2 = localize_target_ms_sx(ioo_coords, rb_vec)

# Calculate errors
err = np.max(abs(np.array([x1,x2]) -target_coords), axis=1)

assert min(err) < 10**-9, "Calculated target coordinates is out of tolerance"
print("All good")

