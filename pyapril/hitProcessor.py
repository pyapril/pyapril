# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
"""

                             Python based Advanced Passive Radar Library (pyAPRiL)

                                          Hit Processor Module


     Description:
     ------------
         Contains the implementation of the most common hit processing algorithms.
                  
             - CA-CFAR processor: Implements an automatic detection with (Cell Averaging - Constant False Alarm Rate) detection.             
        
     Notes:
     ------------

     Features:
     ------------

     Project: pyAPRIL

     Authors: Tamás Pető

     License: GNU General Public License v3 (GPLv3)

     Changelog :
         - Ver 1.0.0    : Initial version (2017 11 02)
         - Ver 1.0.1    : Faster CFAR implementation(2019 02 15)

 """

def CA_CFAR(rd_matrix, win_param, threshold):
    """
    Description:
    ------------
        Cell Averaging - Constant False Alarm Rate algorithm
        
        Performs an automatic detection on the input range-Doppler matrix with an adaptive thresholding.
        The threshold level is determined for each cell in the range-Doppler map with the estimation
        of the power level of its surrounding noise. The average power of the noise is estimated on a 
        rectangular window, that is defined around the CUT (Cell Under Test). In order the mitigate the effect
        of the target reflecion energy spreading some cells are left out from the calculation in the inmediate
        vicinity of the CUT. These cells are the guard cells.
        The size of the estimation window and guard window can be set with the win_param parameter. 
    
    Implementation notes:
    ---------------------

    Parameters:
    -----------

    :param rd_matrix: Range-Doppler map on which the automatic detection should be performed
    :param win_param: Parameters of the noise power estimation window 
                      [Est. window length, Est. window width, Guard window length, Guard window width]
    :param threshold: Threshold level above the estimated average noise power    

    :type rd_matrix: R x D complex numpy array
    :type win_param: python list with 4 elements
    :type threshold: float

    Return values:
    --------------

    :return hit_matrix: Calculated hit matrix

    """
    
    # -- Set inital parameters --
    win_len = win_param[0]
    win_width = win_param[1]
    guard_len = win_param[2]
    guard_width = win_param[3]

    norc = np.size(rd_matrix, 1)  # number of range cells
    noDc = np.size(rd_matrix, 0)  # number of Doppler cells
    hit_matrix = np.zeros((noDc, norc), dtype=float)
    
    # Convert range-Doppler map values to power
    rd_matrix = np.abs(rd_matrix) ** 2
        
    # Generate window mask
    rd_block = np.zeros((2 * win_width + 1, 2 * win_len + 1), dtype=float)
    mask = np.ones((2 * win_width + 1, 2 * win_len + 1))    
    mask[win_width - guard_width:win_width + 1 + guard_width, win_len - guard_len:win_len + 1 + guard_len] = np.zeros(
        (guard_width * 2 + 1, guard_len * 2 + 1))

    cell_counter = np.sum(mask)

    # Convert threshold value
    threshold = 10 ** (threshold / 10)
    threshold /= cell_counter
    
    # -- Perform automatic detection --
    for j in np.arange(win_width, noDc - win_width, 1):  # Range loop
        for i in np.arange(win_len, norc - win_len, 1):  # Doppler loop
            rd_block = rd_matrix[j - win_width:j + win_width + 1, i - win_len:i + win_len + 1]
            rd_block = np.multiply(rd_block, mask)
            cell_SINR = rd_matrix[j, i] / np.sum(rd_block) # esimtate CUT SINR
            
            # Hard decision
            if cell_SINR > threshold:
                hit_matrix[j, i] = 1
                
    return hit_matrix


