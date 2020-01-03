# -*- coding: utf-8 -*-
import numpy as np
from pyargus import directionEstimation as de
"""

                             Python based Advanced Passive Radar Library (pyAPRiL)

                                          Hit Processor Module


     Description:
     ------------
         Contains the implementation of the most common hit processing algorithms.
                  
             - CA-CFAR processor: Implements an automatic detection with (Cell Averaging - Constant False Alarm Rate) detection.
             - Target DOA estimator: Estimates direction of arrival for the target reflection from the range-Doppler
                                     maps of the surveillance channels using phased array techniques.
        
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
         - Ver 1.1.0    : Target DOA estimation (2019 04 11)

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
        of the target reflection energy spreading some cells are left out from the calculation in the immediate
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


def target_DOA_estimation(rd_maps, hit_list, DOA_method, array_alignment):
    """
        Performs DOA (Direction of Arrival) estimation for the given hits. To speed up the calculation for multiple
        hits this function requires the calculated range-Doppler maps from all the surveillance channels.

    Parameters:
    -----------
        :param: rd_maps: range-Doppler matrices from which the azimuth vector can be extracted
        :param: hit_list: Contains the delay and Doppler coordinates of the targets.
        :param: DOA_method: Name of the required algorithm to use for the estimation
        :param: array_alignment: One dimensional array, which describes the active antenna positions

        :type : rd_maps: complex valued numpy array with the size of  Μ x D x R , where R is equal to
                                the number of range cells, and D denotes the number of Doppler cells.
        :type: hit_list: Python list [[delay1, Doppler1],[delay2, Doppler2]...].
        :type: DOA_method: string
        :type: array_alignment: real valued numpy array with size of 1 x M, where M is the number of
                            surveillance antenna channels.

    Return values:
    --------------
        target_doa : Measured incident angles of the targets

    TODO: Extend with decorrelation support
    """
    doa_list = []  # This list will contains the measured DOA values

    # Generate scanning vectors
    thetas = np.arange(0, 180, 1)
    scanning_vectors = de.gen_ula_scanning_vectors(array_alignment, thetas)

    for hit in hit_list:
        azimuth_vector = rd_maps[:, hit[1], hit[0]]
        R = np.outer(azimuth_vector, azimuth_vector.conj())
        if DOA_method == "Fourier":
            doa_res = de.DOA_Bartlett(R, scanning_vectors)
        elif DOA_method == "Capon":
            doa_res = de.DOA_Capon(R, scanning_vectors)
        elif DOA_method == "MEM":
            doa_res = de.DOA_MEM(R, scanning_vectors, column_select=0)
        elif DOA_method == "MUSIC":
            doa_res = de.DOA_MUSIC(R, scanning_vectors, signal_dimension=1)

        hit_doa = thetas[np.argmax(doa_res)]
        doa_list.append(hit_doa)
        print("Estimating DOA for hit-> r:{:d} D:{:.2f} Az:{:.2f} ".format(hit[0], hit[1], hit_doa))
    return doa_list

