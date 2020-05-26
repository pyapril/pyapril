# -*- coding: utf-8 -*-
import numpy as np
from pyargus import directionEstimation as de

"""

                             Python based Advanced Passive Radar Library (pyAPRiL)

                                          Hit Processor Module


     Description:
     ------------
         Contains the implementation of the most common hit processing algorithms.

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
        #print("Estimating DOA for hit-> r:{:d} D:{:.2f} Az:{:.2f} ".format(hit[0], hit[1], hit_doa))
    return doa_list
