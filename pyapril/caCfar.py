# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal


class CA_CFAR():
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
        Implementation based on https://github.com/petotamas/APRiL
    Parameters:
    -----------
    :param win_param: Parameters of the noise power estimation window
                      [Est. window length, Est. window width, Guard window length, Guard window width]
    :param threshold: Threshold level above the estimated average noise power
    :type win_param: python list with 4 elements
    :type threshold: float
    Return values:
    --------------
    """

    def __init__(self, win_param, threshold, rd_size):
        win_width = win_param[0]
        win_height = win_param[1]
        guard_width = win_param[2]
        guard_height = win_param[3]

        # Create window mask with guard cells
        self.mask = np.ones((2 * win_height + 1, 2 * win_width + 1), dtype=bool)
        self.mask[win_height - guard_height:win_height + 1 + guard_height, win_width - guard_width:win_width + 1 + guard_width] = 0

        # Convert threshold value
        self.threshold = 10 ** (threshold / 10)

        # Number cells within window around CUT; used for averaging operation.
        self.num_valid_cells_in_window = signal.convolve2d(np.ones(rd_size, dtype=float), self.mask, mode='same')

    def __call__(self, rd_matrix):
        """
        Description:
        ------------
            Performs the automatic detection on the input range-Doppler matrix.

        Implementation notes:
        ---------------------
        Parameters:
        -----------
        :param rd_matrix: Range-Doppler map on which the automatic detection should be performed
        :type rd_matrix: R x D complex numpy array
        Return values:
        --------------
        :return hit_matrix: Calculated hit matrix
        """
        # Convert range-Doppler map values to power
        rd_matrix = np.abs(rd_matrix) ** 2

        # Perform detection
        rd_windowed_sum = signal.convolve2d(rd_matrix, self.mask, mode='same')
        rd_avg_noise_power = rd_windowed_sum / self.num_valid_cells_in_window
        rd_snr = rd_matrix / rd_avg_noise_power
        hit_matrix = rd_snr > self.threshold

        return hit_matrix, rd_snr
