# - coding: utf-8 -*-
import numpy as np
"""

                             Python based Advanced Passive Radar Library (pyAPRiL)

                                               Channel preparation


     Description:
     ------------

        This file is intended to collect and wrap all the passive radar channel preparation related functions.
        These functions isolate the reference and surveillance channels and perform the necessary filtering steps
        before the detection. The main aim of this functions set is to collect all the required tools to produce the
        reference and surveillance signal processing channels from the raw digitalized and downloaded antenna channels.

        The reference channel preparation related functions are the followings:
          - func: isolate_channels(iq_samples_matrix, ref_position): Use this function if the reference signal is
                  received on a dedicated antenna and receiver chain.
          - func: beamform_reference(iq_signal_matrix, method , params .... ): Creates the reference channel via
                  beam space processing from the available antenna channels matrix.
                  (Only in later versions !)
          - func: regenerate_reference(reference_channel, params ....): Regenerates the reference signal using the
                  available information from its modulation properties. (DVB-T, ..)
                  (Only in later versions !)
          - func: beamform_surveillance(iq_signal_matrix, method , params .... ): Creates the surveillance channel via
                  beam space processing from the available antenna channels matrix.
                  (Only in later versions !)
          - func: time_domain_filter_surveillance(reference_ch, surveillance_ch, ...) Filters the surveillance channel
                  in time domain using available reference channel.
                  (Only in later versions !)
          - func: prefilter_surveillance(reference_ch, surveillance_ch, method, ...) Filters all the surveillance
                  channels before the beamforming is performed. 
                  (Only in later versions !)

     Demonstrations:
     ---------------
        The underlying functions perform demonstration for the previously described channel preparing operations.

     Notes:
     ------------

        TODO: Write demonstration functions
    
     Dependencies:
     -------------
         - pyARGUS library is required to run these filtering functions
         - Most of the algorithms are implemented in the file named "clutterCancellation"

     Features:
     ------------

     Project: pyAPRiL

     Authors: Tamás Pető

     License: GNU General Public License v3 (GPLv3)

     Changelog :
         - Ver 1.0000    : Initial version (2017 04 11)
         - Ver 1.0010    : Surveillance beamformer function (2017 05 13)
         - Ver 1.0011    : Error corr. in ref and surv BF (2017 05 15)
         - Ver 1.0020    : Implementation of time domain filters(2017 06 01)
         - Ver 1.0030    : Implementation of pre-filtering techniques (2017 06)


     Version format: Ver X.ABCD
                         A - Complete code reorganization
                         B - Processing stage modification or implementation
                         C - Algorithm modification or implementation
                         D - Error correction, Status display modification

 """


def isolate_channels(iq_samples_matrix, ref_position):
    """
        Description:
        -----------
        This function assumes that the surveillance and reference channels are digitalized and downloaded together
        in the digital data acquisition stage, however the reference signal is received on a dedicated receiver and
        antenna channel. At the output of the function the isolated reference signal and the rearranged matrix of
        the surveillance channels will appear.

        Parameters:
        -----------

            :param: iq_samples_matrix:  This matrix contains the signal array download from
                               the receiver hardware, which contains all the antenna channels. M is the count of the
                               surveillance channels, while N is the number of samples in a channel.
            :param: ref_position: Index of the reference channel is the iq_sample matrix.
            :type iq_samples_matrix:  M+1 x N complex numpy array
            :type ref_position: int

        Return values:
        --------------
            :return: reference_channel:  Isolated reference channel
            :return: surveillance_channels: Isolated surveillance channels
            :rtype: sureveillance channel: (N by 1 numpy array)
            :rtype: reference_channel: (N by M numpy array)
    """
     # --- Input check ---
    N = np.size(iq_samples_matrix, 1)
    Mp1 = np.size(iq_samples_matrix, 0)
    
    if Mp1 > N:
        print("WARNING: The number of antenna channels is greater than the number of samples in a channel. M+1 > N")
        print("You may flipped the input matrix")
        print("N:", N)
        print("M+1:", Mp1)
    
    if ref_position >= Mp1:
        print("ERROR: Invalid reference channel index. The requested index is greater than the "
              "number of channels.\nValid indexes are:  0..%d" % (Mp1-1))
        print("Requested ch index: ", ref_position)
        print("No valid outputs are generated!")
        return None, None

    # --- Process ---
    reference_channel = np.zeros(N, dtype=complex)
    reference_channel[:] = iq_samples_matrix[ref_position, :]
    
    surveillance_channels = np.zeros((Mp1-1, N), dtype=complex)
    surveillance_channels[:] = np.delete(iq_samples_matrix, ref_position, 0)[:]
    
    return reference_channel, surveillance_channels

