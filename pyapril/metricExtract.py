# -*- coding: utf-8 -*-
#from pyapril import detector
#from pyapril import clutterCancellation as CC
from RDTools import export_rd_matrix_img

# Import APRiL package
import sys
import os
currentPath = os.path.dirname(os.path.realpath(__file__))
april_path = os.path.join(currentPath, "pyapril")
sys.path.insert(0, april_path)
import detector
import clutterCancellation as CC
import channelPreparation as chprep



import numpy as np
"""
                             Python based Advanced Passive Radar Library (pyAPRiL)
                                          Metric Evaluation Module
     Description:
     ------------
        Util functions implemented in this module can be used to evalute the performance of an investigated algorithm, method or
        the radar system.
        For the detailed description and definition of the performance metrics please read the corresponding documentation.
                    
     Notes:
     ------------
     Features:
     ------------
     Project: pyAPRIL
     Authors: Tamás Pető
     License: GNU General Public License v3 (GPLv3)
     Changelog :
         - Ver 1.0.0    : Initial version (2015 09 01)
         - Ver 1.0.1    : Code restructuring (2019 08 03)
         - Ver 1.1.0    : Construction of various wrapper functions (2019 08 10)
         - Ver 1.2.0    : New metrics: L, Rdpi, Rzdc , P, D (2019 08 17)
 """

def extract_delta(rd_matrix, target_rd, win):
    """
    Description:
    ------------
        Extracts the "Delta" detection performance metric from the given range-Doppler map
        with specifing the true target coordinates. For the estimation of the variance of the 
        noise, the utilized traning cells are extracted from the inmediate vicinity of the target 
        correlation peak.
        This detection metric is inspired by the CA-CFAR method.
        
    Implementation notes:
    ---------------------
       The training cells for the noise variance estimation are selected from a rectangular shaped window.       
       
       The Doppler coordinate of the reference target cell is interpreted relative to the zero Doppler line.
       Eg: In case the range-Doppler map has a size of (2D+1) x R, then [D+1, :] is the zero Doppler range slice.
    
    Parameters:
    -----------
    :param rd_matrix: Range-Doppler map on which the metric extraction should be performed
    :paran target_rd: True target cell indices [range cell index, doppler cell index]
    :param win      : Parameters of the noise power estimation window 
                      [Est. window length, Est. window width, Guard window length, Guard window width]
                      
    :type rd_matrix: R x D complex numpy array
    :type target_rd: python list with 2 elements
    :type win      : python list with 4 elements
    
    Return values:
    --------------
    :return delta: Calculated performance metric
    :rtype delta: float
    """  
    
    # Decentralizing doppler index
    target_rd = target_rd.copy()
    target_rd[1] += (np.size(rd_matrix,0)-1)//2
    
    mask = np.ones((2 * win[1] + 1, 2 * win[0] + 1))
    mask[win[1] - win[3]:win[1] + 1 + win[3], win[0] - win[2]:win[0] + 1 + win[2]] =  np.zeros((win[3] * 2 + 1, win[2] * 2 + 1))
    cell_counter = np.sum(mask)

    rd_block = rd_matrix[target_rd[1] - win[1]:target_rd[1] + win[1] + 1, target_rd[0] - win[0]:target_rd[0] + win[0] + 1]
    try:
        rd_block = np.multiply(rd_block, mask)
    except ValueError:
        print("ERROR: Improper range-Doppler matrix shape!")
        print("ERROR: Noise floor is not calculated: 0")
        return 0
    rd_block = np.multiply(rd_block, mask)
    rd_block = np.abs(rd_block)**2
    P_env = np.sum(rd_block)/cell_counter
    P_target = np.abs(rd_matrix[target_rd[1], target_rd[0]])**2
    delta = 10*np.log10(P_target / P_env)
    
    return delta
    
def extract_mu(rd_matrix, target_rd, surv_ch=None, p_surv=None):
    """
    Description:
    ------------
        Calculates the "Mu" detection performance metric using the range-Doppler map and the surveillance channel.
        
    Implementation notes:
    ---------------------
    Either the surv_ch or p_surv paramater has to be defined for the metric calculation.

    The Doppler coordinate of the reference target cell is interpreted relative to the zero Doppler line.
    Eg: In case the range-Doppler map has a size of (2D+1) x R, then [D+1, :] is the zero Doppler range slice.
       
    Parameters:
    -----------
    :param rd_matrix: Range-Doppler map on which the metric extraction should be performed
    :paran target_rd: True target cell indices [range cell index, doppler cell index]
    :param surv_ch  : Surveillance channel samples (default: None)
    :param p_sruv   : Average power of the surveillance channel
                      
    :type rd_matrix: R x D complex numpy array
    :type target_rd: python list with 2 elements
    :type surv_ch  : complex numpy array
    :type p_surv   : float
    
    Return values:
    --------------
    :return delta: Calculated performance metric
    :rtype delta: float
    """  
    
    # Decentralizing doppler index
    target_rd = target_rd.copy()
    target_rd[1] += (np.size(rd_matrix,0)-1)//2
    
    if p_surv is None:
        if surv_ch is not None:
            p_surv = (np.dot(surv_ch, surv_ch.conj())).real
        else:
            return None
    P_target = np.abs(rd_matrix[target_rd[1], target_rd[0]])**2
    mu = 10*np.log10(P_target / p_surv)    
    return mu    


def extract_noise_floor(rd_matrix, win, win_pos):
    """
    Description:
    ------------
        Estimates the noise floor of the range-Doppler map in the specified region using
        a rectangular shaped window. The size and the position of the estimation window
        is configurable via the input parameters.
        
        The estimation is performed with average statistic.
        
    Implementation notes:
    ---------------------
       
    Parameters:
    -----------
    :param rd_matrix: Range-Doppler map on which the metric extraction should be performed
    :paran win_shape: Rectangular estimation window parameters [onse sided window_length, one sided window_width]
    :param win_pos  : range-Doppler coordinates of the estimation window [range_cell_index, Doppler_cell_index]
                      
    :type rd_matrix: R x D complex numpy array
    :type win_shape: python list with 2 elements
    :type win_pos  : python list with 2 elements
    
    Return values:
    --------------
    :return nf: Estimated noise floor
    :rtype  nf: float
    """    
    # Decentralizing doppler index
    win_pos = win_pos.copy()
    win_pos[1] += (np.size(rd_matrix,0)-1)//2
    
    mask = np.ones((2 * win[1] + 1, 2 * win[0] + 1))    
    
    # Specifiy arbitrary mask here
    cell_counter = np.sum(mask)

    rd_block = rd_matrix[win_pos[1] - win[1]:win_pos[1] + win[1] + 1, win_pos[0] - win[0]:win_pos[0] + win[0] + 1]
    try:
        rd_block = np.multiply(rd_block, mask)
    except ValueError:
        print("ERROR: Improper range-Doppler matrix shape!")
        print("ERROR: Noise floor is not calculated: 0")
        return 0
    rd_block = np.abs(rd_block)**2
    P_env = np.sum(rd_block)/cell_counter
    
    return 10*np.log10(P_env)

def extract_peak_clutter(rd_matrix):
    """
    Description:
    ------------
        Extracts the power of the strongest signal component from the range-Dopler
        decomposition. 
        We can safely assume that this component sets the noise floor level.
        
    Implementation notes:
    ---------------------
       
    Parameters:
    -----------
    :param rd_matrix: Range-Doppler map on which the metric extraction should be performed
                      
    :type rd_matrix: R x D complex numpy array
    
    Return values:
    --------------
    :return p: Power of the strongest clutter component
    :rtype  p: float
    """    
    max_amp = np.max(np.abs(rd_matrix))
    return max_amp**2

def extract_dynamic_range(rd_matrix, win, win_pos):
    """
    Description:
    ------------
        Extracts the power of the strongest signal component from the range-Dopler
        decomposition. 
        We can safely assume that this component sets the noise floor level.
        
    Implementation notes:
    ---------------------
       
    Parameters:
    -----------
    :param rd_matrix: Range-Doppler map on which the metric extraction should be performed
    :param win      : Parameters of the noise power estimation window 
                      [Est. window length, Est. window width, Guard window length, Guard window width]
    :param win_pos  : range-Doppler coordinates of the noise floor 
                      estimation window [range_cell_index, Doppler_cell_index]
    
    :type rd_matrix: R x D complex numpy array
    :type win      : python list with 4 elements
    :type win_pos  : python list with 2 elements
    
    Return values:
    --------------
    :return dyn_range: Dynamic range of the range-Doppler domain
    :rtype  dyn: float
    """        
    # Decentralizing doppler index
    win_pos = win_pos.copy()
    win_pos[1] += (np.size(rd_matrix,0)-1)//2
    
    mask = np.ones((2 * win[1] + 1, 2 * win[0] + 1))    
    
    # Specifiy arbitrary mask here
    cell_counter = np.sum(mask)

    rd_block = rd_matrix[win_pos[1] - win[1]:win_pos[1] + win[1] + 1, win_pos[0] - win[0]:win_pos[0] + win[0] + 1]
    try:
        rd_block = np.multiply(rd_block, mask)
    except ValueError:
        print("ERROR: Improper range-Doppler matrix shape!")
        print("ERROR: Noise floor is not calculated: 0")
        return 0
    rd_block = np.abs(rd_block)**2
    P_env = np.sum(rd_block)/cell_counter
    P_peak = np.max(np.abs(rd_matrix))**2
    dynamic_range = P_peak/P_env
    return dynamic_range

def extract_alpha(rd_matrix, target_rd, win, win_pos):
    """
    Description:
    ------------
        Extracts the "Alpha" detection performance metric from the given range-Doppler map
        with specifing the true target coordinates and a region where only noise is expected.
        
        In contrast to the "Delta" metric the average noise power is not estimated from the 
        inmediate vicinity of the target correlation peak, but from a distant place, where
        the target energy contribution is negligible.
        
        
    Implementation notes:
    ---------------------
       The training cells for the noise variance estimation are selected from a rectangular shaped window.
    
        The Doppler coordinate of the reference target cell is interpreted relative to the zero Doppler line.
        Eg: In case the range-Doppler map has a size of (2D+1) x R, then [D+1, :] is the zero Doppler range slice.
    
    Parameters:
    -----------
    :param rd_matrix: Range-Doppler map on which the metric extraction should be performed
    :paran target_rd: True target cell indices [range cell index, doppler cell index]
    :param win      : Parameters of the noise power estimation window 
                      [Est. window length, Est. window width, Guard window length, Guard window width]
    :param win_pos  : range-Doppler coordinates of the noise floor 
                  estimation window [range_cell_index, Doppler_cell_index]

    :type rd_matrix: R x D complex numpy array
    :type target_rd: python list with 2 elements
    :type win      : python list with 4 elements
    :type win_pos  : python list with 2 elements
    
    Return values:
    --------------
    :return alpha: Calculated performance metric
    :rtype alpha: float
    """  
    
    # Decentralizing doppler index
    target_rd = target_rd.copy()
    win_pos = win_pos.copy()
    target_rd[1] += (np.size(rd_matrix,0)-1)//2
    win_pos[1] += (np.size(rd_matrix,0)-1)//2
    
    mask = np.ones((2 * win[1] + 1, 2 * win[0] + 1))    
    # Specifiy arbitrary mask here
    cell_counter = np.sum(mask)
    rd_block = rd_matrix[win_pos[1] - win[1]:win_pos[1] + win[1] + 1, win_pos[0] - win[0]:win_pos[0] + win[0] + 1]
    try:
        rd_block = np.multiply(rd_block, mask)
    except ValueError:
        print("ERROR: Improper range-Doppler matrix shape!")
        print("ERROR: Noise floor is not calculated: 0")
        return 0
    rd_block = np.abs(rd_block)**2
    P_env = np.sum(rd_block)/cell_counter
    P_target = np.abs(rd_matrix[target_rd[1], target_rd[0]])**2
    alpha = 10*np.log10(P_target / P_env)   
    
    return alpha


def eval_clutter_filt_perf(**kwargs):
    """
    Description:
    ------------
        This function evalutes a number of clutter cancellation performance
        metric. A metric is evaluted only when all the required parameters are specified for it.
        The required parameter list of the implemented metrics are listed as follows:
            
            - CA (Clutter Attenuation):
                -- surv_ch
                -- surv_ch_filt
            - Rnf (Noise floor reduction):
                -- rd_matrix
                -- rd_matrix_filt
                -- win
                -- win_pos
            - Mu (Peak to channel power ratio)
                -- rd_matrix
                -- rd_matrix_filt
                -- target_rd
                -- surv_ch
                -- survch_filt
            - Delta ("quasi" SINR improvement)
                -- rd_matrix
                -- rd_matrix_filt
                -- target_rd
                -- win
            - Alpha (Peak to noise floor ratio)
                -- rd_matrix
                -- rd_matrix_filt
                -- target_rd
                -- win
                -- win_pos
            - L (Target power loss)
                -- rd matrix
                -- rd_matrix_filt
                -- target_rd
            - Rdpi (Direct path interference reduction)
                -- rd_matrix
                -- rd_matrix_filt
            - Rzdc (Zero Doppler clutter reduction)
                -- rd_matrix
                -- rd_matrix_filt
                -- max_clutter_delay
            - P (Peak clutter reduction)
                -- rd_matrix
                -- rd_matrix_filt
            - D (Dynamic range compression)
                -- rd_matrix
                -- rd_matrix_filt
                -- win
                -- win_pos
                
    Implementation notes:
    ---------------------
       
    Parameters:
    -----------
        
     **kwargs
        The required metric extraction parameters are received through the **kwargs interface
        Valid keys are the followins:
        
        :key surv_ch           : Surveillance channel
        :key surv_ch_filt      : Clutter filtered surveillance channel
        :key rd_matrix         : Range-Doppler map without clutter filtering
        :key rd_matrix_filt    : Range-Doppler map with clutter filtering
        :key target_rd         : True target cell indices [range cell index, doppler cell index]
        :key win               : Parameters of the noise power estimation window 
                                 [Est. window length, Est. window width, 
                                 Guard window length, Guard window width]         
        :key win_pos           : range-Doppler coordinates of the estimation 
                                 window [range_cell_index, Doppler_cell_index]
        :key max_clutter_delay : Maximum time delay value of the considered clutter
                                 components
                          
        :type surv_ch          : 1 x N complex numpy array
        :type surv_ch_filt     : 1 x N complex numpy array
        :type rd_matrix        : R x D complex numpy array
        :type rd_matrix_filt   : R x D complex numpy array
        :type target_rd        : pytohn list with 2 elements
        :type win              : python list with 4 elements
        :type win_pos          : python list with 2 elements
        :type max_clutter_delay: integer 

    
    Return values:
    --------------
    :return metrics: Calculated metrics
    :rtype metrics: dictionary {short metric name[str]: value[float]}
    """
    metrics = {}
    
    # kwargs processing
    surv_ch = kwargs.get('surv_ch')
    surv_ch_filt = kwargs.get('surv_ch_filt')
    rd_matrix = kwargs.get('rd_matrix')
    rd_matrix_filt = kwargs.get('rd_matrix_filt')    
    target_rd = kwargs.get('target_rd')
    win = kwargs.get('win')
    win_pos = kwargs.get('win_pos')
    max_clutter_delay = kwargs.get('max_clutter_delay')
    
    """
    CA - Clutter Attenuation
    """
    if not( surv_ch is None or surv_ch_filt is None):
        ca = 10*np.log10((np.dot(surv_ch, surv_ch.conj())).real / (np.dot(surv_ch_filt, surv_ch_filt.conj())).real)
        metrics.update({'CA': ca})
        
    """
    Rnf - Noise floor reduction
    """
    if not( rd_matrix is None or rd_matrix_filt is None or\
        win is None or win_pos is None):
        nf = extract_noise_floor(rd_matrix, [win[0], win[1]], win_pos)        
        nf_filt = extract_noise_floor(rd_matrix_filt, [win[0], win[1]], win_pos)
        Rnf = nf - nf_filt
        metrics.update({'R nf': Rnf})
        
    """
    Mu - Peak to channel power ratio
    """
    if not( surv_ch is None or rd_matrix is None or \
         surv_ch_filt is None or rd_matrix_filt is None or \
         target_rd is None):
        mu = extract_mu(rd_matrix, target_rd, surv_ch=surv_ch)
        mu_filt = extract_mu(rd_matrix_filt, target_rd, surv_ch=surv_ch_filt)
        mu_imp = mu_filt-mu
        metrics.update({'Mu imp': mu_imp})
    
    """
    Delta - "quasi" SINR improvement
    """
    if not( rd_matrix is None or rd_matrix_filt is None or\
            target_rd is None or win is None):
        delta = extract_delta(rd_matrix, target_rd, win)
        delta_filt = extract_delta(rd_matrix_filt, target_rd, win)
        delta_imp = delta_filt-delta
        metrics.update({'Delta imp': delta_imp})
    
    """
    Alpha - Peak to noise floor ratio
    """
    if not( rd_matrix is None or rd_matrix_filt is None or\
            target_rd is None or win is None or win_pos is None):
        alpha = extract_alpha(rd_matrix, target_rd, [win[0], win[1]], win_pos)
        alpha_filt = extract_alpha(rd_matrix_filt, target_rd, [win[0], win[1]], win_pos)
        alpha_imp = alpha_filt-alpha
        metrics.update({'Alpha imp': alpha_imp})     
    
    """
    L - Target peak power loss 
    """
    if not (rd_matrix is None or rd_matrix_filt is None or target_rd is None):
        offset = (np.size(rd_matrix,0)-1)//2 # Doppler index offset        
        target_pow_f = np.abs(rd_matrix_filt[target_rd[1]+offset, target_rd[0]])
        target_pow = np.abs(rd_matrix[target_rd[1]+offset, target_rd[0]])
        L = 20* np.log10(target_pow_f/target_pow)
        metrics.update({'L':L})
    
    """
    Rdpi - Direct path interference reduction
    """
    if not (rd_matrix is None or rd_matrix_filt is None):
        zd_index = (np.size(rd_matrix,0)-1)//2+1        
        dpi_pow_f = np.abs(rd_matrix_filt[zd_index, 0]) 
        dpi_pow = np.abs(rd_matrix[zd_index, 0])
        Rdpi = 20* np.log10(dpi_pow/dpi_pow_f)
        metrics.update({'R dpi':Rdpi})        
        
    """
    Rzdc - Zero Doppler clutter reduction
    """
    if not (rd_matrix is None or rd_matrix_filt is None or max_clutter_delay is None):
        zd_index = (np.size(rd_matrix,0)-1)//2+1
        clutter_pow_f = np.sum(np.abs(rd_matrix_filt[zd_index, 0:max_clutter_delay])**2)
        clutter_pow   = np.sum(np.abs(rd_matrix     [zd_index, 0:max_clutter_delay])**2)        
        Rzdc = 10* np.log10(clutter_pow/clutter_pow_f)
        metrics.update({'R zdc':Rzdc})
        
    
    """
    P - Peak clutter reduction
    """
    if not (rd_matrix is None or rd_matrix_filt is None):
        peak_clutter_pow = extract_peak_clutter(rd_matrix)
        peak_clutter_pow_f = extract_peak_clutter(rd_matrix_filt)
        P = 10*np.log10(peak_clutter_pow/peak_clutter_pow_f)
        metrics.update({'P':P})

    """
    D - Dynamic range compression
    """
    if not (rd_matrix is None or rd_matrix_filt is None):
        d = extract_dynamic_range(rd_matrix, [win[0], win[1]], win_pos)
        d_f = extract_dynamic_range(rd_matrix_filt, [win[0], win[1]], win_pos)
        D = 10*np.log10(d-d_f)
        metrics.update({'D':D})        
    return metrics


def eval_clutter_filter_raw(ref_ch, surv_ch, filter_method, **kwargs):
    """
    Description:
    ------------
      Evaluates the performance of a clutter filter algorithm directly from the
      raw iqsamples.
        
    Implementation notes:
    ---------------------
       Since some metrics requires the knowledge of the range-Doppler matrix, this
       function automatically calculates the filtered and unfiltered range-Doppler maps.
       The range-Doppler map calculation is done with the so-called overlap and save method.
       (Please check the documentation of the detector module for further information)
       The maximum range and maximum Doppler frequency parameters of the range-Doppler map 
       are automatically determined from the input data (target, win, win_pos) 
       in case the 'Delta' or the 'Alpha' metric is requested. In other cases the
       size of calculated range-Doppler must be specified explicity via the rd_size
       parameter. The automatically calculated values can be overriden with the rd_size parameter
       in all cases.
    
    Parameters:
    -----------
    
    :param ref_ch        : Reference channel samples
    :param surv_ch       : Surveillance channel samples
    :param filter_method : Name of the used clutter filtering algorithm.eg: "SMI-MRE"
                           For the list of the available clutter filtering algorithms
                           please check the documenation of clutterCancellation module.

    :type ref_ch        : N x 1 complex numpy array
    :type surv_ch       : N x 1 complex numpy array
    :type filter_method : string
    
     **kwargs
        The required metric extraction parameters are received through the **kwargs interface
        Valid keys are the followings:
        
        :key rd_windowing  : Windowing function used on the range-Doppler map
                             For detailed description check the documenation of the
                             detector module. 
        :key rd_size       : Sizze of the calculated range-Doppler map, measured in
                             cells. (Interpreted as one sided in the Doppler domain)
        :type rd_windowing : string
        :type rd_size      : Python list with 2 integer elements [int int]
        
    For the description of the further **kwargs parameters please check the 
    function header or the documenation of the "eval_clutter_filt_perf" function.
    
    
    Return values:
    --------------
    :return metrics: Calculated clutter filter performance metrics
    :rtype metrics: dictionary {short metric name[str]: value[float]}
    """ 
    
    # kwargs processing
    target_rd = kwargs.get('target_rd')
    win = kwargs.get('win')
    win_pos = kwargs.get('win_pos')
    rd_windowing = kwargs.get('rd_windowing')
    max_clutter_delay = kwargs.get('max_clutter_delay')
    rd_size = kwargs.get('rd_size')
    
    # Perform clutter filtering        
    surv_ch_filt = chprep.time_domain_filter_surveillance(ref_ch, surv_ch, filter_method, **kwargs)
    
    # Calculate range-Doppler maps if requested
    if not ((target_rd is None or win is None) and rd_size is None):        
        if rd_size is None:            
            target_Doppler = abs(target_rd[1])+win[1]
            if win_pos is not None: win_pos_Doppler = abs(win_pos[1])+win[1]
            else: win_pos_Doppler=0         
            max_Doppler = target_Doppler if target_Doppler > win_pos_Doppler else win_pos_Doppler            
            
            target_range_pow2 = int(2**(np.ceil(np.log2(target_rd[0]+win[0]))))
            if win_pos is not None: win_pos_range_pow2 = int(2**(np.ceil(np.log2(win_pos[0]+win[0]))))
            max_Range = target_range_pow2 if target_range_pow2 > win_pos_range_pow2 else win_pos_range_pow2
        
        else:
            max_Range   = rd_size[0]
            max_Doppler = rd_size[1]
     
        max_Doppler/=(2*np.size(ref_ch))        
        if rd_windowing is not None:
            surv_ch = detector.windowing(surv_ch, rd_windowing)
            surv_ch_filt = detector.windowing(surv_ch_filt, rd_windowing)
        
        rd_matrix = detector.cc_detector_ons(ref_ch, surv_ch, 1, max_Doppler, max_Range, verbose=0, Qt_obj=None)
        rd_matrix_filt = detector.cc_detector_ons(ref_ch, surv_ch_filt,  1, max_Doppler, max_Range, verbose=0, Qt_obj=None)    
    else:
        rd_matrix = None
        rd_matrix_filt = None

    # Evaluate metrics    
    metrics = eval_clutter_filt_perf(surv_ch=surv_ch, 
                                     surv_ch_filt=surv_ch_filt,
                                     rd_matrix=rd_matrix, 
                                     rd_matrix_filt=rd_matrix_filt,
                                     target_rd=target_rd, 
                                     win=win, 
                                     win_pos=win_pos,
                                     max_clutter_delay=max_clutter_delay)
    return metrics



def eval_metrics_on_track(target_rds, rd_matrices, **kwargs):
    """
    Description:
    ------------
        Evaluates all the possible detection metrics through the trajectory of a detected target.
        A metric is evaluted only when all the required parameters are specified for it.
        The required parameter list of the implemented metrics are listed as follows:
            
            - Mu (Peak to channel power ratio)
                -- rd_matrices (mandatory)
                -- target_rds  (mandatory)
                -- surv_ch or p_surv_ch
            - Delta ("quasi" SINR improvement)
                -- rd_matrices (mandatory)
                -- target_rds  (mandatory)
                -- win
            - Alpha (Peak to noise floor ratio)
                -- rd_matrices (mandatory)
                -- target_rds  (mandatory)
                -- win
                -- win_pos
        
    Implementation notes:
    ---------------------
        Non valid detections should be noted with [-1, x] values in the target_rds array.
        These CPIs are ignored.
    
    Parameters:
    -----------    
    :param target_rds : A matrix made from the true target cell indices 
                        [range cell index 1, doppler cell index
                         range cell index 2, doppler cell index
                         ...               , ...                ]
    :param rd_matrices : Range-Doppler maps on which the metric extraction should be performed
                      
    :type target_rds : P x 2 real valued numpy array, where P is the number of CPIs
    :type rd_matrices: P x D x R complex numpy array
    
     **kwargs
        The required metric extraction parameters are received through the **kwargs interface
        Valid keys are the followins:
        
        :key surv_ch       : Surveillance channel
        :key p_surv_ch     : Average power of the surveillance channel
        :key win           : Parameters of the noise power estimation window 
                             [Est. window length, Est. window width, Guard window length, Guard window width]         
        :key win_pos       : range-Doppler coordinates of the estimation window [range_cell_index, Doppler_cell_index]
                      
                          
        :type surv_ch       : P x N complex numpy array
        :type p_surv_ch     : P x 1 real valued numpy array
        :type win           : python list with 4 elements
        :type win_pos       : python list with 2 elements

    Return values:
    --------------
    :return metric_array : Calculated performance metrics
    :rtype metric_array  : M x P sized real valued numpy array
    """ 
    
    # kwargs processing
    surv_ch = kwargs.get('surv_ch')
    p_surv_ch = kwargs.get('p_surv_ch')
    win = kwargs.get('win')
    win_pos = kwargs.get('win_pos')
    
    P = np.size(target_rds, 0) # Number of Coherent Processing Intervals
    valid_cpi_indices = []
    deltas = []
    alphas = []
    mus  = []
    # Evaluation on full target track
    for p in range(P):
        if not (target_rds[p, 0] < 0 ): 
            valid_cpi_indices.append(p)        
            
            # Delta
            if win is not None:
                delta = extract_delta(rd_matrices[p, :, :], target_rds[p,:], win)
                deltas.append(delta)
            
            # Alpha
            if win is not None and win_pos is not None:
                alpha = extract_alpha(rd_matrices[p, :, :], target_rds[p,:], [win[0], win[1]], win_pos)        
                alphas.append(alpha)      

            # Mu
            if surv_ch is not None or p_surv_ch is not None:
                mu = extract_mu(rd_matrices[p, :, :], target_rds[p,:], surv_ch=surv_ch, p_surv=p_surv_ch)
                mus.append(mu)
    
    # Assmeble metric array
    metric_array = np.array(valid_cpi_indices)[:,None]
    
    if len(deltas) > 0:
            metric_array = np.concatenate([metric_array, np.array(deltas)[:,None]], axis=1)
    if len(alphas) > 0:
            metric_array =np.concatenate([metric_array, np.array(alphas)[:, None]], axis=1)            
    if len(mus) > 0:
            metric_array = np.stack([metric_array, np.array(mus)])

    return metric_array
    

def eval_clutter_filter_on_track_raw(iq_fname_temp, start_ind, 
                                     stop_ind, filter_method, 
                                     ref_ch_ind=0, surv_ch_ind=1,
                                     **kwargs):
    """
    Description:
    ------------
      Evaluates the performance of a clutter filter algorithm directly from 
      raw iqsamples on multiple records.
      
      In the first step it loads the raw iq samples of the p-th CPI. Then gets
      the targets true range-Doppler coordinates for the p-th CPI, from the 
      "target_rds" array (In case it is available).
      
      These information along with the metric calculation and clutter filter
      parameters are passed to the wrapped "eval_clutter_filter_raw" function,
      which automatically calculates the performance metric for the current record 
      (p-th CPI).
      This process is performed iteratively for all the available records. At the
      end of the process, the function composes a multidimensional array, in which
      each column corresponds to a specific type of metric. Values in each column
      express the clutter filter performance variation along the inspected target
      tracjectory according to a specific metric.
      
        
    Implementation notes:
    ---------------------
        IQ files are assumed to have the following naming convention.
        <Some sepcific name>_<cpi index>.npy
        E.g.: "PassiveRadarIQ_0.npy", ... "PassiveRadarIQ_15.npy"
        
        The field <Some sepcific name> is expected to received thorugh the 
        iq_fname_temp parameter.
        
        The expected format of the raw iq data file is complex numpy array (.npy)
        The loaded IQ sample array could contains data from multiple antenna channels.
        These individual channels are assumed to be packed into the different rows of the
        iq sample matrix. (The assumed format is M x N, where M is the number of channels
        and N is the number of iq samples.)
        The channels to be processed can be selected with the 'ref_ch_ind' and the
        'surv_ch_ind' parameters.
    
    Parameters:
    -----------
    TODO: Fill up these field
    :param iq_fname_temp: Common section of the iq file names (See imp. notes)
    :param start_ind    : Index of the first processed file
    :param stop_ind     : Index of the last processed file
        
    :param filter_metod : Name of the used clutter filtering algorithm.eg: "Winer-SMI-MRE"
                          For the list of the available clutter filtering algorithms
                          please check the documenation of clutterCancellation module.
    :param ref_ch_ind   : row index of the reference channel in the loaded iq sample matrix
                          (default value: 0)  
    :param surv_ch_ind  : row index of the surveillance channel in the loaded iq sample matrix 
                          (default value: 1)
                          
    :type iq_fname_temp : string
    :type start_ind     : int
    :type stop_ind      : int
    :type filter_method : string
    :type ref_ch_ind    : int
    :type surv_ch_ind   : int
                      
    
     **kwargs
        The required metric extraction parameters are received through the **kwargs interface
        Valid keys are the followins:
        
        :key target_rds  : A matrix made from the true target cell indices 
                           [range cell index 1, doppler cell index
                            range cell index 2, doppler cell index
                            ...               , ...                ]
                           The true coordinates of the target is required by some
                           of the performance metrics.(Such as: 'Delta', 'Alpha', 'Mu'..)
        
        :key rd_windowing  : Windowing function used on the range-Doppler map
                             For detailed description check the documenation of the
                             detector module. 
        
        :type rd_windowing : string
        :type target_rds   : P x 2 real valued numpy array, where P is the number of CPIs
        
    For the description of the further **kwargs parameters please check the 
    function header or the documenation of the "eval_clutter_filt_perf" function.
    
    
    Return values:
    --------------
    :return metric_array: Array of calculated performance metrics
    :rtype metric_array: real valued numpy array
    """ 
    # kwargs processing
    target_rds = kwargs.get('target_rds')
    
    metric_array = []
    metric_list = []    
    for p in np.arange(start_ind, stop_ind, 1):
        
        metric_list = []
        filename = iq_fname_temp+str(p)+".npy"
        #print("Processing: {:s}".format(filename))

        #Load
        # TODO: Handle channel select proplery
        iq = np.load(filename)
        ref_ch = iq[ref_ch_ind,:]
        surv_ch = iq [surv_ch_ind,:]
        
        if target_rds is not None:
            # Get real target coordinates
            target_rd = target_rds[p-start_ind, :]
            # Check validity
            if not (target_rd[0] < 0 ):                 
                metric_list.append(p)      

        else:
            metric_list.append(p)  
            target_rd=None
            
        # Evaluting metrics
        if metric_list: # Check emptyness
            if metric_list[-1] == p:
                metrics = eval_clutter_filter_raw(ref_ch, surv_ch, filter_method,
                                                  target_rd=target_rd, **kwargs)            
                # Extending metric array with
                for (metric, value) in metrics.items():
                        #print("{:s} : {:.2f} dB".format(metric,value))
                        metric_list.append(value)
                if len(metric_array) > 0:
                    metric_array = np.concatenate([metric_array, np.array(metric_list)[:,None]], axis=1)
                else:
                    metric_array = np.array(metric_list)[:,None]      


    return metric_array

def scan_time_domain_dimension(statistic, dim_list, iq_fname_temp, start_ind,
                               stop_ind, filter_method, **kwargs):
    """
    Description:
    ------------
      This function explores the performance improvement dependency of a 
      clutter filter in the time domain dimension.
      
      To achieve this, the function automatically sweeps through the previously
      set dimension region and evaluates the performance metrics for each dimension size.
              
    Implementation notes:
    ---------------------
        This function use the 'eval_clutter_filter_on_track_raw' to estimate the
        performance of the currently set filter dimension parameter.
    Parameters:
    -----------
    These input parameters are mandatory!
    
    :param statistic     : Type of the final metric statistic 'avg' / 'max' / 'median'
    :param dim_list      : List of the interested tap sizes eq:[2,3,4,5,6 ... , 64]
    :param iq_fname_temp : Template name of the processed iq files
                           (This field is required by the "eval_clutter_filter_on_track_raw" 
                            function, check its header for more details)
    :param start_ind     : Index of the first processed file 
                           (This field is required by the "eval_clutter_filter_on_track_raw" 
                            function, check its header for more details)
    :param stop_ind      : Index of the last processed file 
                           (This field is required by the "eval_clutter_filter_on_track_raw" 
                            function, check its header for more details)

    :param filter_method : Name of the inspected clutter filter algorithm
                           (This field is required by the "eval_clutter_filter_on_track_raw" 
                           function, check its header for more details)
    
    
    :type statistic      : string
    :type dim_list       : python list
    :type iq_fname_temp  : string
    :type start_ind      : int
    :type stop_ind       : int
    :type filter_method  : string
    
     **kwargs
        The required metric extraction and filter parameters are received 
        through the **kwargs interface. For more detailed description please 
        check the function header or the documenation of the 
        "eval_clutter_filter_on_track_raw" function.
        
    
    Return values:
    --------------
    :return scan_res: Array of calculated performance metrics 
    :rtype  scan_res: real valued numpy array
    
    None: Unknown statistic type is defined
    """
    scan_res = None # Results array of the parameter scan
    for K in dim_list:
        print("Current filter dimension is {:d}".format(K))
        metric_array = eval_clutter_filter_on_track_raw(iq_fname_temp=iq_fname_temp,
                                                        start_ind=start_ind,
                                                        stop_ind=stop_ind,
                                                        filter_method=filter_method,
                                                        K=K, **kwargs)
        
        metric_statistic = [K]
        
        # Calculate statistics:
        for m in np.arange(1, metric_array.shape[0], 1):
            if statistic == 'avg':
                stat = np.average(10**(metric_array[m, :]/20))
            elif statistic == 'max':
                stat = np.max(10**(metric_array[m, :]/20))
            elif statistic == 'median':
                stat = np.median(10**(metric_array[m, :]/20))
            else:
                print("ERROR: Unknown statistic requested:", statistic)
                return None
                                
            metric_statistic.append(20*np.log10(stat))
    
        if scan_res is not None:
            scan_res = np.concatenate([scan_res, np.array(metric_statistic)[:,None]], axis=1)
        else:
            scan_res = np.array(metric_statistic)[:,None]     
    
    # Normalization
    for i in np.arange(1, scan_res.shape[0], 1):
        scan_res[i,:] -= max(scan_res[i,:])
    
    return scan_res

def scan_ECA(statistic, time_range, doppler_range, iq_fname_temp, start_ind, stop_ind,
             filter_method, **kwargs):
    """
    Description:
    ------------
      This function explores the performance improvement dependency of a 
      clutter filter in the time domain dimension.
      
      To achieve this, the function automatically sweeps through the previously
      set dimension region and evaluates the performance metrics for each dimension size.
              
    Implementation notes:
    ---------------------
        This function use the 'eval_clutter_filter_on_track_raw' to estimate the
        performance of the currently set filter dimension parameter.
    Parameters:
    -----------
    These input parameters are mandatory!
    
    :param statistic     : Type of the final metric statistic 'avg' / 'max' / 'median'
    :param time_range    : Time domain scan will be performed on this region 
                           Format:[start,stop,step] eg:[2,64,1]
    :param doppler_range : Doppler domain scan will be performed on this region 
                           Format:[start,stop,step] eg:[0,3,1]
    :param iq_fname_temp : Template name of the processed iq files
                           (This field is required by the "eval_clutter_filter_on_track_raw" 
                            function, check its header for more details)
    :param start_ind     : Index of the first processed file 
                           (This field is required by the "eval_clutter_filter_on_track_raw" 
                            function, check its header for more details)
    :param stop_ind      : Index of the last processed file 
                           (This field is required by the "eval_clutter_filter_on_track_raw" 
                            function, check its header for more details)

    :param filter_method : Name of the inspected clutter filter algorithm
                           (This field is required by the "eval_clutter_filter_on_track_raw" 
                           function, check its header for more details)
    
    
    :type statistic      : string
    :type time_range     : python list
    :type doppler_range  : python list
    :type iq_fname_temp  : string
    :type start_ind      : int
    :type stop_ind       : int
    :type filter_method  : string
    
     **kwargs
        The required metric extraction and filter parameters are received 
        through the **kwargs interface. For more detailed description please 
        check the function header or the documenation of the 
        "eval_clutter_filter_on_track_raw" function.
        
    
    Return values:
    --------------
    :return scan_res: Array of calculated performance metrics 
    :rtype  scan_res: real valued numpy array
    
    None: Unknown statistic type is defined
    """
    time_domain_dims = np.arange(time_range[0],time_range[1], time_range[2])
    doppler_domain_dims = np.arange(doppler_range[0],doppler_range[1], doppler_range[2])
    
    scan_res = None # Results array of the parameter scan
    K_ind = -1
    D_ind = -1
    for D in doppler_domain_dims:
        D_ind += 1
        K_ind = -1
        for K in time_domain_dims:
            K_ind+=1
            print("Current filter dimensions are K:{:d} D:{:d}".format(K,D))
            metric_array = eval_clutter_filter_on_track_raw(iq_fname_temp=iq_fname_temp,
                                                            start_ind=start_ind,
                                                            stop_ind=stop_ind,
                                                            filter_method=filter_method,
                                                            K=K, 
                                                            D=D,
                                                            **kwargs)
            
            metric_statistic = []
            
            # Calculate statistics:
            for m in np.arange(1, metric_array.shape[0], 1):
                if statistic == 'avg':
                    stat = np.average(10**(metric_array[m, :]/20))
                elif statistic == 'max':
                    stat = np.max(10**(metric_array[m, :]/20))
                elif statistic == 'median':
                    stat = np.median(10**(metric_array[m, :]/20))
                else:
                    print("ERROR: Unknown statistic requested:", statistic)
                    return None
                                    
                metric_statistic.append(20*np.log10(stat))
            
            if scan_res is None:                
                scan_res = np.zeros((time_domain_dims.shape[0], doppler_domain_dims.shape[0], len(metric_statistic)))
            
            scan_res[K_ind, D_ind,:] = np.array(metric_statistic)
        
    # Normalization
    for i in range(len(metric_statistic)):
        scan_res[:,:,i] -= np.max(scan_res[:,:,i])
    
    return scan_res





