# -*- coding: utf-8 -*-
import numpy as np
from pyargus import directionEstimation as de

"""

                             Python based Advanced Passive Radar Library (pyAPRiL)

                                          Hit Processor Module


     Description:
     ------------
         Contains the implementation of the most common hit processing algorithms.
             - Plot extraction    
             - Hit clusterization             
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
         - Ver 2.0.0    : Plot extraction (2022 05 25)

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

class Plot:
     """
        This class stores the parameters of a plot extracted from the
        range-Doppler map after detection including:
            - range
            - Doppler
            - azimuth
            - elevation
        information.
     """
     def __init__(self, plot_ID, range_cell, Doppler_cell):
         """
             range_cell : range cell of the first detected hit
             Doppler_cell : Doppler cell of the first detected hit

             Further range and Doppler cells may be added later as the target
             amoebae is beeing discoved.

             Generally ,azimuth and elevation information is calculated only 
             for one cell that has the largest SINR value
         """
         self.plot_ID       = plot_ID # Numerical identifier
         self.range_cells   = [] # Filled up from hit matrix
         self.Doppler_cells = [] # Filled up from hit matrix
         self.sinr_inf      = [] # Filled up from SINR matrix
         self.azimuth   = 0 # Filled up from DoA estimation 
         self.elevation = 0 # Filled up from DoA estimation 

         self.range_cells.append(range_cell)
         self.Doppler_cells.append(Doppler_cell)
         
         self.weighted_rd = np.array([0,0], dtype=float) # [Doppler, range] cell indexes
         self.max_rd      = np.array([0,0]) # Doppler, range cell index that has maximum SINR

     def add_hit(self, range_cell, Doppler_cell):
         self.range_cells.append(range_cell)
         self.Doppler_cells.append(Doppler_cell)
         
     def max_snr_rd_calc(self):
         self.max_rd = [self.Doppler_cells[np.argmax(self.sinr_inf)], self.range_cells[np.argmax(self.sinr_inf)]]

     def central_mass_calc(self):
         # Convert sinr values from dB
         sinrs = 10**(np.array(self.sinr_inf)/20)
         for hit_index in range(len(self.range_cells)):
           self.weighted_rd[1]  += sinrs[hit_index] * self.range_cells[hit_index]
           self.weighted_rd[0]  += sinrs[hit_index] * self.Doppler_cells[hit_index]
         
         self.weighted_rd /= sum(sinrs)


def hit_search(plot, hit_matrix, range_cell, Doppler_cell, d_max):
    """
    Description:
        -----------

        Search for additional hits in the hit matrix in the immediate vicinity
        of the given range and Doppler cell. The d_max parameter controlls the
        maximum allowable distance from the reference range and Doppler cells.
        Hits that are located further away then d_max will not be enrolled.
       
        
        Parameters:
        -----------

            :param: plot: Plot object. Further found hits are added to this object
            :param: range_cell: Reference point in range
            :param: Doppler_cell: Reference point in Doppler
            :param: d_max: Maximum cell distance from the reference point

        Return values:
        --------------
            :return: plot object is modified
    """
    d_size = np.size(hit_matrix, 0)
    r_size = np.size(hit_matrix, 1)
    for d in np.arange(-d_max, d_max+1, 1):
        for r in np.arange(-d_max, d_max+1, 1):
            if  0 < Doppler_cell+d < d_size and 0 < range_cell + r < r_size:  # Out of hit matrix check
                if (hit_matrix[Doppler_cell+d, range_cell+r]) == 1:
                    plot.add_hit(range_cell+r, Doppler_cell+d)
                    hit_matrix[Doppler_cell+d, range_cell+r] = 0
                    hit_search(plot, hit_matrix, range_cell+r, Doppler_cell+d, d_max)


def plot_extractor(hit_matrix_orig, sinr_matrix=None, d_max=1):
    """
        Description:
        -----------

        Combine a group of hits into a plot. The d_max parameter controlls the
        maximum allowable distance between two hits to be combined into a single 
        plot.
        
        If the sinr_matrix is specified the plot extractor will use the SINR 
        values of the hits as weight coefficients to obtain the centrall mass point
        of the plot in the range and Doppler domain.
        
        Parameters:
        -----------

            :param: hit_matrix_orig: (D by R size real numpy array) This matrix has the same size
                                     as the range-Doppler map and contains the hit positions. It has only
                                     "0" and "1" values.
            :param: sinr_matrix: (D by R size real numpy array) This matrix has the same size
                                     as the range-Doppler map and contains the estimated Signal to Noise Plus
                                     Interference Ratio values of the cells.
            :param: d_max: (int) Maximum allowable distance between two hits

        Return values:
        --------------
            :return: plot_list: (list of Plot classes) Extracted Plots            
        
    """

    Doppler_cells = np.size(hit_matrix_orig, 0)
    range_cells = np.size(hit_matrix_orig, 1)
    hit_matrix = np.zeros((Doppler_cells, range_cells), dtype=int)
    hit_matrix[:] = hit_matrix_orig[:]

    if sinr_matrix is None:
        sinr_matrix =np.ones((Doppler_cells, range_cells), dtype=int)
        
    plot_list = []
    plot_cntr = 0
   
    for d in range(Doppler_cells):
        for r in range(range_cells):
            if hit_matrix[d, r] == 1:
               
                # Register hit
                plot_inst0 =Plot(plot_cntr, r, d)
                plot_list.append(plot_inst0)
                plot_cntr +=1
               
                # Delete hit from hit matrix
                hit_matrix[d, r] = 0
               
                # Search for additional hits in the vicinity of the founded plot
                hit_search(plot_inst0, hit_matrix, r, d, d_max)           
                #print("New plot with %d hits"%len(plot_inst0.range_cells))
                
    # Fill up plots with SINR information           
    for plot in plot_list:
        for hit_index in range(len(plot.range_cells)):
            r = plot.range_cells[hit_index]
            D = plot.Doppler_cells[hit_index]
            plot.sinr_inf.append(sinr_matrix[D, r])  # Add SINR information
   
    # Calculate weighted and maxium range-Doppler cell indexes
    for plot in plot_list:
        plot.central_mass_calc()
        plot.max_snr_rd_calc()
    return plot_list

def plot_filter(plot_list, range_gate, Doppler_gate, rd_size):
    """
        Applies various filterings on the plot list.
        
        Range gate filter:
        With specifing the range gate, the filter removes 
        those plots from the plot list that has smaller range than the range gate

        Doppler gate filter:
        With specifing the Doppler gate parameter, the filter removes 
        those plots from the plot list that has smaller absolute Doppler frequency than the Doppler gate


    Parameters
    ----------
    :param: plot_list : Plot list
    
    range_gate : range cell limit. Plots bellow this range cell limit will be filtered
    
    :param: Doppler_gate : Doppler cell limit. Plots having smaller Doppler frequency than 
                          this limit in absolute value will be filtered
    :param: rd_size      : range-Doppler matrix size in the following format [Doppler, range]
    
    :type:plot_list: Python list of Plot objects
    :type range_gate: integer
    :type Doppler_gate: integer
    :type rd_size: Pythong list, that has 2 integers



    Returns
    -------
    Filtered plot list    

    """
    filter_list = []
    # Filter close range plots
    for plot in plot_list:        
        if plot.weighted_rd[1] <= range_gate:
            filter_list.append(plot)
            print("Range gate - plot filtered: {0}".format(plot.weighted_rd))
    for plot_rm in filter_list:
        plot_list.remove(plot_rm)
    
    
    filter_list = []
    # Filter small Doppler plots
    for plot in plot_list:
        if abs(plot.weighted_rd[0]-(rd_size[0]-1)//2) <= Doppler_gate:
            filter_list.append(plot)          
            print("Doppler gate: -plot filtered: {0}".format(plot.weighted_rd))
    for plot_rm in filter_list:
        plot_list.remove(plot_rm)

    return plot_list

"""
#
#   D E M O
#
# This code demonstrates the operation of the plot extractor

# Generate test hit matrix
hit_matrix_orig  = np.zeros((15,10),dtype=int)
sinr_matrix = np.zeros( (15,10), dtype = float)
# Target #1
hit_matrix_orig [2,4] = 1
hit_matrix_orig [3,3] = 1
hit_matrix_orig [3,4] = 1

sinr_matrix [2,4] = 5
sinr_matrix [3,3] = 4
sinr_matrix [3,4] = 1

# Target #2
hit_matrix_orig [4,8] = 1
sinr_matrix [4,8] = 10

# Target #3
hit_matrix_orig [6,6] = 1
hit_matrix_orig [7,6] = 1

sinr_matrix [6,6] = 5
sinr_matrix [7,6] = 1

plot_list = plot_extractor(hit_matrix_orig, sinr_matrix)
for plot in plot_list:
    print(plot.weighted_rd)

plot_list = plot_filter(plot_list, range_gate=4, Doppler_gate=1, rd_size=hit_matrix_orig.shape)
for plot in plot_list:
    print(plot.weighted_rd)
"""
