# -*- coding: utf-8 -*-
import numpy as np

"""

                             Python based Advanced Passive Radar Library (pyAPRiL)

                                    Target Localization Module


     Description:
     ------------
         Contains the implementations for target localization (bi and multistatic)
             - Multistatic target localization with the Spherical Intersection method
           
        
     Implementation Notes:
     ----------------------

     Project: pyAPRIL
     Authors: Tamás Pető
     License: GNU General Public License v3 (GPLv3)
 """
 
def solve_quadratic_eq(a,b,c):
     """
     Solves a quadratic equation.
     
     a*x2 +b*x + c = 0

     Parameters
     ----------
     a : float         
     b : float         
     c : float
         
     Returns
     -------
     x1 : Solution #1
     x2 : Solution #2
         
     """
     x1 = (-b + np.sqrt(b**2-4*a*c)) / (2*a)
     x2 = (-b - np.sqrt(b**2-4*a*c)) / (2*a)     
     return x1,x2
 
def localize_target_ms_sx(ioo_coords, rb_vec):
    """
    This function solves the multistatic localization problem using 
    the Spherical intersection approcch described in [1]
    
    Source: [1] Malanowski M: Signal Processing for Passive Radar (Section 8.2)
    
    Parameters:
    ----------
    :param ioo_coords: x,y,z descartes coordinates of the used illuminators [m]
    :param rb_vec: measured bistatic ranges [m]
        
    :type ioo_coords: float numpy array with size (NTx) x 3, where NTx is the 
        number of illuminators
    :type rb_vec: float numpy array with size (NTx) x 1, where NTx is the 
        number of illuminators
            
    Returns:
    ----------
    :return x1,x2 : 2 Solutions of the calculated x,y,z descartes coordinates 
        of the target
    :rtype x1,x2  : 1x3 float numpy arrays 

    None: Incorrect input parameters
    """
    
    if ioo_coords.shape[0] != rb_vec.shape[0]:
        print("ERROR: The number of bistatic range measurements and the number of IoO should match")
        return None
    
    # Calculate the S_ auxiliary matrix
    S  = ioo_coords
    S_ = np.linalg.inv(S.T @ S) @ S.T

    # Calculate baseline distances for each illuminators
    L = np.array([np.sqrt(ioo_coords[i,:] @ ioo_coords[i,:].T) for i in range(ioo_coords.shape[0])])

    # Calculate the range sum vector (baseline corrected bistatic ranges)
    r = rb_vec + L

    # Prepare auxiliary vectors
    z = 1/2 *( L**2 - r**2 )

    a = S_ @ z
    b = S_ @ r

    Rt1,Rt2 = solve_quadratic_eq( a = (b.T@b-1),
                                  b =  2*a.T@b,
                                  c = (a.T@a))
    x1 = S_ @ (z + r*Rt1)
    x2 = S_ @ (z + r*Rt2)
    
    return x1, x2
