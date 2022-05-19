import numpy as np
# For coordinate conversions
from pygeodesy import ecef
from pygeodesy import Datums
from pygeodesy.ellipsoidalVincenty import LatLon
"""

                             Python based Advanced Passive Radar Library (pyAPRiL)

                                          Target Parameter Calulcation Tools 


     Description:
     ------------
         This module contains an auxiliary function, that can be used to calulcate the observable
         bistatic parameters of a target reflection.
    
    Dependencies:
    -------------
        - numpy
        - pygeodesy
     
     Project: pyAPRIL
     Authors: Tamás Pető
     License: GNU General Public License v3 (GPLv3)

     Changelog :
         - Ver 1.0.0    : Initial version (2020 02 25)
     
 """
 
def calculate_bistatic_target_parameters(radar_lat, radar_lon, radar_ele, radar_bearing,
                                         ioo_lat, ioo_lon, ioo_ele,
                                         target_lat, target_lon, target_ele, target_speed, target_dir,
                                         wavelength):
    """
        Description:
        ------------
            This function calculates the bistatic range, bistatic Doppler frequency,
            and bearing angle of the observed target using the positions and the 
            velocities of the target and the location of the radar and the IoO.
        
        Dependencies:
        -------------
            - PyGeodesy
        
        Parameters:
        -----------
        :param radar_lat: Radar latitude coordinate [deg]
        :param radar_lon: Radar longitude coordinate [deg]
        :param radar_ele: Radar elevation [m]
        :param radar_bearing: Boresight direction of the surveillance antenna of the radar.
                              Interpreted as clockwise from the north pole [deg]  
            
        :param ioo_lat: Transmitter latitude coordinate [deg]
        :param ioo_lon: Transmitter longitude coordinate [deg]
        :param ioo_ele: Transmitter elevation [m]. For broadcast towers this is typically
                        defined as the sum of the ASL and AGL hegihts.
                        ASL: Above Seal Level
                        AGL: Above Ground Level
                        
        :param target_lat: Target latitude coordinate [deg]
        :param target_lon: Target longitude coordinate [deg]
        :param target_ele: Target altitude [m]
        :param target_speed: Target ground speed [meter per second]
        :param target_dir: Target moving direction [deg]
                           Interpreted as clockwise from the north pole [deg]   
        
        :param wavelength: Wavelength of the used IoO [m]
            
        :type radar_lat: float [-90 .. 90]
        :type radar_lon: float [-90 .. 90]
        :type radar_ele: float
        :type radar_bearing: float [0 .. 180]
        
        :type ioo_lat: float [-90 .. 90]
        :type ioo_lon: float [-90 .. 90]
        :type ioo_ele: float
        
        
        :type target_lat: float [-90 .. 90]
        :type target_lon: float [-90 .. 90]
        :type target_ele: float
        :type target_speed: flat
        :type target_dir: float [0 .. 180]
        
        :type wavelength: float
        
        Return values:
        --------------
        :return calculated target parameters:
                (target bistatic range [m],
                target Doppler frequency [Hz]],
                target azimuth bearing [deg])
        :rtype calculated target parameters: python list with 3 elements 
                [float, float, float]
    """
    
    ecef_converter = ecef.EcefYou(Datums.Sphere)

    # Convert geodetic radar position to ECEF coordinates
    radar_ecef = ecef_converter.forward(radar_lat, radar_lon, radar_ele)
    radar_ecef = np.array([radar_ecef[0],radar_ecef[1], radar_ecef[2]])
    
    # Convert geodetic IoO position to ECEF coordinates
    ioo_ecef = ecef_converter.forward(ioo_lat, ioo_lon, ioo_ele)
    ioo_ecef = np.array([ioo_ecef[0],ioo_ecef[1], ioo_ecef[2]])
    
    # Baseline distance
    L = np.sqrt(np.sum(np.abs(radar_ecef-ioo_ecef)**2)) 
    
    # Convert geodetic (lat, lon, elevation) target position to ECEF coordinates    
    target_ecef = ecef_converter.forward(target_lat, target_lon, target_ele)
    target_ecef = np.array([target_ecef[0],target_ecef[1], target_ecef[2]])

    # Target to IoO distance
    Rt = np.sqrt(np.sum(np.abs(target_ecef-ioo_ecef)**2)) 
    
    # Target to radar distance
    Rr = np.sqrt(np.sum(np.abs(target_ecef-radar_ecef)**2)) 
    
    # Bistatic distance
    Rb = Rt+Rr-L
        
    # Generate speed vector
    target_latlon       = LatLon(target_lat, target_lon) # Target initial coordinate
    speed_vector_latlon = target_latlon.destination(1, target_dir)  # Target coordinate 1m away in the current direction
    speed_vector_ecef   = ecef_converter.forward(speed_vector_latlon.lat, 
                                                 speed_vector_latlon.lon, 
                                                 target_ele)
    speed_vector_ecef   = np.array([speed_vector_ecef[0],speed_vector_ecef[1], speed_vector_ecef[2]])
    speed_vector        = speed_vector_ecef-target_ecef # Create vector in Cartesian space
    speed_vector       /= np.sqrt(np.sum(np.abs(speed_vector)**2)) # Normalize
    
    # Generate target to IoO vector
    target_to_ioo_vector  = (ioo_ecef-target_ecef)/Rt
    
    # Generate target to radar vector
    target_to_radar_vector = (radar_ecef-target_ecef)/Rr
    
    # Calculate target Doppler
    fD = 1*(target_speed / wavelength) * \
         ( (np.sum(speed_vector*target_to_ioo_vector)) + (np.sum(speed_vector*target_to_radar_vector)))
    
    # Caculate target azimuth direction 
    # Formula is originated from: https://www.movable-type.co.uk/scripts/latlong.html
    lat1 = np.deg2rad(radar_lat)
    lon1 = np.deg2rad(radar_lon)

    lat2 = np.deg2rad(target_lat)
    lon2 = np.deg2rad(target_lon)    
    
    target_doa = np.arctan2(np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1),
                         np.sin(lon2-lon1)*np.cos(lat2)) 
    
    theta =   90-np.rad2deg(target_doa) - radar_bearing + 90
    
    return (Rb, fD, theta)

