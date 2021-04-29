# py A          P         R      i  L

# Advanced   Passive   Radar     Library

pyAPRiL is a python based signal processing library which implements passive radar signal processing algorithms. All the algorithms are tested and verified through real field measurement data and simulations. The corresponding references are highlited where applicable. Algorithms are researched and developed on real life DVB-T and FM
based passive radar systems.

### The package is organized as follows:

* pyAPRiL: Main package directory
	* **channelPreparation**: This file contains wrapper functions for the different algorithms that aims to prepare the  reference and the sureveillance signals for the detection stage. Such as reference signal regeneration and clutter cancellation techniques.
	* **clutterCancellation** : It describes a huge variety of clutter cancellation techniques implemented in the space, time and the space-time domain. In the current version, the following algorithms are available:
	    * Timde-domain: 
	        * Wiener-SMI: Wiener filter with Sample Matrix Inversion
	        * Wiener-SMI-MRE: Wiener filter with Sample Matrix Inversion and Minimum Redundancy Estimation
	        * ECA: Extensive Cancellation Algorithm
	        * ECA-B: Extensive Cancellation Algorithm - Batched
	        * ECA-S: Extensive Cancellation Algorithm - Sliding
	    * Space-domain (Beamforming):
	        *    Max-SIR: Maximum Signal-to-Interference Ratio
	        *    MVDR: Minimum Variance Distortionless Response
	        *    Principal eigenvalue beaformer
	        *    Beamstearing
	        *    Explicit coefficient beamformer
	* **hitProcessor**: Implements a number of hit and plot processing related functions such as the CFAR and the plot extractor. Implemented algorithms:
	    * CA-CFAR
	    *  Target DoA estimator (from multichannel RD matrix using the pyArgus library)
	* **detector** : In this file a number of implementation of the cross-correlation detector can be found.
	    * Cross-correlation detector - Time domain implementation
	    * Cross-correlation detector - Frequency domain implementation
	    * Cross-correlation detector - Overlap and save implementation
	* **metricExtract**: Implements various clutter cancellation and detection performance evaluation function. The supported metrics are the followings:
	    * CA: Clutter Attenuation
	    * Rnf: Noise floor reduction
	    * Mu : Peak-to-channel power ratio
	    * Delta: Estimated target SINR
	    * Alpha: Target peak to noise floor ratio
	    * L: Target peak power loss
	    * Rzdc: Zero Doppler clutter reduction
	    * P: Peak clutter reduction
	    * D: Dynamic range compression
	* **targetParameterCalculator**: Calculates the expected parameters of an observed target (bistatic range, Doppler, ...) from geodetic data.
	* **RDTools**: range-Doppler matrix image export and plotting tools
	* **sim**: IoO simulator (FM)
	* **docs**: Contains Ipython notebook files with demonstrations. (Currently not available)
	* **testing**: Contains demonstration functions.

### Installing from Python Package Index:
```python
pip install pyapril
```

### Version history
* 1.0.0 Inital version
* 1.1.0 Automatic detection added (CA-CFAR)
* 1.2.0 Add temporal clutter filters (ECA, ECA-B, ECA-S)
* 1.3.0 Clutter filter and detection performance evaluation
* 1.4.0 Target DoA estimation
* 1.5.0 Surveillance channel beamforming
* 1.6.0 Target parameter calculator
* 1.7.0 FM IoO simulator
### Acknowledgements
This work was supported by the Microwave Remote Sensing Laboratory of BME ([Radarlab](http://radarlab.mht.bme.hu)). A special thanks of mine goes to the  RTL-SDR site ([RTL-SDR](https://www.rtl-sdr.com/)) who helped this project becoming mature with the development and testing of the Kerberos SDR.


For further information on passive radars check: [tamaspeto.com](https://tamaspeto.com)
*Tamás Pető*
*2017-2021, Hungary*



