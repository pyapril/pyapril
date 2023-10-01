# py A          P         R      i  L

# Advanced   Passive   Radar     Library

pyAPRiL is a python based DSP library which implements passive radar algorithms. The ultimate goal of the library is to make available the so far ellaborated passive radar algorithms to everyone including sceintific researchers, radar system designers and amateurs. All the implemented methods are tested and verified through real-life systems, field measurements and simulations.

### Project guidelines:
- This project respects the authors of all the contributions, therfore references are always highlighted where applicable.
- Understanding the operation of the implemented algorithms is always of primary importance. The efficient execution and concise coding do not belong to the principles of the project, but encoured  when it is not at the expense of comprehensibility and does not change the essential operation of the originally proposed algorithm.
### Contributions:
Contributions are welcome from anyone who shares the passion of passive radars and wants to help make opensource passive radar community even better. Whether you're a seasoned developer or just getting started, there are plenty of ways to get involved and contribute to pyAPRiL. By contributing to the project, you'll be helping to make a positive impact on passive radar developers around the globe and advancing the deeper sceintific understanding of this novel technology. Whether you can contribute code, documentation, or simply provide feedback, every contribution helps us move closer to our shared goal.
### Restrictions:
This projects strictly focuses on the support of scientific researches. The source code of the library is licensed under the GNU General Public License version 3 (GPLv3), which is a copyleft license. This means that anyone who uses, modifies, or distributes the project must also license their work under the same terms and conditions. Commercial use of this software is permitted, provided that any results obtained from such use are made available to the public under the same license. By adhering to these restrictions, this project gains comliance with the REGULATION (EU) 2021/821 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL.

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
	    * Cross-correlation detector - Batched implementation (Overlap and save)
		* Doppler frequency windowing
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
	* **docs**: Contains Ipython notebook files with demonstrations.
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
* 1.7.1 Extend RD Tools with Hit matrix plot
* 1.7.2 Adaptive RD matrix scaling to improve the visibility of the plots
* 1.7.3 Improved generality for metric extraction
* 1.7.4 Improve and fix metric extraction
* 1.7.5 Update project organization (2023 04)
### Acknowledgements
This work was supported by the Microwave Remote Sensing Laboratory of BME ([Radarlab](http://radarlab.mht.bme.hu)). A special thanks of mine goes to the  RTL-SDR site ([RTL-SDR](https://www.rtl-sdr.com/)) and to the KrakenRF (https://www.krakenrf.com/) who helped this project becoming mature with the development and testing the algorithms using Kerberos SDR (later KrakenSDR).

For further information on passive radars check: [passiveradar.eu](https://passiveradar.eu)



