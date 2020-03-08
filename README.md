# py A          P         R      i  L
# Advanced   Passive   Radar     Library

pyAPRiL is a python based signal processing library which implements passive radar signal processing algorithms. All the algorithms are tested and verified through real field measurement data and simulations. The corresponding references are highlited where applicable. Algorithms are researched and developed on real life DVB-T and FM
based passive radar systems.

##### The package is organized as follows:

* pyAPRiL: Main package directory
	* channelPreparation: This file contains wrapper functions for the different algorithms that aims to prepare the  reference and the sureveillance signals for the detection stage. Such as reference signal regeneration and clutter cancellation techniques.
	* clutterCancellation: It describes a huge variety of clutter cancellation techniques implemented in the space, time and the space-time domain.
	* hitProcessor: Implements a number of hit and plot processing related functions such as the CFAR and the plot extractor.
	* detector: In this file a number of implementation of the cross-correlation detector can be found.
	* docs: Contains Ipython notebook files with demonstrations.
	* test: Contains demonstration functions.

##### Installing from Python Package Index:
```python
pip install pyapril
```

##### Version history
* 1.0.0 Inital version
* 1.1.0 Automatic detection added (CA-CFAR)
* 1.2.0 Add temporal clutter filters (ECA, ECA-B, ECA-S)
* 1.3.0 Clutter filter and detection performance evaluation
* 1.4.0 Target DoA estimation
* 1.5.0 Surveillance channel beamforming
* 1.6.0 Target parameter calculator

##### Acknowledgements
This work was supported by the Microwave Remote Sensing Laboratory of BME ([Radarlab](http://radarlab.mht.bme.hu)). A special thanks of mine goes to the  RTL-SDR site ([RTL-SDR](https://www.rtl-sdr.com/)) who helped this project becoming mature with the development and testing of the Kerberos SDR.


For further information on passive radars check: [tamaspeto.com](https://tamaspeto.com)
*Tamás Pető*
*2017-2020, Hungary*



