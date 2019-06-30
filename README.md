# M2_Internship

In this repository is shared the work done during an M2 internship at the IPNL. The subject was the "Modelisation of slitless spectroscopic observations from the Auxiliary Telescope within the framework of the Large Synaptic Survey Telescope". 

The files contain the code made to model the 3 effects effects considered here (i.e. the defocus, the atmospheric differential refraction and the effect of the atmospheric turbulence), the report of the internship, the slides used for the presentation and the docs used in the internship.

In 1_Simulations is shown the heart of the work I did.
	In 1_ADR there are codes to plot the ADR from a .fits file, to take all headers from a fits file and put it into a json file and to plot the ADR from the json created. All this in order to build a code which I included in the library "LSST/Spectractor" which is shown in the "spectractor" folder. 
	In 2_Atmosphere are realisations of the atmosphere with HCIPy and Soapy but as it was too slow, I did plot directly the result from the Fourier transform and from an Hankel transform. Then I compared them as shown.
	Finally in 3_Defocus are shown attempt to model the defocus with HCIPy, POPPY and their comparison.

In 2_docs are shown useful docs I used and the other docs are the slides used for the presentation and the internship report. 


All the steps of the work done here are not shown as it was done on gitlab and not on github.
