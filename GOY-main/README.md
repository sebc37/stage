# GOY
GOY model implementation in C, initialy written by Emmanuel Leveque, with various tweaks for IO, samplings, and reconstruction by N.B.G.

# installation
- download the full zip/tar and unzip it in some directory.
- from there, type "make".

# use
- set the parameters you want in the "parameters.h" file. It is strongly advised to keep the "parameters.h" file alongside the results, in order to keep track of the very parameters used in the simulation.
- run the code by typing "./run".
  Depending on your requirements in "parameters.h", several files will be created, containing:
     - the time evolution of the various shells  --> file "data.dat"
     - the reconstructed velocity signal  --> file "v.dat" (if you asked for the reconstruction with the option DO_RECONSTRUCT)
     - the time evolution of the flux of energy betweeen shell k and k+1 --> file "flux.dat" (if you asked for with the option DO_FLUX)
     - ...
     
# tips
- it is possible to "continue" a previous run, in order to add more points to either the time evolutions or to some stats.
  this is particularly useful after the transients have vanished.
- the tmp/ directory contains the cache to allow such "continuation" runs.
- an example file is provided to load the data in Python.
  
