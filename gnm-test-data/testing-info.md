# Testing data

The Python implementation implemented here is to be tested against the MATLAB Brain Toolbox implementation.
As the model is probablistic and seeds are not transferable across platforms, here we ran simulation to check that in the long-run, we get similar results.

Therefore, the MATLAB model was run using the following paramter:
* m = 10
* n = 10,000
* eta = 3
* gamma =3
* A = 10 by 10 zeros -> no seed connections