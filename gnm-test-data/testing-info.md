# Testing data

The Python implementation implemented here is to be tested against the MATLAB Brain Toolbox implementation.
As the model is probabilistic and seeds are not transferable across platforms, here we ran simulation to check that in the long-run, we get similar results.

Therefore, the MATLAB model was run using the following parameter:
* m = 10 (target connections)
* n = 10,000 (runs)
* eta = 3 (weighting for distance matrix)
* gamma = 3 (weighting for value matrix)
* A = 10 by 10 zeros -> no seed connections
* D = 10x10 from paper (TODO: add details here)