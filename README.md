# CompBio Dissertation: Testing the robustness of GNMs for capturing patterns of spontaneous activity in developing neural circuits

## Generative network models (GNMs)
Python implementation of thirteen different generative network building rules available in [GNMs](./gnm/).
A more detailed description of the individual models is available in [GNM docs](./gnm/gnm-docs.md).

## Spike time tiling coefficient 
C implementation packaged as python package located in [Spike Correlation](./spike_corr/).
Could either directly run the binary but was created on Mac, or build the package by running the *setup.py* file in the same folder.
Need to make sure to include below headers for compilation and point at the header files that come with the versions specified in the conda environment (see below).
```C
#include <Python.h>
#include <numpy/arrayobject.h>
```

## Application to Cortex and Hippocampus data
We run the generative models on microelectrode array data from the cortex and hippocampus respectively, the data is available in this [GitHub repo](https://github.com/sje30/g2chvcdata). 
Code to load the spike data is available in [G2C data](./g2c_data) (g2c = genes to cognition).

## Reproducibility
All scripts should run in the conda environment as specified in the [requirements file](./req.txt).
When conda is installed, the below command should reconstruct the environment including all package versions used here.
```console
$ conda create --name <PUT YOUR EVN NAME HERE> --file ./req.txt
```
You should then be able to activate the environment:
```console
$ conda activate <PUT YOUR EVN NAME HERE>
```

## Initial project proposal
Generative modelling techniques have been used to summarise pairwise connectivity patterns in recordings of spontaneous neural activity [1].
In the original paper, homophilic wiring principles best accounted for the network topologies inferred from experimental recordings.
One limitation of this current approach is that it is unclear how robust the
methods are when given relatively small recording durations (e.g. 5
minutes).

The plan in this project will be to test the robustness of networks created by subsampling from longer duration recordings (e.g. taking 10 min segments from the 30 min recordings) to see the variability of the networks that are generated.
This will effectively test the reliability of the STTC (Spike Time Tiling Coefficient) when presented with shorter recordings.

Two further aspects of this project to explore:

1.  How well can the method recreate synthetic networks where we know the structure of the network generating the activity?  (e.g. using the Izhikevich model for spiking networks.)

2. One (currently) unpublished critique of our STTC method is that it cannot generate -1 for 'perfectly anticorrelated' spike trains.  
How well does our STTC method work with anticorrelated spike trains, and how might it affect network generation?


## Reference

Akarca D, Dunn AWE, Hornauer PJ, Ronchi S, Fiscella M, Wang C, Terrigno
M, Jagasia R, Vértes PE, Mierau SB, Paulsen O, Eglen SJ, Hierlemann A,
Astle DE, Schröter M (2022) Homophilic wiring principles underpin
neuronal network topology in vitro. bioRxiv:2022.03.09.483605 