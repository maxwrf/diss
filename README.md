# CompBio Dissertation: Testing the robustness of GNMs for capturing patterns of spontaneous activity in developing neural circuits

## Generative network models (GNMs)
Python implementation of thirteen different generative network building rules available in [GNMs](./gnm/).
A more detailed description of the individual models is available in [GNM docs](./gnm/gnm-docs.md).

The models were previously implemented as part of the Matlab [Brain Connectivity Toolbox](https://sites.google.com/site/bctnet/).

### Testing the GNM implementation
The model will build the network probabilistically, such that there is no straightforward way to put the models to test.
I have run 10,000 iterations of the model in MatLab (data available in [GNM testing data](./gnm-test-data/)) and my Python implementation and compared wether the same connections were build with the same frequency, which I can confirm.

## Spike time tiling coefficient  (STTC)
C implementation packaged as python package located in [Spike Correlation](./spike_corr/).
Could either directly run the binary but was created on Mac, or build the package by running the *setup.py* file in the same folder.
Need to make sure to include below headers for compilation and point at the header files that come with the versions specified in the conda environment (see below).
```C
#include <Python.h>
#include <numpy/arrayobject.h>
```

### Testing the STTC
I have tested the C + Python implementation against the original R implementation, and can generate the same results for any test data.
Corresponding test files are available in [Spike Correlation](./spike_corr/).

## Application to Cortex and Hippocampus data
I currently try to run generative models on microelectrode array data from the cortex and hippocampus respectively, the data is available in this [GitHub repo](https://github.com/sje30/g2chvcdata). 
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
Akarca, Danyal, Alexander W. E. Dunn, Philipp J. Hornauer, Silvia Ronchi, Michele Fiscella, Congwei Wang, Marco Terrigno, et al. ‘Homophilic Wiring Principles Underpin Neuronal Network Topology in Vitro’. Preprint. Neuroscience, 10 March 2022. https://doi.org/10.1101/2022.03.09.483605.

Akarca, Danyal, Petra E. Vértes, Edward T. Bullmore, the CALM team, Kate Baker, Susan E. Gathercole, Joni Holmes, et al. ‘A Generative Network Model of Neurodevelopmental Diversity in Structural Brain Organization’. Nature Communications 12, no. 1 (9 July 2021): 4216. https://doi.org/10.1038/s41467-021-24430-z.

Charlesworth, Paul, Ellese Cotterill, Andrew Morton, Seth Gn Grant, and Stephen J Eglen. ‘Quantitative Differences in Developmental Profiles of Spontaneous Activity in Cortical and Hippocampal Cultures’. Neural Development 10, no. 1 (December 2015): 1. https://doi.org/10.1186/s13064-014-0028-0.

Cutts, Catherine S., and Stephen J. Eglen. ‘Detecting Pairwise Correlations in Spike Trains: An Objective Comparison of Methods and Application to the Study of Retinal Waves’. The Journal of Neuroscience 34, no. 43 (22 October 2014): 14288–303. https://doi.org/10.1523/JNEUROSCI.2767-14.2014.
