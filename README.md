# MPhil in Computation Biology Dissertation: Testing the robustness of generative networks for capturing patterns of spontaneous activity in developing neural circuits

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