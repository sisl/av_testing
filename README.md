# Autonomous Vehicle Testing
Code repository for "Optimal Testing of Self-Driving Cars" available (here). The following files can be found in this repository:
* ```finite_horizon.jl```: Main file for defining a finite-horizon vehicle testing problem and solving for policies.
* ```param_sweep.jl```: File that defines problems with wide array of parameters, solves them, then stores the policies.
* ```ViewResults.ipynb```: Notebook that loads policies and plots them. If you clone this repository and open the notebook, you can interact with many of the plots and vary problem parameters.

Running the notebook requires access to the file ```param_sweep_full.jld```. You can either run ```param_sweep.jl``` to generate the file yourself or download the file from the release associated with this repository.


