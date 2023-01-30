# Code for _HOHo-QAOA: Hamiltonian-Oriented Homotopy QAOA_

This is a repository for the code used to calculate the numerical results presented in the article "*HOHo-QAOA: Hamiltonian-Oriented Homotopy QAOA*" .

## Software installation

The code was used on Ubuntu OS 20.04. It is not guarantee it will work on other operating systems

### Anaconda

Anaconda distribution can be downloaded from https://www.anaconda.com/products/individual.

For this project, we created a conda environment with the required python packages. Type the following command on the terminal at the root project folder to install it:

    conda env create -f mc-qaoa.yml

To activate the environment:

    conda activate mc-qaoa 

### Julia

Julia can be downloaded from the official website https://julialang.org/. The version 1.6.2 was used.

In order to set up the environment, please go to the directory `src/qaoa_efficiency_analysis`. Then run julia and use

```
julia> ]
(@v1.6) activate .
(@v1.6) instantiate
```
This will install the required packages based on _Manifest.toml_ file. The environment will be activated on its own when running scripts.

## Reproducing data

### Objective Hamiltonian Generation


To generate the objective Hamiltonians, run the following script on the `scr` directory:

    python hamiltonian_generate.py

This script generates 100 samples for each node value. The value of nodes are ranged from 6 to 18 with increment step equal to 2. The graph objects are saved in `compare_data/graph/` and the objective Hamiltonians in `compare_data/obj/`.

### Comparing energy evaluation

We compare the true energy based on total Hamiltonian eigenvalues for 10 nodes running the following script:

    python true_energy.py

This script calculates the energy for different initial $\alpha$ values and step. The data is saved on `compare_data/data/`. 

To calculate the `QAOA` and `HOHo-QAOA` energies, run the the following commands on the julia scripts folder `qaoa_efficiency_analysis`:

    for VAR in {2..18..2}; do julia sparse_generator_preparation.jl $VAR; done

This script prepares the sparse matrices used for the optimization. Then:

    julia maxcut_qaoa_experiment.jl


To evaluate the data generated in `true_energy.py` and `maxcut_qaoa_experiment.jl`, run the script:

    python plot_data_generate.py

And plot the results with:

    python plot.py

### QAOA energy landscape

We generate and plot data for energy landscape for QAOA with the following command:

    python qaoa_energy_landscape_plot.py

### QAOA methods comparison

To run the optimization experiments for `QAOA`, `T-QAOA` and `HOHo-QAOA`, use the following command:

    julia maxcut_multi_qaoa_experiment.jl 

The data is saved by default in `"scr/compare_data/data/<graph_type>/data_<date_of_experiment>`. Alternatively, you can pass an argument for saving the data in the a directory giving its path:

    julia maxcut_multi_qaoa_experiment.jl -outnew <path/to/directory>

If the a result in the directory already exists, it will be skipped. 

For the paper, we saved the experiment esults on the following directories:

```
scr/compare_data/data/barabasi-albert/5_to_100_layers
scr/compare_data/data/barabasi-albert/t-vs-qaoa
scr/compare_data/data/barabasi-albert/t-zero-rand
scr/compare_data/data/barabasi-albert/5_to_100_layers
```
and the plots were generated with:

    julia plotting_code.jl

The plots will be saved to `scr/plot`.
