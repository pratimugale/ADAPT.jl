# TetrisADAPT.jl

This library contains the implementation of the MosaicADAPT-QAOA algorithm. At each layer of QAOA, it calculates the gradients of all mixer operators, and is an extension of [ADAPT.jl](https://github.com/KarunyaShirali/ADAPT.jl). It selects all disjoint mixer operators at a layer and maximizes the sum of gradients of all chosen operators using KaMIS. It gets the optimal set of tiling operators by solving the Maximum Weight Independent Set Problem, where the operators are the nodes and the node weights are proportional to the gradients. 

## Installation

1. Create a new python virtual environment, and activate it. Eg. `python3 -m venv venv && source venv/bin/activate`
2. Set the JULIA_PYTHON environment variable to path where the python binary for the virtual environment is `export JULIA_PYTHON="<path to python binary, eg. root_path/venv/bin/python>"`
3. Install the required python packages (Qiskit related), and the Julia package using `make install`
4. Run `make smoke` to execute a simplistic example of maxcut_qaoa that uses vanilla adapt and uses the qiskit interface to validate the installation.
5. To clone the KaMIS repo - Run `make install-kamis`. Depending on the platform being used, KaMIS installation instructions may differ. Please see the KaMIS [installation guide](https://github.com/KarlsruheMIS/KaMIS/tree/master#installation-from-source) for this. 
6. Run `make smoke-kamis` to verify if the integration with KaMIS works. 
