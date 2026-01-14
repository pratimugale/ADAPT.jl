# TetrisADAPT.jl

This repository is a library that contains the implementation of the optimized TETRIS-ADAPT-VQE algorithm that uses KaMIS to get optimal set of tiling operators by solving the Maximum Weight Independent Set Problem. 

## Installation

1. Create a new python virtual environment, and activate it. Eg. `python3 -m venv venv && source venv/bin/activate`
2. Set the JULIA_PYTHON environment variable to path where the python binary for the virtual environment is `export JULIA_PYTHON="<path to python binary, eg. root_path/venv/bin/python>"`
3. Install the required python packages (Qiskit related), and the Julia package using `make install`
4. Run `make smoke` to execute a simplistic example of maxcut_qaoa that uses vanilla adapt and uses the qiskit interface to validate the installation.
5. 


## TODO:
1. Unskip docs step in CI