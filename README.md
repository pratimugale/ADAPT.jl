# MosaicADAPT-QAOA

This library contains the implementation of the MosaicADAPT-QAOA algorithm introduced in [Q3SAT-GPT: A Generative Model for Discovering Quantum Circuits for the 3-SAT Problem](https://arxiv.org/abs/2604.27324). The repo is a fork of [ADAPT.jl](https://github.com/KarunyaShirali/ADAPT.jl).

## Method

At each layer of QAOA, MosaicADAPT-QAOA calculates the gradients of all mixer operators like ADAPT-QAOA. However, instead of selecting just one operator at each layer, it selects all disjoint mixer operators at a layer. MosaicADAPT-QAOA formulates disjoint operator selection as a maximum weight independent set problem on a specially constructed graph, where nodes represent mixer operators and edges connect operators with overlappting support. Then, MosaicADAPT-QAOA uses the [KaMIS MMWIS](https://github.com/KarlsruheMIS/KaMIS) solver to solve this problem.

Another method to select operators is using a greedy strategy (similar to [TETRIS-ADAPT-VQE](https://arxiv.org/abs/2209.10562)). We apply the same strategy to QAOA, and refer to this strategy as TETRIS-QAOA for comparison.

![Operator selection strategies](assets/images/operator_selection_strategies.png)
![Incompatibility graph](assets/images/incompatibility_graph.png)

## Installation

1. Create a new python virtual environment, and activate it. Eg. `python3 -m venv venv && source venv/bin/activate`
2. Set the JULIA_PYTHON environment variable to path where the python binary for the virtual environment is `export JULIA_PYTHON="<path to python binary, eg. root_path/venv/bin/python>"`
3. Install the required python packages (Qiskit related), and the Julia package using `make install`
4. Run `make smoke` to execute a simplistic example that runs ADAPT-QAOA. It validates the installation of the library.
5. To clone the KaMIS repo - Run make install-kamis. Depending on the platform being used, KaMIS installation instructions may differ. Please see the [KaMIS installation guide](https://github.com/KarlsruheMIS/KaMIS/tree/master#installation-from-source) for this.
6. Run make smoke-kamis to verify if the integration of this library with KaMIS works.

## Usage 

To run the MosaicADAPT-QAOA, TETRIS-QAOA variants, use the following commands:
1. TETRIS-QAOA (greedy selection): `make run-tetris-qaoa`
2. MosaicADAPT-QAOA (MWIS-based selection): `make run-mosaic-qaoa`

## Citation

If you found our work useful, please cite [arXiv preprint](https://arxiv.org/abs/2604.27324):
```
@misc{ugale2026q3satgptgenerativemodeldiscovering,
      title={Q3SAT-GPT: A Generative Model for Discovering Quantum Circuits for the 3-SAT Problem}, 
      author={Pratim Ugale and Ilya Tyagin and Karunya Shirali and Kien X. Nguyen and Ilya Safro},
      year={2026},
      eprint={2604.27324},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2604.27324}, 
}
```
