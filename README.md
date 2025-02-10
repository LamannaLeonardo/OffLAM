# Lifted Action Models Learning from Partial Traces
[![DOI](https://zenodo.org/badge/734644047.svg)](https://zenodo.org/badge/latestdoi/734644047)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official code of the Offline Learning of Action Models (OffLAM) algorithm. The learned models used in the paper experimental evaluation can be downloaded from this [link](https://drive.google.com/file/d/1eh51H8lPrUGTRPoDKLH8eOFl7RWg12qu/view?usp=share_link).  


## Installation
The following instructions have been tested on macOS Ventura 13.3.1


1. Clone this repository:
```
 git clone https://github.com/LamannaLeonardo/OffLAM.git
```

2. Create a Python 3.9 virtual environment using conda (or [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-a-new-virtual-environment)):
```
 conda create -n offlam python=3.9
```

3. Activate the environment with conda (or [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#activate-a-virtual-environment)):
```
 conda activate offlam
```

4. Install dependencies:
```
pip install numpy pandas matplotlib openpyxl
```

7. Download the validator [VAL](https://github.com/KCL-Planning/VAL) binary files into the directory `Util/Grounding`, the files already provided in this repository work with macOS.
Alternatively, you can compile [VAL](https://github.com/KCL-Planning/VAL) and copy into the directory `Util/Grounding` the following files: `Analyse`, `Instantiate`, `Parser` and `libVAL.so`.


10. Check everything is correctly installed by running `main.py` script.


## Execution

### Running OffLAM
The OffLAM algorithm can be run for learning from traces with partially observable states, partially observable actions, and partially observable states and actions. The experiment can be changed by setting the `EXP` variable in `Configuration.py`.

e.g. to run OffLAM on the task of learning from traces with partial states, set `EXP = PARTIAL_STATE_EXP` in `Configuration.py`


### Log and results
When you execute OffLAM, a new directory with all logs and results is created in the `Results/10 traces/OffLAM` folder. For instance, when you run OffLAM for the first time, the logs and results are stored in the folder `Results/10 traces/OffLAM/EXP/run0`. For each considered domain (e.g. blocksworld), a subdirectory is created (e.g. `Results/10 traces/OffLAM/EXP/run0/blocksworld`), which consists of a log file and learned action model for each degree of observability in [0.1, 1].


## OffLAM traces
For every domain and observability degree considered in the paper, the set of traces generated for evaluating OffLAM can be downloaded from this [link](https://zenodo.org/records/11635434).  To reproduce the paper results, download the OffLAM traces, unzip them into the directory `Analysis`, and run OffLAM.


## Custom domain learning
For running OffLAM on a custom domain (e.g. "testworld"), you need to provide an input domain file `Analysis/Benchmarks/testworld.pddl` and a set of input plan traces with one file for each plan trace in the directory `Analysis/Input traces/testworld/$EXP` where the value of `EXP` can be set in `Configuration.py` (default value corresponds to plan traces with partially observable states). The input planning domain must contain at least the predicates, object types, and operator signatures, an example of (empty) input planning domain is `Analysis/Benchmarks/testworld.pddl`. Examples of input plan traces with partial states can be found in the directory `Analysis/Input traces/testworld/partial_states`, notice that OffLAM can learn a planning domain from plan traces of different environments (e.g. it is possible to learn a planning domain from small environments and exploit the learned domain in large environments). 

To run OffLAM on the custom domain "testworld" run the command `python test.py -d testworld`


## Citations
```
@article{lamanna2024lifted,
  title={Lifted Action Models Learning from Partial Traces},
  author={Lamanna, Leonardo and Serafini, Luciano and Saetti, Alessandro and Gerevini, Alfonso and Traverso, Paolo},
  journal={Artificial Intelligence},
  volume={339},
  pages={104256},
  year={2025},
  publisher={Elsevier}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.
