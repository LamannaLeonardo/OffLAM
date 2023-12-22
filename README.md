# Lifted Action Models Learning from Partial Traces

This repository contains the official code of the Offline Learning of Action Models (OffLAM) algorithm. The learned models used in the paper experimental evaluation can be downloaded from this [link](https://drive.google.com/file/d/1eh51H8lPrUGTRPoDKLH8eOFl7RWg12qu/view?usp=share_link).  


## Installation
The following instructions have been tested on macOS Ventura 13.3.1


1. Clone this repository:
```
 git clone https://github.com/LamannaLeonardo/OffLAM.git
```

2. Create a Python 3.9 virtual environment using conda or pip:
```
 conda create -n offlam python=3.9
```

3. Activate the environment:
```
 conda activate offlam
```

4. Install dependencies:
```
pip install numpy pandas
```

7. Download the validator [VAL](https://github.com/KCL-Planning/VAL) binary files into the directory "Util/Grounding", the files already provided in this repository work with macOS.


8. Check everything is correctly installed by running "main.py" script.


## Execution

### Running OffLAM
The OffLAM algorithm can be run for learning from traces with partially observable states, partially observable actions, and partially observable states and actions.  
The experiment can be changed by setting the "EXP" variable in "Configuration.py".

e.g. to run OffLAM on the task of learning from traces with partial states, set `EXP = PARTIAL_STATE_EXP` in "Configuration.py"


### Log and results
When you execute OffLAM, a new directory with all logs and results is created in the "Results/10 traces/OffLAM" folder. For instance, when you run OffLAM for the first time, the logs and results are stored in the folder "Results/10 traces/OffLAM/EXP/run0". For each considered domain (e.g. blocksworld), a subdirectory is created (e.g. "Results/10 traces/OffLAM/EXP/run0/blocksworld"), which consists of a log file and learned action model for each degree of observability in [0.1, 1].


## OffLAM traces
For every domain and observability degree considered in the paper, the set of traces generated for evaluating OffLAM can be downloaded from this [link](https://drive.google.com/file/d/1kPkH07RR9TJEoMWkwImBYm0irYEj9oLg/view?usp=share_link)


## License
This project is licensed under the MIT License - see the [LICENSE](/License) file for details.
