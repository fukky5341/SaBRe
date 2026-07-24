# Installation Guide from Scratch

## Clone the repository
```
git clone https://github.com/fukky5341/SaBRe.git
cd sabre
```
The full code is available in the [https://github.com/fukky5341/SaBRe](https://github.com/fukky5341/SaBRe).


## Install Gurobi (solver)

Reproducing experiments requires a Gurobi license. Please install Gurobi from the official website: [gurobi installation](https://www.gurobi.com/). Free academic licenses for students and researchers [Gurobi academic license](https://www.gurobi.com/academia/academic-program-and-licenses) are provided if needed.

Aside from the official instructions, the following steps might be helpful.

- Login to the Gurobi user portal.
- Go to the ["Licenses - Request" tab](https://portal.gurobi.com/iam/licenses/request), genearte a "WLS Academic" license if you don't have one. If you already have a "WLS Academic" license, you might get an "[LICENSES_ACADEMIC_EXISTS] Cannot create academic license as other academic licenses already exists" error.
- Go to the "Home" tab, click "Licenses - Open the WLS manager" to open the WLS manager.
- In the WLS manager, you should see a license under the "Licenses" tab. Click "extend" if it has expired (it might take some time to take effect).
- Go to the "API Keys" tab, click the "CREATE API KEY" button to create a new license, download the generated `gurobi.lic` file by following the instructions and place it at the proper location.

Before running the experiments, ensure that your Gurobi license is properly installed and gurobipy works in Python.


## Install uv (python environment manager)
Please install by following guide: [uv installation](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)

The following command might be helpful for installation:

- For macOS/Linux:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- For Windows (PowerShell):
```
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, ensure that `uv` command is available in your terminal. You might need to run the command shown in the output of the installation script.


## Setup python version
The project requires Python 3.12. Please install and pin the version using uv:
```
uv python install 3.12
cd [repository folder]
uv python pin 3.12
```


## Create uv environment and install dependencies
```
uv sync
```
This command:
- creates a virtual environment (.venv)
- installs all dependencies from `pyproject.toml`
- ensure the environment uses Python 3.12


## Install Datasets
In our experiments, we use the following datasets:
- GTSRB
- CIFAR10
- MNIST
- ACAS Xu

For all datasets, you do not need to download them separately as they have been prepared or are automatically downloaded when needed.


## Verify the Installation
Run

```bash

uv run run_experiment_rs_is.py --quicktest

```

If the installation succeeds, you should see

```
** Run mnistF quicktest **
Running quicktest with d_eps=1, i_eps=2, RS/IS mode=RS_random_Z
execution: (d:1, i:2, idx:0)

...

Done!
```

and a log file will be generated in `nnRelationalVerify/experiment_result/mnistF/RS_random_Z_threshold/d1_e2/0/log.md`.