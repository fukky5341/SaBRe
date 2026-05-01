# SABRE: Splitting Approximated Bounds for Relational Verification

## Table of Contents
- [Tables of Experiment Results](#tables-of-experiment-results)
- [Installation Guide](#installation-guide)
- [Example in Section III-B](#the-example-in-section-iii-b)
- [Running Experiments](#running-experiments)
- [Project Structure](#project-structure)


## Tables of Experiment Results

### Table IV: RQ2 — Comparison between relational neuron splitting and individual neuron splitting

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="3">ACAS Xu</th>
      <th colspan="3">MNIST-F</th>
      <th colspan="3">MNIST-C</th>
      <th colspan="3">CIFAR</th>
    </tr>
    <tr>
      <th>s#</th><th>p#</th><th>&Delta; t</th>
      <th>s#</th><th>p#</th><th>&Delta; t</th>
      <th>s#</th><th>p#</th><th>&Delta; t</th>
      <th>s#</th><th>p#</th><th>&Delta; t</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ClasIS</td>
      <td>12</td><td>133.9</td><td>87.51</td>
      <td>23</td><td>135.9</td><td>79.24</td>
      <td>6</td><td>32.0</td><td>89.49</td>
      <td>28</td><td>21.6</td><td>71.82</td>
    </tr>
    <tr>
      <td>DualIS</td>
      <td>9</td><td>117.3</td><td>89.46</td>
      <td>27</td><td>124.7</td><td>73.22</td>
      <td>8</td><td>33.3</td><td>90.86</td>
      <td><b>31</b></td><td>21.4</td><td>68.63</td>
    </tr>
    <tr>
      <td>SaBRe</td>
      <td><b>67</b></td><td>83.4</td><td>34.24</td>
      <td><b>54</b></td><td>59.9</td><td>30.73</td>
      <td><b>27</b></td><td>27.5</td><td>71.37</td>
      <td>23</td><td>28.3</td><td>75.03</td>
    </tr>
  </tbody>
</table>

### TABLE V: RQ2–Relational neuron splitting vs. individual neuron splitting in CIFAR

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="3">&epsilon; = 1/256</th>
      <th colspan="3">&epsilon; = 2/256</th>
      <th colspan="3">&epsilon; = 3/256</th>
    </tr>
    <tr>
      <th>s#</th><th>p#</th><th>Δt</th>
      <th>s#</th><th>p#</th><th>Δt</th>
      <th>s#</th><th>p#</th><th>Δt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ClasIS</td>
      <td>16</td><td>22.1</td><td>56.49</td>
      <td>8</td><td>22.6</td><td>77.34</td>
      <td>4</td><td>19.3</td><td>93.54</td>
    </tr>
    <tr>
      <td>DualIS</td>
      <td>16</td><td>22.3</td><td>56.71</td>
      <td>11</td><td>20.8</td><td>67.03</td>
      <td>4</td><td>21.0</td><td>90.72</td>
    </tr>
    <tr>
      <td>SaBRe</td>
      <td>5</td><td>41.2</td><td>87.03</td>
      <td>10</td><td>22.6</td><td>64.55</td>
      <td>8</td><td>15.3</td><td>74.48</td>
    </tr>
  </tbody>
</table>

### TABLE VI: RQ3–Comparison of neuron selection SABRE v.s. RandRS

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="3">ACAS Xu</th>
      <th colspan="3">MNIST-F</th>
      <th colspan="3">MNIST-C</th>
      <th colspan="3">CIFAR</th>
    </tr>
    <tr>
      <th>s#</th><th>p#</th><th>Δt</th>
      <th>s#</th><th>p#</th><th>Δt</th>
      <th>s#</th><th>p#</th><th>Δt</th>
      <th>s#</th><th>p#</th><th>Δt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>RandRS</td>
      <td>44</td><td>199.2</td><td>60.35</td>
      <td>44</td><td>117.6</td><td>48.11</td>
      <td>11</td><td>38.7</td><td>90.43</td>
      <td>8</td><td>24.4</td><td>87.17</td>
    </tr>
    <tr>
      <td>SaBRe</td>
      <td><b>67</b></td><td>100.8</td><td>34.24</td>
      <td><b>54</b></td><td>67.3</td><td>30.73</td>
      <td><b>27</b></td><td>28.2</td><td>71.37</td>
      <td><b>23</b></td><td>18.0</td><td>75.03</td>
    </tr>
  </tbody>
</table>


## Installation Guide

### 1. Clone the repository
```
git clone https://github.com/fukky5341/SaBRe.git
cd sabre
```
The full code is available in the [https://github.com/fukky5341/SaBRe](https://github.com/fukky5341/SaBRe).

### 2. Install Gurobi (solver)


Reproducing experiments requires a Gurobi license. Please install Gurobi from the official website: [gurobi installation](https://www.gurobi.com/). Free academic licenses for students and researchers [Gurobi academic license](https://www.gurobi.com/academia/academic-program-and-licenses) are provided if needed.

Aside from the official instructions, the following steps might be helpful.

- Login to the Gurobi user portal.
- Go to the ["Licenses - Request" tab](https://portal.gurobi.com/iam/licenses/request), genearte a "WLS Academic" license if you don't have one. If you already have a "WLS Academic" license, you might get an "[LICENSES_ACADEMIC_EXISTS] Cannot create academic license as other academic licenses already exists" error.
- Go to the "Home" tab, click "Licenses - Open the WLS manager" to open the WLS manager.
- In the WLS manager, you should see a license under the "Licenses" tab. Click "extend" if it has expired (it might take some time to take effect).
- Go to the "API Keys" tab, click the "CREATE API KEY" button to create a new license, download the generated `gurobi.lic` file by following the instructions and place it at the proper location.

Before running the experiments, ensure that your Gurobi license is properly installed and gurobipy works in Python.


### 3. Install uv (python environment manager)
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


### 4. Setup python version
The project requires Python 3.12. Please install and pin the version using uv:
```
uv python install 3.12
cd [repository folder]
uv python pin 3.12
```

### 5. Create uv environment and install dependencies
```
uv sync
```
This command:
- creates a virtual environment (.venv)
- installs all dependencies from `pyproject.toml`
- ensure the environment uses Python 3.12



## Example in Section III-B
The details of the example in Section III-B is provided in [example](example/example.ipynb). You can run the notebook to visualize the bounds and splitting process.

<figure>
    <img src="example/bounds.png" alt="Relational backsubstitution example" width="400">
    <figcaption>Figure 1: Relational backsubstitution in example 1 in Section III-B.</figcaption>
</figure>
<figure>
    <img src="example/splitting.png" alt="Relational splitting example" width="600">
    <figcaption>Figure 2: Individual and Relational splitting comparison in example 1 in Section III-B.</figcaption>
</figure>



## Running Experiments
### Binary Search (RQ4)
To run the binary search experiments:
```
uv run run_experiment_bs.py
```
In this experiment, we compare the performance of our method SABRE (RS_dual_Z) with baselines: RaVeN (base), ClasIS (IS_dual_ind), DualIS (IS_dual), and RandRS (RS_random_Z) via binary search on ACAS Xu, MNIST-F, MNIST-C, CIFAR. In binary search, each approach explores the maximum verifiable input relational distance.

The results and logs are generated in `experiment_results/binary_search`. The processing status are written to the log files, and the final result is given at the bottom of the log file.

### RQ1-RQ3
To run the experiments used in RQ1-RQ3:
```
uv run run_experiment_rs_is.py
```
In this experiment, we compare the performance of our method SABRE (RS_dual_Z) with baselines: RaVeN (base), ClasIS (IS_dual_ind), DualIS (IS_dual), and RandRS (RS_random_Z) on ACAS Xu, MNIST-F, MNIST-C, CIFAR. For a given instance with output relational threshold, we evaluate whether each approach can verify or find counterexamples for the instance within the time limit.

The results and logs are generated in `experiment_results/` network-wisely. The processing status are written to the log files, and the final result is given at the bottom of the log file.


## Project Structure
```
sabre/
 ├─ run_experiment_bs.py    # Entry point for binary search experiments
 ├─ run_experiment_rs_is.py    # Entry point for experiments
 ├─ example/    # Example used in Section III-B
 ├─ relational_bounds/    # Relational bound propagation modules
 ├─ relu/    # Handle ReLU transformation in relational bound propagation
 ├─ relational_split/    # Branch-and-bound with relational splitting
 ├─ individual_split/    # Branch-and-bound with individual splitting
 ├─ relational_property/    # LP formulation for relational properties
 ├─ dual/    # Dual formulation for neuron selection
 ├─ (common, data, network_converters, ...)/  # Common utilities, datasets, and network converters
 ├─ pyproject.toml    # Project dependencies
 └─ README.md
```