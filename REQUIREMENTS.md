# Requirements

## Supported Operating Systems

The artifact has been developed and tested on:

- Ubuntu 22.04 LTS (recommended)

Other recent Linux distributions may also work but have not been extensively tested.

Windows and macOS are not officially supported.

---

## Hardware Requirements

### Tested Environment

The experiments in the paper were conducted on the following machine:

- AWS EC2 c4.2xlarge
- 8 vCPUs (Intel Xeon E5-2666 v3)
- 15 GB RAM
- At least 20 GB available disk space

### Recommended Environment

To run the smoke test and small-scale experiments:

- 4 CPU cores
- 8 GB RAM
- 10 GB available disk space

For reproducing the complete experimental results reported in the paper, we recommend hardware comparable to the tested environment.

GPU acceleration is **not required**.

---

## Software Requirements

The artifact requires:

- Python 3.12
- uv (Python package manager)
- Gurobi Optimizer 12.x (or a compatible version below 13.0)
- Docker (optional, recommended)

Python dependencies are automatically installed using:

```bash
uv sync
```

---

## Internet Access

Internet access is required only for:

- cloning the repository;
- installing Python packages;
- downloading a Gurobi license (if needed).


Internet access may also be required when using a Web License Service (WLS) Gurobi license.

---

## Expected Runtime

Neural network verification is NP-complete, and runtime depends heavily on the benchmark and verification instance.

All expected runtimes are reported in [README.md #Running Experiments](README.md#running-experiments).
