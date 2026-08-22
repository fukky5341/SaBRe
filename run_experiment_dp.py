from nnRelationalVerify.experiment import execute_experiment_mnistF
import sys
import os
from pathlib import Path


os.chdir(Path(__file__).resolve().parent / "nnRelationalVerify")

import platform
import torch

if platform.machine() in ("arm64", "aarch64"):
    torch.backends.mkldnn.enabled = False

# delta_eps = 1/256 * d_eps
# eps = 1/256 * d_eps * i_eps


def run_exp(dataset, d_eps, i_eps, net_idx1=None, net_idx2=None, RS_mode=None, IS_mode=None, time=None, exe_start=0, exe_end=10, inputs_num=10, perturb_ratio=None):
    if dataset == "mnistF":
        if time is None:
            time = 600
        threshold_analysis = True
        execute_experiment_mnistF(d_eps=d_eps, i_eps=i_eps, RS_mode=RS_mode, IS_mode=IS_mode, split_limit=100, exe_start=exe_start,
                                  exe_end=exe_end, inputs_num=inputs_num, time_budget=time, threshold_analysis=threshold_analysis,
                                  dimensional_perturbation=True, perturb_ratio=perturb_ratio
                                  )
    else:
        print("Invalid dataset name. Use 'mnistF'.")
        sys.exit(1)


def main():
    for d_val in [1, 2, 3]:
        for i_val in [2, 3, 4]:
            for perturb_ratio in [1.0, 0.5, 0.25, 0.125]:
                for rsis_mode in ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']:
                    print(f"Running experiments with d_eps={d_val}, i_eps={i_val}, RS/IS mode={rsis_mode}")

                    # mnistF
                    print("** Run mnistF dp **")
                    exe_start = 0
                    exe_end = 5  # run [0, 1, ..., 12]
                    inputs_num = 5
                    if rsis_mode.startswith('RS'):
                        run_exp("mnistF", RS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num, perturb_ratio=perturb_ratio)  # run [0, 1, ..., 12]
                    elif rsis_mode.startswith('IS'):
                        run_exp("mnistF", IS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num, perturb_ratio=perturb_ratio)  # run [0, 1, ..., 12]


    print("Done!")


if __name__ == "__main__":
    main()
