from nnRelationalVerify.experiment import execute_experiment_mnistF, execute_experiment_cifar, \
    execute_experiment_acasxu, execute_experiment_mnistC, execute_experiment_gtsrb
from pathlib import Path
import argparse
import os
import sys


os.chdir(Path(__file__).resolve().parent / "nnRelationalVerify")

import platform
import torch

if platform.machine() in ("arm64", "aarch64"):
    torch.backends.mkldnn.enabled = False

# delta_eps = 1/256 * d_eps
# eps = 1/256 * d_eps * i_eps


def run_exp(dataset, d_eps, i_eps, net_idx1=None, net_idx2=None, RS_mode=None, IS_mode=None, time=None, exe_start=0, exe_end=10, inputs_num=10):
    if dataset == "acasxu":
        time = 420
        threshold_analysis = True
        execute_experiment_acasxu(net_idx1=net_idx1, net_idx2=net_idx2, d_eps=d_eps, RS_mode=RS_mode, IS_mode=IS_mode,
                                  split_limit=100, inputs_num=inputs_num, exe_start=exe_start, exe_end=exe_end,
                                  time_budget=time, threshold_analysis=threshold_analysis)
    elif dataset == "mnistF":
        if time is None:
            time = 600
        threshold_analysis = True
        execute_experiment_mnistF(d_eps=d_eps, i_eps=i_eps, RS_mode=RS_mode, IS_mode=IS_mode, split_limit=100, exe_start=exe_start,
                                  exe_end=exe_end, inputs_num=inputs_num, time_budget=time, threshold_analysis=threshold_analysis)
    elif dataset == "mnistC":
        if time is None:
            time = 600
        threshold_analysis = True
        execute_experiment_mnistC(d_eps=d_eps, i_eps=i_eps, RS_mode=RS_mode, IS_mode=IS_mode, split_limit=100, exe_start=exe_start,
                                     exe_end=exe_end, inputs_num=inputs_num, time_budget=time, threshold_analysis=threshold_analysis)
    elif dataset == "cifar":
        threshold_analysis = True
        execute_experiment_cifar(d_eps=d_eps, i_eps=i_eps, RS_mode=RS_mode, IS_mode=IS_mode, split_limit=100, exe_start=exe_start,
                                 exe_end=exe_end, inputs_num=inputs_num, time_budget=time, threshold_analysis=threshold_analysis)
    elif dataset == "gtsrb":
        threshold_analysis = True
        execute_experiment_gtsrb(d_eps=d_eps, i_eps=i_eps, RS_mode=RS_mode, IS_mode=IS_mode, split_limit=100, exe_start=exe_start,
                                 exe_end=exe_end, inputs_num=inputs_num, time_budget=time, threshold_analysis=threshold_analysis)
    else:
        print("Invalid dataset name. Use 'mnistF', 'mnistC', 'cifar', 'gtsrb', or 'acasxu'.")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RS/IS experiments with CLI-controlled networks and input caps.")
    parser.add_argument(
        "--quicktest",
        action="store_true",
        help="Run a single mnistF instance with d=1 and i=2 across all four RS/IS methods.",
    )
    parser.add_argument(
        "--networks",
        nargs="+",
        choices=["gtsrb", "cifar", "mnistC", "mnistF", "acasxu"],
        default=["gtsrb", "cifar", "mnistC", "mnistF", "acasxu"],
        help="Networks to run. Provide one or more names.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Number of instances to run per network (capped at the number available).",
    )
    return parser.parse_args()


def resolve_exe_end(num: int | None, inputs_num: int) -> int:
    if num is None:
        return inputs_num
    if num < 1:
        raise ValueError("--num must be at least 1")
    return min(num, inputs_num)


def main() -> None:
    args = parse_args()
    selected_networks = set(args.networks)

    if args.quicktest:
        print("** Run mnistF quicktest **")
        exe_start = 0
        exe_end = 1
        inputs_num = 13
        d_val = 1
        i_val = 2
        for rsis_mode in ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']:
            print(f"Running quicktest with d_eps={d_val}, i_eps={i_val}, RS/IS mode={rsis_mode}")
            if rsis_mode.startswith('RS'):
                run_exp("mnistF", RS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)
            elif rsis_mode.startswith('IS'):
                run_exp("mnistF", IS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)
        print("Done!")
        return

    if "gtsrb" in selected_networks:
        for d_val in [1, 2, 3]:
            for i_val in [2, 3, 4]:
                for rsis_mode in ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']:
                    print(f"Running experiments with d_eps={d_val}, i_eps={i_val}, RS/IS mode={rsis_mode}")
                    print("** Run gtsrb **")
                    if d_val == 1:
                        time = 1800
                    elif d_val == 2:
                        time = 3600
                    elif d_val == 3:
                        time = 7200
                    else:
                        raise ValueError("d_val should be 1, 2, or 3")
                    exe_start = 0
                    inputs_num = 5
                    exe_end = resolve_exe_end(args.num, inputs_num)
                    if rsis_mode.startswith('RS'):
                        run_exp("gtsrb", RS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, time=time, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)
                    elif rsis_mode.startswith('IS'):
                        run_exp("gtsrb", IS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, time=time, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)

    if "cifar" in selected_networks:
        for d_val in [1, 2, 3]:
            for i_val in [2, 3, 4]:
                for rsis_mode in ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']:
                    print(f"Running experiments with d_eps={d_val}, i_eps={i_val}, RS/IS mode={rsis_mode}")
                    print("** Run cifar **")
                    if d_val == 1:
                        time = 1800
                    elif d_val == 2:
                        time = 3600
                    elif d_val == 3:
                        time = 7200
                    else:
                        raise ValueError("d_val should be 1, 2, or 3")
                    exe_start = 0
                    inputs_num = 16 if d_val in [1, 2] else 10
                    exe_end = resolve_exe_end(args.num, inputs_num)
                    if rsis_mode.startswith('RS'):
                        run_exp("cifar", RS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, time=time, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)
                    elif rsis_mode.startswith('IS'):
                        run_exp("cifar", IS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, time=time, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)

    if "mnistC" in selected_networks:
        for d_val in [1, 2, 3]:
            for i_val in [2, 3, 4]:
                for rsis_mode in ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']:
                    print(f"Running experiments with d_eps={d_val}, i_eps={i_val}, RS/IS mode={rsis_mode}")
                    print("** Run mnistC **")
                    exe_start = 0
                    inputs_num = 13
                    exe_end = resolve_exe_end(args.num, inputs_num)
                    if rsis_mode.startswith('RS'):
                        run_exp("mnistC", RS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)
                    elif rsis_mode.startswith('IS'):
                        run_exp("mnistC", IS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)

    if "mnistF" in selected_networks:
        for d_val in [1, 2, 3]:
            for i_val in [2, 3, 4]:
                for rsis_mode in ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']:
                    print(f"Running experiments with d_eps={d_val}, i_eps={i_val}, RS/IS mode={rsis_mode}")
                    print("** Run mnistF **")
                    exe_start = 0
                    inputs_num = 13
                    exe_end = resolve_exe_end(args.num, inputs_num)
                    if rsis_mode.startswith('RS'):
                        run_exp("mnistF", RS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)
                    elif rsis_mode.startswith('IS'):
                        run_exp("mnistF", IS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)

    if "acasxu" in selected_networks:
        exe_start = 0
        inputs_num = 10
        exe_end = resolve_exe_end(args.num, inputs_num)
        for d_val in [10]:
            for net_idx1 in [1, 2]:
                for net_idx2 in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    for rsis_mode in ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']:
                        print(f"Running experiments with d_eps={d_val}, net_idx1={net_idx1}, net_idx2={net_idx2}, RS/IS_mode={rsis_mode}")
                        print("** Run acasxu **")
                        if rsis_mode.startswith('RS'):
                            run_exp("acasxu", RS_mode=rsis_mode, d_eps=d_val, i_eps=None, net_idx1=net_idx1, net_idx2=net_idx2,
                                    exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)
                        elif rsis_mode.startswith('IS'):
                            run_exp("acasxu", IS_mode=rsis_mode, d_eps=d_val, i_eps=None, net_idx1=net_idx1, net_idx2=net_idx2,
                                    exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)

    print("Done!")


if __name__ == "__main__":
    main()
