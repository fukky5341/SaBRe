from nnRelationalVerify.max_binary_search import (perform_binary_search_acasxu, 
                               perform_binary_search_mnistC, 
                               perform_binary_search_mnistF,
                               perform_binary_search_cifar,
                               perform_binary_search_gtsrb)
import argparse
import sys
from math import pi
import os
from pathlib import Path


os.chdir(Path(__file__).resolve().parent / "nnRelationalVerify")

import platform
import torch

if platform.machine() in ("arm64", "aarch64"):
    torch.backends.mkldnn.enabled = False


def run_exp(dataset, net_idx1=None, net_idx2=None, RSIS_mode_list=None, time_budget=None, exe_start=0, exe_end=10, d_eps=None, i_eps=None, threshold_analysis=True):
    if dataset == "cifar":
        time_budget = 18000
        time_budget_for_one = 4000
        ini_d_eps = 8
        ini_i_eps = 8
        max_iter = 100
        perform_binary_search_cifar(d_eps=d_eps, i_eps=i_eps, ini_d_eps=ini_d_eps, ini_i_eps=ini_i_eps, RSIS_mode_list=RSIS_mode_list,
                                   exe_start=exe_start, exe_end=exe_end, time_budget=time_budget, bs_max_iter=max_iter, time_budget_for_one=time_budget_for_one)
    if dataset == "gtsrb":
        time_budget = 18000
        time_budget_for_one = 4000
        ini_d_eps = 12
        ini_i_eps = 12
        max_iter = 100
        perform_binary_search_gtsrb(d_eps=d_eps, i_eps=i_eps, ini_d_eps=ini_d_eps, ini_i_eps=ini_i_eps, RSIS_mode_list=RSIS_mode_list,
                                   exe_start=exe_start, exe_end=exe_end, time_budget=time_budget, bs_max_iter=max_iter, time_budget_for_one=time_budget_for_one, threshold_analysis=threshold_analysis)
    elif dataset == "mnistC":
        time_budget = 3600
        time_budget_for_one = 800
        ini_d_eps = 12
        ini_i_eps = 12
        max_iter = 100
        perform_binary_search_mnistC(d_eps=d_eps, i_eps=i_eps, ini_d_eps=ini_d_eps, ini_i_eps=ini_i_eps, RSIS_mode_list=RSIS_mode_list,
                                      exe_start=exe_start, exe_end=exe_end, time_budget=time_budget, bs_max_iter=max_iter, time_budget_for_one=time_budget_for_one)
    elif dataset == "mnistF":
        time_budget = 2700
        time_budget_for_one = 600
        ini_d_eps = 12
        ini_i_eps = 12
        max_iter = 100
        perform_binary_search_mnistF(d_eps=d_eps, i_eps=i_eps, ini_d_eps=ini_d_eps, ini_i_eps=ini_i_eps, RSIS_mode_list=RSIS_mode_list,
                                   exe_start=exe_start, exe_end=exe_end, time_budget=time_budget, bs_max_iter=max_iter, time_budget_for_one=time_budget_for_one)
    elif dataset == "acasxu":
        time_budget = 1200
        time_budget_for_one = 420
        max_iter = 100
        ini_d_eps = 1
        perform_binary_search_acasxu(net_idx1=net_idx1, net_idx2=net_idx2, ini_d_eps=ini_d_eps, RSIS_mode_list=RSIS_mode_list,
                                  exe_start=exe_start, exe_end=exe_end, time_budget=time_budget, bs_max_iter=max_iter, time_budget_for_one=time_budget_for_one)
    else:
        print("Invalid dataset name. Use 'gtsrb', 'acasxu', 'mnistC', 'mnistF', or 'cifar'.")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run binary search experiments for selected networks.")
    parser.add_argument(
        "--networks",
        nargs="+",
        choices=["cifar", "gtsrb", "mnistC", "mnistF", "acasxu"],
        default=["cifar", "gtsrb", "mnistC", "mnistF", "acasxu"],
        help="One or more networks to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_networks = set(args.networks)

    if "cifar" in selected_networks:
        cifar_bs_experiment_map = {
            0: (2, 3),  # d_eps=2, i_eps=3
            1: (2, 4),  # d_eps=2, i_eps=4
        }
        for key in cifar_bs_experiment_map:
            d_eps, i_eps = cifar_bs_experiment_map[key]
            rsis_mode_list = ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']
            exe_start = 0
            exe_end = 16  # run [0, 1, ..., 15]
            print("** Run cifar **")
            run_exp("cifar", RSIS_mode_list=rsis_mode_list, d_eps=d_eps, i_eps=i_eps, exe_start=exe_start, exe_end=exe_end)

    if "gtsrb" in selected_networks:
        gtsrb_bs_experiment_map = {
            0: (2, 4),
            1: (3, 3),
            2: (3, 4),
        }
        for key in gtsrb_bs_experiment_map:
            d_eps, i_eps = gtsrb_bs_experiment_map[key]
            rsis_mode_list = ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']
            exe_start = 0
            exe_end = 10  # run [0, 1, ..., 9]
            threshold_analysis = True
            print("** Run gtsrb **")
            run_exp("gtsrb", RSIS_mode_list=rsis_mode_list, d_eps=d_eps, i_eps=i_eps, exe_start=exe_start, exe_end=exe_end, threshold_analysis=threshold_analysis)

    if "mnistC" in selected_networks:
        mnistC_bs_experiment_map = {
            0: (2, 4),
            1: (3, 3),
            2: (3, 4),
        }
        for key in mnistC_bs_experiment_map:
            d_eps, i_eps = mnistC_bs_experiment_map[key]
            rsis_mode_list = ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']
            exe_start = 0
            exe_end = 13  # run [0, 1, ..., 12]
            print("** Run mnistC **")
            run_exp("mnistC", RSIS_mode_list=rsis_mode_list, d_eps=d_eps, i_eps=i_eps, exe_start=exe_start, exe_end=exe_end)

    if "mnistF" in selected_networks:
        mnistF_bs_experiment_map = {
            0: (2, 4),
            1: (3, 3),
            2: (3, 4),
        }
        for key in mnistF_bs_experiment_map:
            d_eps, i_eps = mnistF_bs_experiment_map[key]
            rsis_mode_list = ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']
            exe_start = 0
            exe_end = 13  # run [0, 1, ..., 12]
            print("** Run mnistF **")
            run_exp("mnistF", RSIS_mode_list=rsis_mode_list, d_eps=d_eps, i_eps=i_eps, exe_start=exe_start, exe_end=exe_end)

    if "acasxu" in selected_networks:
        acasxu_bs_experiment_map = {
            0: (1, 1),
            1: (1, 2),
            2: (1, 3),
            3: (1, 4),
            4: (1, 5),
        }
        for key in acasxu_bs_experiment_map:
            net_idx1, net_idx2 = acasxu_bs_experiment_map[key]
            rsis_mode_list = ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']
            print(f"Running experiments with net_idx1={net_idx1}, net_idx2={net_idx2}")
            print("** Run acasxu **")
            run_exp("acasxu", RSIS_mode_list=rsis_mode_list, net_idx1=net_idx1, net_idx2=net_idx2)

    print("Done!")


if __name__ == "__main__":
    main()
