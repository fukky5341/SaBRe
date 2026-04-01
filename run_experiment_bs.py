from max_binary_search import (perform_binary_search_acasxu, 
                               perform_binary_search_mnistConv, 
                               perform_binary_search_mnist4,
                               perform_binary_search_cifar)
import sys
from math import pi


def run_exp(dataset, net_idx1=None, net_idx2=None, RSIS_mode_list=None, time_budget=None, exe_start=0, exe_end=10, d_eps=None, i_eps=None):
    if dataset == "cifar":
        time_budget = 18000
        time_budget_for_one = 4000
        ini_d_eps = 8
        ini_i_eps = 8
        max_iter = 100
        perform_binary_search_cifar(d_eps=d_eps, i_eps=i_eps, ini_d_eps=ini_d_eps, ini_i_eps=ini_i_eps, RSIS_mode_list=RSIS_mode_list,
                                   exe_start=exe_start, exe_end=exe_end, time_budget=time_budget, bs_max_iter=max_iter, time_budget_for_one=time_budget_for_one)
    elif dataset == "mnistConv":
        time_budget = 3600
        time_budget_for_one = 800
        ini_d_eps = 12
        ini_i_eps = 12
        max_iter = 100
        perform_binary_search_mnistConv(d_eps=d_eps, i_eps=i_eps, ini_d_eps=ini_d_eps, ini_i_eps=ini_i_eps, RSIS_mode_list=RSIS_mode_list,
                                      exe_start=exe_start, exe_end=exe_end, time_budget=time_budget, bs_max_iter=max_iter, time_budget_for_one=time_budget_for_one)
    elif dataset == "mnist4":
        time_budget = 2700
        time_budget_for_one = 600
        ini_d_eps = 12
        ini_i_eps = 12
        max_iter = 100
        perform_binary_search_mnist4(d_eps=d_eps, i_eps=i_eps, ini_d_eps=ini_d_eps, ini_i_eps=ini_i_eps, RSIS_mode_list=RSIS_mode_list,
                                   exe_start=exe_start, exe_end=exe_end, time_budget=time_budget, bs_max_iter=max_iter, time_budget_for_one=time_budget_for_one)
    elif dataset == "acasxu":
        time_budget = 1200
        time_budget_for_one = 420
        max_iter = 100
        ini_d_eps = 1
        perform_binary_search_acasxu(net_idx1=net_idx1, net_idx2=net_idx2, ini_d_eps=ini_d_eps, RSIS_mode_list=RSIS_mode_list,
                                  exe_start=exe_start, exe_end=exe_end, time_budget=time_budget, bs_max_iter=max_iter, time_budget_for_one=time_budget_for_one)
    else:
        print("Invalid dataset name. Use 'acasxu', 'mnistConv', 'mnist4', or 'cifar'.")
        sys.exit(1)


# cifar
cifar_bs_experiment_map = {
    0: (2, 3),  # d_eps=2, i_eps=3
    1: (2, 4)  # d_eps=2, i_eps=4
}
d_eps, i_eps = cifar_bs_experiment_map[0]
rsis_mode_list = ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']
exe_start = 0
exe_end = 16  # run [0, 1, ..., 15]
print("** Run cifar **")
run_exp("cifar", RSIS_mode_list=rsis_mode_list, d_eps=d_eps, i_eps=i_eps, exe_start=exe_start, exe_end=exe_end)  # run [0, 1, ..., 10]


# mnistConv
mnistConv_bs_experiment_map = {
    0: (2, 4),  # d_eps=3, i_eps=4
    1: (3, 3),  # d_eps=3, i_eps=4
    2: (3, 4),  # d_eps=3, i_eps=4
}
d_eps, i_eps = mnistConv_bs_experiment_map[0]
rsis_mode_list = ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']
exe_start = 0
exe_end = 13  # run [0, 1, ..., 12]
print("** Run mnistConv **")
run_exp("mnistConv", RSIS_mode_list=rsis_mode_list, d_eps=d_eps, i_eps=i_eps, exe_start=exe_start, exe_end=exe_end)  # run [0, 1, ..., 12]


# mnist4
mnist4_bs_experiment_map = {
    0: (2, 4),  # d_eps=3, i_eps=4
    1: (3, 3),  # d_eps=3, i_eps=4
    2: (3, 4),  # d_eps=3, i_eps=4
}
d_eps, i_eps = mnist4_bs_experiment_map[0]
rsis_mode_list = ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']
exe_start = 0
exe_end = 13  # run [0, 1, ..., 12]
print("** Run mnist4 **")
run_exp("mnist4", RSIS_mode_list=rsis_mode_list, d_eps=d_eps, i_eps=i_eps, exe_start=exe_start, exe_end=exe_end)  # run [0, 1, ..., 12]


# acasxu
acasxu_bs_experiment_map = {
    0: (1, 1),  # net_idx1=1, net_idx2=1
    1: (1, 2),  # net_idx1=1, net_idx2=2
    2: (1, 3),  # net_idx1=1, net_idx2=3
    3: (1, 4),   # net_idx1=1, net_idx2=4
    4: (1, 5),   # net_idx1=1, net_idx2=5
}
net_idx1, net_idx2 = acasxu_bs_experiment_map[0]
rsis_mode_list = ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']
print(f"Running experiments with net_idx1={net_idx1}, net_idx2={net_idx2}")
# acasxu
print("** Run acasxu **")
run_exp("acasxu", RSIS_mode_list=rsis_mode_list, net_idx1=net_idx1, net_idx2=net_idx2)


print("Done!")
