from experiment import execute_experiment_mnist4, execute_experiment_cifar, \
    execute_experiment_acasxu, execute_experiment_mnistConv, execute_experiment_gtsrb
import sys

# delta_eps = 1/256 * d_eps
# eps = 1/256 * d_eps * i_eps


def run_exp(dataset, d_eps, i_eps, net_idx1=None, net_idx2=None, RS_mode=None, IS_mode=None, time=None, exe_start=0, exe_end=10, inputs_num=10, perturb_ratio=None):
    if dataset == "acasxu":
        time = 420
        threshold_analysis = True
        execute_experiment_acasxu(net_idx1=net_idx1, net_idx2=net_idx2, d_eps=d_eps, RS_mode=RS_mode, IS_mode=IS_mode,
                                  split_limit=100, exe_start=exe_start, exe_end=exe_end, time_budget=time, threshold_analysis=threshold_analysis)
    elif dataset == "mnist4":
        if time is None:
            time = 600
        threshold_analysis = True
        execute_experiment_mnist4(d_eps=d_eps, i_eps=i_eps, RS_mode=RS_mode, IS_mode=IS_mode, split_limit=100, exe_start=exe_start,
                                  exe_end=exe_end, inputs_num=inputs_num, time_budget=time, threshold_analysis=threshold_analysis,
                                  dimensional_perturbation=True, perturb_ratio=perturb_ratio
                                  )
    elif dataset == "mnistConv":
        if time is None:
            time = 600
        threshold_analysis = True
        execute_experiment_mnistConv(d_eps=d_eps, i_eps=i_eps, RS_mode=RS_mode, IS_mode=IS_mode, split_limit=100, exe_start=exe_start,
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
        print("Invalid dataset name. Use 'mnist2', 'mnist4', 'mnistConv', 'cifar', or 'acasxu'.")
        sys.exit(1)


# for d_val in [1, 2, 3]:
#     for i_val in [2, 3, 4]:
#         # for rsis_mode in ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']:
#         for rsis_mode in ['RS_dual_Z', 'IS_dual_ind']:
#             print(f"Running experiments with d_eps={d_val}, i_eps={i_val}, RS/IS mode={rsis_mode}")

#             # gtsrb
#             print("** Run gtsrb **")
#             if d_val == 1:
#                 time = 1800
#             elif d_val == 2:
#                 time = 3600
#             elif d_val == 3:
#                 time = 7200
#             else:
#                 raise ValueError("d_val should be 1, 2, or 3")
#             d_eps = d_val
#             i_eps = i_val
#             exe_start = 0
#             exe_end = 10
#             inputs_num = 10
#             if rsis_mode.startswith('RS'):
#                 run_exp("gtsrb", RS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, time=time, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)
#             elif rsis_mode.startswith('IS'):
#                 run_exp("gtsrb", IS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, time=time, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)

# for d_val in [1, 2, 3]:
#     for i_val in [2, 3, 4]:
#         for rsis_mode in ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']:
#             print(f"Running experiments with d_eps={d_val}, i_eps={i_val}, RS/IS mode={rsis_mode}")

#             # cifar
#             print("** Run cifar **")
#             if d_val == 1:
#                 time = 1800
#             elif d_val == 2:
#                 time = 3600
#             elif d_val == 3:
#                 time = 7200
#             else:
#                 raise ValueError("d_val should be 1, 2, or 3")
#             d_eps = d_val
#             i_eps = i_val
#             exe_start = 0
#             if d_val == 1 or d_val == 2:
#                 exe_end = 16
#                 inputs_num = 16
#             else:
#                 exe_end = 10
#                 inputs_num = 10
#             if rsis_mode.startswith('RS'):
#                 run_exp("cifar", RS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, time=time, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)
#             elif rsis_mode.startswith('IS'):
#                 run_exp("cifar", IS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, time=time, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)


for d_val in [1, 2, 3]:
    for i_val in [2, 3, 4]:
        for perturb_ratio in [0.5, 0.25, 0.125]:
            for rsis_mode in ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']:
                print(f"Running experiments with d_eps={d_val}, i_eps={i_val}, RS/IS mode={rsis_mode}")
                
    #             # mnistConv
    #             print("** Run mnistConv **")
    #             exe_start = 0
    #             exe_end = 13  # run [0, 1, ..., 12]
    #             inputs_num = 13
    #             if rsis_mode.startswith('RS'):
    #                 run_exp("mnistConv", RS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)  # run [0, 1, ..., 12]
    #             elif rsis_mode.startswith('IS'):
    #                 run_exp("mnistConv", IS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)  # run [0, 1, ..., 12]

                # mnist4
                print("** Run mnist4 dp **")
                exe_start = 0
                exe_end = 5  # run [0, 1, ..., 12]
                inputs_num = 5
                if rsis_mode.startswith('RS'):
                    run_exp("mnist4", RS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num, perturb_ratio=perturb_ratio)  # run [0, 1, ..., 12]
                elif rsis_mode.startswith('IS'):
                    run_exp("mnist4", IS_mode=rsis_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num, perturb_ratio=perturb_ratio)  # run [0, 1, ..., 12]


# for d_val in [10]:
#     for net_idx1 in [1, 2]:
#         for net_idx2 in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
#             for rsis_mode in ['RS_random_Z', 'RS_dual_Z', 'IS_dual', 'IS_dual_ind']:
#                 print(f"Running experiments with d_eps={d_val}, net_idx1={net_idx1}, net_idx2={net_idx2}, RS/IS_mode={rsis_mode}")
#                 # acasxu
#                 print("** Run acasxu **")
#                 if rsis_mode.startswith('RS'):
#                     run_exp("acasxu", RS_mode=rsis_mode, d_eps=d_val, i_eps=None, net_idx1=net_idx1, net_idx2=net_idx2)
#                 elif rsis_mode.startswith('IS'):
#                     run_exp("acasxu", IS_mode=rsis_mode, d_eps=d_val, i_eps=None, net_idx1=net_idx1, net_idx2=net_idx2)


print("Done!")
