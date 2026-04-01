from relational_property.relational_analysis import RelationalAnalysis, relational_analysis_back, RelationalProperty
import random
from common import Status
from relational_split.rs_back import RS
from individual_split.is_back import IS
from relational_bounds.relational_back_substitution import IndividualAndRelationalBounds
from util import util
from common.dataset import Dataset
from specs import spec
import itertools
import os
import torch
import time
from threshold.get_threshold import get_thresholds

"""
variables
    exe_limit: limit the number of executions for each input (curently for acasxu only)
    inputs_num: number of inputs to be used (currently for mnist and cifar only)
"""

generator = torch.Generator()
generator.manual_seed(42)


def execute_experiment_acasxu(net_idx1=1, net_idx2=1, d_eps=1, RS_mode=None, IS_mode=None, threshold_analysis=False, time_budget=600, split_limit=5, exe_limit=None, exe_start=None, exe_end=None):
    dataset = Dataset.ACAS
    net_name = f'onnx/acasxu_op11/ACASXU_{net_idx1}_{net_idx2}.onnx'
    dataset_name = "acasxu"

    if RS_mode is not None and IS_mode is not None:
        raise ValueError("Only one of RS_mode or IS_mode should be specified.")
    if RS_mode is not None:
        if threshold_analysis:
            result_file_path = f"experiment_result/{dataset_name}/{RS_mode}_threshold/net_{net_idx1}_{net_idx2}_d_{d_eps}/"
        else:
            result_file_path = f"experiment_result/{dataset_name}/{RS_mode}/net_{net_idx1}_{net_idx2}_d_{d_eps}/"
    elif IS_mode is not None:
        if threshold_analysis:
            result_file_path = f"experiment_result/{dataset_name}/{IS_mode}_threshold/net_{net_idx1}_{net_idx2}_d_{d_eps}/"
        else:
            result_file_path = f"experiment_result/{dataset_name}/{IS_mode}/net_{net_idx1}_{net_idx2}_d_{d_eps}/"
    os.makedirs(result_file_path, exist_ok=True)
    relational_prop = RelationalProperty.GLOBAL_ROBUSTNESS
    lp_analysis = True
    global_target = True

    return execute_experiment(dataset, net_name, result_file_path,
                              d_eps=d_eps, i_eps=1, net_idx1=net_idx1, net_idx2=net_idx2,
                              relational_prop=relational_prop, RS_mode=RS_mode, IS_mode=IS_mode,
                              exe_limit=exe_limit, exe_start=exe_start, exe_end=exe_end,
                              split_limit=split_limit, time_budget=time_budget, lp_analysis=lp_analysis,
                              threshold_analysis=threshold_analysis, global_target=global_target)


def execute_experiment_cifar(d_eps=None, i_eps=2, RS_mode=None, IS_mode=None, threshold_analysis=False, time_budget=600, split_limit=5, inputs_num=50, exe_start=None, exe_end=None):
    dataset = Dataset.CIFAR10
    net_name = 'onnx/cifar10_conv_exp.onnx'
    dataset_name = "cifar10"

    if d_eps is None:
        result_folder_path = f"experiment_result/PE/{dataset_name}/"
    else:
        result_folder_path = f"experiment_result/{dataset_name}/"

    if RS_mode is not None and IS_mode is not None:
        raise ValueError("Only one of RS_mode or IS_mode should be specified.")
    if RS_mode is not None:
        if threshold_analysis:
            result_file_path = f"{result_folder_path}{RS_mode}_threshold/"
        else:
            result_file_path = f"{result_folder_path}{RS_mode}/"
    elif IS_mode is not None:
        if threshold_analysis:
            result_file_path = f"{result_folder_path}{IS_mode}_threshold/"
        else:
            result_file_path = f"{result_folder_path}{IS_mode}/"
    os.makedirs(result_file_path, exist_ok=True)
    relational_prop = RelationalProperty.GLOBAL_ROBUSTNESS
    lp_analysis = True
    global_target = True

    return execute_experiment(dataset, net_name, result_file_path,
                              d_eps=d_eps, i_eps=i_eps, relational_prop=relational_prop,
                              RS_mode=RS_mode, IS_mode=IS_mode, exe_start=exe_start, exe_end=exe_end,
                              split_limit=split_limit, inputs_num=inputs_num,
                              time_budget=time_budget, lp_analysis=lp_analysis, threshold_analysis=threshold_analysis,
                              global_target=global_target)


def execute_experiment_mnist4(d_eps=None, i_eps=2, RS_mode=None, IS_mode=None, threshold_analysis=False, time_budget=600, split_limit=5, inputs_num=50, exe_limit=1, exe_start=None, exe_end=None):
    dataset = Dataset.MNIST
    net_name = 'onnx/mnist-net_256x4.onnx'
    dataset_name = "mnist-256x4"

    if d_eps is None:
        result_folder_path = f"experiment_result/PE/{dataset_name}/"
    else:
        result_folder_path = f"experiment_result/{dataset_name}/"

    if RS_mode is not None and IS_mode is not None:
        raise ValueError("Only one of RS_mode or IS_mode should be specified.")
    if RS_mode is not None:
        if threshold_analysis:
            result_file_path = f"{result_folder_path}{RS_mode}_threshold/"
        else:
            result_file_path = f"{result_folder_path}{RS_mode}/"
    elif IS_mode is not None:
        if threshold_analysis:
            result_file_path = f"{result_folder_path}{IS_mode}_threshold/"
        else:
            result_file_path = f"{result_folder_path}{IS_mode}/"
    os.makedirs(result_file_path, exist_ok=True)
    relational_prop = RelationalProperty.GLOBAL_ROBUSTNESS
    lp_analysis = True
    global_target = True

    return execute_experiment(dataset, net_name, result_file_path,
                              d_eps=d_eps, i_eps=i_eps, relational_prop=relational_prop,
                              RS_mode=RS_mode, IS_mode=IS_mode, exe_start=exe_start, exe_end=exe_end,
                              split_limit=split_limit, inputs_num=inputs_num,
                              time_budget=time_budget, lp_analysis=lp_analysis,
                              threshold_analysis=threshold_analysis,
                              exe_limit=exe_limit, global_target=global_target)


def execute_experiment_mnistConv(d_eps=None, i_eps=2, RS_mode=None, IS_mode=None, threshold_analysis=False, time_budget=600, split_limit=5, inputs_num=50, exe_start=None, exe_end=None):
    dataset = Dataset.MNIST
    net_name = 'onnx/mnist_conv_exp.onnx'
    dataset_name = "mnist-conv"
    result_file_path = f"experiment_result/{dataset_name}/"

    if d_eps is None:
        result_folder_path = f"experiment_result/PE/{dataset_name}/"
    else:
        result_folder_path = f"experiment_result/{dataset_name}/"

    if RS_mode is not None and IS_mode is not None:
        raise ValueError("Only one of RS_mode or IS_mode should be specified.")
    if RS_mode is not None:
        if threshold_analysis:
            result_file_path = f"{result_folder_path}{RS_mode}_threshold/"
        else:
            result_file_path = f"{result_folder_path}{RS_mode}/"
    elif IS_mode is not None:
        if threshold_analysis:
            result_file_path = f"{result_folder_path}{IS_mode}_threshold/"
        else:
            result_file_path = f"{result_folder_path}{IS_mode}/"
    os.makedirs(result_file_path, exist_ok=True)
    relational_prop = RelationalProperty.GLOBAL_ROBUSTNESS
    lp_analysis = True
    global_target = True

    return execute_experiment(dataset, net_name, result_file_path,
                              d_eps=d_eps, i_eps=i_eps, relational_prop=relational_prop,
                              RS_mode=RS_mode, IS_mode=IS_mode, exe_start=exe_start, exe_end=exe_end,
                              split_limit=split_limit, inputs_num=inputs_num,
                              time_budget=time_budget, lp_analysis=lp_analysis,
                              threshold_analysis=threshold_analysis, global_target=global_target)


def execute_experiment(dataset, net_name, result_file_path, d_eps=None, i_eps=None, net_idx1=None, net_idx2=None, exe_limit=None,
                       relational_prop=None, exe_start=None, exe_end=None, time_budget=200, inputs_num=50,
                       split_limit=1000, RS_mode=None, IS_mode=None, lp_analysis=False, threshold_analysis=False,
                       global_target=False):
    net = util.get_net(net_name=net_name, dataset=dataset)

    backprop_mode = "DP"
    refine_bounds_prop = True
    if dataset == Dataset.ACAS:
        count = 10
    else:
        count = inputs_num
    split_limit = split_limit
    time_budget = time_budget  # seconds
    if relational_prop == RelationalProperty.GLOBAL_ROBUSTNESS:
        delta_eps = 1/256 * d_eps
        eps = 1/256 * d_eps * i_eps
        generator = torch.Generator()
        seed_value = int(100*d_eps + i_eps)
        generator.manual_seed(seed_value)
    if dataset == Dataset.ACAS:
        delta_eps = 0.01 * d_eps
        i_eps = None
        eps = None

    if exe_limit is not None:
        if exe_limit <= 1:
            exe_limit = None

    props, _ = spec.get_specs(dataset=dataset, eps=eps, count=count, shuffle=True, generator=generator)

    if threshold_analysis:
        thresholds = get_thresholds(net_name, d_eps, i_eps, net_idx1=net_idx1, net_idx2=net_idx2)
        if len(thresholds) < len(props):
            raise ValueError(f"The number of thresholds ({len(thresholds)}) does not match the number of properties ({len(props)}).")

    if exe_start is not None and exe_end is not None:
        props = props[exe_start:exe_end]  # if exe_start is 0 and exe_end is 40, then props will be from 0 to 39 (inclusive 0, exclusive 40)
        thresholds = thresholds[exe_start:exe_end] if threshold_analysis else None

    for exe_idx in range(len(props)):
        if relational_prop == RelationalProperty.GLOBAL_ROBUSTNESS:
            inp1_prop = props[exe_idx]
            inp2_prop = props[exe_idx]

        inp1_correct_label = inp1_prop.out_constr.label.item()  # int
        inp2_correct_label = inp2_prop.out_constr.label.item()

        if relational_prop == RelationalProperty.GLOBAL_ROBUSTNESS:
            log_exe_idx = exe_idx + exe_start if (exe_start is not None and exe_end is not None) else exe_idx
            if dataset != Dataset.ACAS:
                log_file = f"{result_file_path}d{d_eps}_e{i_eps}/{log_exe_idx}/"
            else:
                log_file = f"{result_file_path}{log_exe_idx}/"
        else:
            log_file = f"{result_file_path}e{i_eps}/{log_exe_idx}/"
        os.makedirs(log_file, exist_ok=True)

        if threshold_analysis:
            curr_threshold = thresholds[exe_idx]
        else:
            curr_threshold = None

        with open(f"{log_file}log.md", 'w') as f:
            f.write(f"## Execution arguments:\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Network: {net_name}\n")
            f.write(f"Relational property: {relational_prop.name}\n")
            f.write(f"LP Analysis: {lp_analysis}\n")
            f.write(f"Epsilon: {eps}\n")
            f.write(f"Delta epsilon: {delta_eps}\n")
            f.write(f"execution index: ({d_eps}, {i_eps}, {log_exe_idx})\n")
            f.write(f"Time budget: {time_budget} seconds\n")
            f.write(f"Split limit: {split_limit}\n")
            if relational_prop == RelationalProperty.GLOBAL_ROBUSTNESS and threshold_analysis:
                f.write(f"Threshold: {curr_threshold}\n")

        print(f"execution: (d:{d_eps}, i:{i_eps}, idx:{log_exe_idx})")

        # $ BASE (RaVeN, DiffPoly)
        print("base")
        iarb = IndividualAndRelationalBounds(net=net, inp1_prop=inp1_prop, inp2_prop=inp2_prop, delta_eps=delta_eps, device='cpu',
                                             dataset=dataset, refine_bounds_prop=refine_bounds_prop, log_file=log_file, backprop_mode=backprop_mode)
        start_time_base_iar = time.time()
        inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs = iarb.run()
        end_time_base_iar = time.time()
        time_base_iar = end_time_base_iar - start_time_base_iar
        with open(f"{log_file}log.md", 'a') as f:
            f.write(f"\n### BASE IAR bounds\n")
            f.write(f"Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)\n")
            for layer_idx in range(len(inp1_lbs[-1])):
                f.write(f"{layer_idx}: ({inp1_lbs[-1][layer_idx]:.7f}, {inp1_ubs[-1][layer_idx]:.7f}, ")
                f.write(f"{inp2_lbs[-1][layer_idx]:.7f}, {inp2_ubs[-1][layer_idx]:.7f}, ")
                f.write(f"{d_lbs[-1][layer_idx]:.7f}, {d_ubs[-1][layer_idx]:.7f})\n")
        # Relational property analysis
        with open(f"{log_file}log.md", 'a') as f:
            f.write(f"\n## BASE Relational Analysis\n")
        relAna_base = RelationalAnalysis(relational_prop=relational_prop, lp_analysis=lp_analysis, global_target=global_target,
                                         inp1_correct_label=inp1_correct_label, inp2_correct_label=inp2_correct_label, threshold=curr_threshold, log_file=log_file)
        start_time_base_ra = time.time()
        status_base, inp1_label_base, inp2_label_base, rel_dist_base = relational_analysis_back(IARb=iarb, RelAna=relAna_base, log_file=log_file)
        end_time_base_ra = time.time()
        time_base_ra = end_time_base_ra - start_time_base_ra
        # time
        time_base = time_base_iar + time_base_ra
        with open(f"{log_file}log.md", 'a') as f:
            f.write(f"\n## BASE Result\n")
            f.write(f"execution time: IAR + RelationalAnalysis = {time_base_iar:.2f} + {time_base_ra:.2f} = {time_base:.2f} seconds\n")
            f.write(f"status: {status_base}\n")
            if relational_prop == RelationalProperty.GLOBAL_ROBUSTNESS:
                if rel_dist_base is not None:
                    f.write(f"relational distance\n")
                    for dim, dist in rel_dist_base.items():
                        f.write(f"Output dim: {dim}, lower bound: {dist[0]:.7f}, upper bound: {dist[1]:.7f}\n")

        if status_base == Status.UNKNOWN:
            left_time_budget = time_budget - time_base
            left_time = time_base

            # $ RS
            if RS_mode is not None:
                print("RS")
                with open(f"{log_file}log.md", 'a') as f:
                    f.write(f"\n# Relational Split (RS) starts\n")

                if left_time < time_budget:
                    rs = RS(log_file=log_file, RS_mode=RS_mode, split_limit=split_limit, relational_prop=relational_prop)
                    # rs = RS(IndividualAndRelationalBounds=iarb, log_file=log_file, RS_mode='RS_random', split_limit=split_limit)
                    if RS_mode in ['RS_dual_ABCD', 'RS_dual_ABCDMZ', 'RS_dual_MZ', 'RS_random_ABCD', 'RS_random_Z',
                                   'RS_dual_A', 'RS_dual_B', 'RS_dual_C', 'RS_dual_D', 'RS_dual_M', 'RS_dual_Z']:
                        start_time_rs = time.time()
                        result, inp1_label, inp2_label = rs.run_iterative_RS_backend(IARb=iarb, RelAna=relAna_base,
                                                                                     time_budget=left_time_budget, lp_analysis=lp_analysis)
                        end_time_rs = time.time()
                        time_rs = end_time_rs - start_time_rs + time_base
                        print(f"RS result: {result}")
                        with open(f"{log_file}log.md", 'a') as f:
                            f.write(f"\n## RS Result\n")
                            f.write(f"status: {result}\n")
                            f.write(f"execution time: (base) + (rs) = {time_base:.2f} + {(end_time_rs - start_time_rs):.2f} = {time_rs:.2f} seconds\n")

            # $ IS
            elif IS_mode is not None:
                print("IS")
                with open(f"{log_file}log.md", 'a') as f:
                    f.write(f"\n# Indivdual Split (IS) starts\n")
                if left_time < time_budget:
                    start_time_is = time.time()
                    ind_sp = IS(log_file=log_file, IS_mode=IS_mode, split_limit=split_limit, relational_prop=relational_prop)
                    result, inp1_label, inp2_label = ind_sp.run_iterative_IS_backend(IARb=iarb, RelAna=relAna_base,
                                                                                 time_budget=left_time_budget, lp_analysis=lp_analysis)
                    end_time_is = time.time()
                    time_is = end_time_is - start_time_is + time_base
                    print(f"IS result: {result}")
                    with open(f"{log_file}log.md", 'a') as f:
                        f.write(f"\n## IS Result\n")
                        f.write(f"status: {result}\n")
                        f.write(f"execution time: (base) + (is) = {time_base:.2f} + {(end_time_is - start_time_is):.2f} = {time_is:.2f} seconds\n")
