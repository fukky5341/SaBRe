import os
import time
import torch

from util import util
from relational_property.relational_analysis import (
    RelationalAnalysis,
    relational_analysis_back,
    RelationalProperty,
)
from common.dataset import Dataset
from common import Status
from specs import spec
from relational_split.rs_back import RS
from individual_split.is_back import IS
from threshold.get_threshold import get_thresholds_bs
from relational_bounds.relational_back_substitution import IndividualAndRelationalBounds


generator = torch.Generator()
generator.manual_seed(42)  # fixed seed for reproducibility

TOL = 1e-6
MNIST_DENOM = 256.0
CIFAR10_DENOM = 256.0


def perform_binary_search_acasxu(
    net_idx1=1,
    net_idx2=1,
    ini_d_eps=1,
    RSIS_mode_list=None,
    time_budget=1000,
    time_budget_for_one=420,
    bs_max_iter=100,
    exe_start=None,
    exe_end=None
):
    dataset = Dataset.ACAS
    net_name = f"onnx/acasxu_op11/ACASXU_{net_idx1}_{net_idx2}.onnx"
    dataset_name = "acasxu"
    d_eps = 1

    mode_path_pairs = []
    for mode in RSIS_mode_list:
        if mode.startswith("RS"):
            result_file_path = (
                f"experiment_result/binary_search/{dataset_name}/{mode}/"
                f"net_{net_idx1}_{net_idx2}_d_{d_eps}/"
            )
        elif mode.startswith("IS"):
            result_file_path = (
                f"experiment_result/binary_search/{dataset_name}/{mode}/"
                f"net_{net_idx1}_{net_idx2}_d_{d_eps}/"
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Mode should start with 'RS' or 'IS'.")
        os.makedirs(result_file_path, exist_ok=True)
        mode_path_pairs.append((mode, result_file_path))

    return binary_search_back(
        mode_path_pairs=mode_path_pairs,
        net_name=net_name,
        dataset=dataset,
        net_idx1=net_idx1,
        net_idx2=net_idx2,
        time_budget=time_budget,
        time_budget_for_one=time_budget_for_one,
        bs_max_iter=bs_max_iter,
        d_eps=None,
        i_eps=None,
        ini_d_eps=ini_d_eps,
        ini_i_eps=None,
        exe_start=exe_start,
        exe_end=exe_end
    )


def perform_binary_search_mnistConv(
    d_eps=3,
    i_eps=4,
    ini_d_eps=12,
    ini_i_eps=12,
    RSIS_mode_list=None,
    time_budget=2000,
    time_budget_for_one=600,
    bs_max_iter=100,
    exe_start=None,
    exe_end=None,
):
    dataset = Dataset.MNIST
    net_name = "onnx/mnist_conv_exp.onnx"
    dataset_name = "mnist-conv"

    return perform_binary_search_mnist_cifar(
        d_eps=d_eps,
        i_eps=i_eps,
        ini_d_eps=ini_d_eps,
        ini_i_eps=ini_i_eps,
        RSIS_mode_list=RSIS_mode_list,
        time_budget=time_budget,
        time_budget_for_one=time_budget_for_one,
        bs_max_iter=bs_max_iter,
        exe_start=exe_start,
        exe_end=exe_end,
        dataset=dataset,
        net_name=net_name,
        dataset_name=dataset_name,
    )

def perform_binary_search_mnist4(
    d_eps=3,
    i_eps=4,
    ini_d_eps=12,
    ini_i_eps=12,
    RSIS_mode_list=None,
    time_budget=2000,
    time_budget_for_one=600,
    bs_max_iter=100,
    exe_start=None,
    exe_end=None,
):
    dataset = Dataset.MNIST
    net_name = 'onnx/mnist-net_256x4.onnx'
    dataset_name = "mnist-256x4"

    return perform_binary_search_mnist_cifar(
        d_eps=d_eps,
        i_eps=i_eps,
        ini_d_eps=ini_d_eps,
        ini_i_eps=ini_i_eps,
        RSIS_mode_list=RSIS_mode_list,
        time_budget=time_budget,
        time_budget_for_one=time_budget_for_one,
        bs_max_iter=bs_max_iter,
        exe_start=exe_start,
        exe_end=exe_end,
        dataset=dataset,
        net_name=net_name,
        dataset_name=dataset_name,
    )

def perform_binary_search_cifar(
    d_eps=2,
    i_eps=4,
    ini_d_eps=8,
    ini_i_eps=8,
    RSIS_mode_list=None,
    time_budget=7200,
    time_budget_for_one=2400,
    bs_max_iter=100,
    exe_start=None,
    exe_end=None,
):
    dataset = Dataset.CIFAR10
    net_name = 'onnx/cifar10_conv_exp.onnx'
    dataset_name = "cifar10"

    return perform_binary_search_mnist_cifar(
        d_eps=d_eps,
        i_eps=i_eps,
        ini_d_eps=ini_d_eps,
        ini_i_eps=ini_i_eps,
        RSIS_mode_list=RSIS_mode_list,
        time_budget=time_budget,
        time_budget_for_one=time_budget_for_one,
        bs_max_iter=bs_max_iter,
        exe_start=exe_start,
        exe_end=exe_end,
        dataset=dataset,
        net_name=net_name,
        dataset_name=dataset_name,
    )

def perform_binary_search_mnist_cifar(
    d_eps=3,
    i_eps=4,
    ini_d_eps=12,
    ini_i_eps=12,
    RSIS_mode_list=None,
    time_budget=2000,
    time_budget_for_one=600,
    bs_max_iter=100,
    exe_start=None,
    exe_end=None,
    dataset=Dataset.MNIST,
    net_name="onnx/mnist_conv_exp.onnx",
    dataset_name="mnist-conv",
):

    mode_path_pairs = []
    for mode in RSIS_mode_list:
        if mode.startswith("RS"):
            result_file_path = f"experiment_result/binary_search/{dataset_name}/{mode}/"
        elif mode.startswith("IS"):
            result_file_path = f"experiment_result/binary_search/{dataset_name}/{mode}/"
        else:
            raise ValueError(f"Invalid mode: {mode}. Mode should start with 'RS' or 'IS'.")
        os.makedirs(result_file_path, exist_ok=True)
        mode_path_pairs.append((mode, result_file_path))

    return binary_search_back(
        mode_path_pairs=mode_path_pairs,
        net_name=net_name,
        dataset=dataset,
        net_idx1=None,
        net_idx2=None,
        time_budget=time_budget,
        time_budget_for_one=time_budget_for_one,
        bs_max_iter=bs_max_iter,
        d_eps=d_eps,
        i_eps=i_eps,
        ini_d_eps=ini_d_eps,
        ini_i_eps=ini_i_eps,
        exe_start=exe_start,
        exe_end=exe_end,
    )


def get_acasxu_input_diff(inp1_prop, diff):
    inp1_lb = inp1_prop.input_props[0].input_lb
    inp1_ub = inp1_prop.input_props[0].input_ub
    diff_t = torch.full_like(inp1_lb, diff)
    updated_diff = torch.min(diff_t, inp1_ub - inp1_lb)
    return updated_diff


def get_acasxu_max_input_diff(inp_diff_tensor):
    # if inp_diff_tensor is not a tensor (e.g., float), return it directly
    if not isinstance(inp_diff_tensor, torch.Tensor):
        return inp_diff_tensor
    # Return the maximum input diff value
    # e.g., if inp_diff_tensor = [0.1, 0.2, 0.3, 0.4, 0.5], return 0.5
    return torch.max(inp_diff_tensor).item()


def convert_candidate_to_delta_eps(dataset, inp1_prop, candidate):
    """
    candidate:
      - ACAS: real-valued diff
      - MNIST: integer k, meaning epsilon = k / 256
      - CIFAR10: integer k, meaning epsilon = k / 256
    """
    if dataset == Dataset.ACAS:
        return get_acasxu_input_diff(inp1_prop, candidate)
    elif dataset == Dataset.MNIST:
        return candidate / MNIST_DENOM
    elif dataset == Dataset.CIFAR10:
        return candidate / CIFAR10_DENOM
    else:
        raise NotImplementedError("Dataset not supported.")


def convert_eps_to_candidate(dataset, eps):
    if eps is None:
        return None
    if dataset == Dataset.ACAS:
        return float(eps)
    elif dataset == Dataset.MNIST:
        k = int(round(eps * MNIST_DENOM))
        return max(1, k)
    elif dataset == Dataset.CIFAR10:
        k = int(round(eps * CIFAR10_DENOM))
        return max(1, k)
    else:
        raise NotImplementedError("Dataset not supported.")


def get_initial_search_range(dataset, ini_d_eps, start_candidate=None):
    """
    Returns the binary-search domain [low, high].

    ACAS:
      search real value in [start_candidate, ini_d_eps]

    MNIST or CIFAR10:
      search discrete integer k in [start_candidate, ini_d_eps],
      corresponding to eps in {k / 256 | k = start_candidate, ..., ini_d_eps}
    """
    if dataset == Dataset.ACAS:
        low = 0.0 if start_candidate is None else float(start_candidate)
        return low, float(ini_d_eps), False
    elif dataset in (Dataset.MNIST, Dataset.CIFAR10):
        low = 1 if start_candidate is None else int(start_candidate)
        return low, int(ini_d_eps), True
    else:
        raise NotImplementedError("Dataset not supported.")


def build_relational_analysis(inp1_correct_label, inp2_correct_label, threshold, log_file):
    rel_prop = RelationalProperty.GLOBAL_ROBUSTNESS
    relAna = RelationalAnalysis(
        rel_prop,
        lp_analysis=True,
        global_target=True,
        inp1_correct_label=inp1_correct_label,
        inp2_correct_label=inp2_correct_label,
        threshold=threshold,
        log_file=log_file,
    )
    return relAna


def run_iar(inp1_prop, inp2_prop, net, dataset, delta_eps, device, refine_bounds_prop, log_file, back_prop_mode):
    iarb = IndividualAndRelationalBounds(
        inp1_prop,
        inp2_prop,
        net,
        dataset,
        delta_eps,
        device,
        refine_bounds_prop,
        log_file,
        back_prop_mode,
    )
    inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs = iarb.run()
    return iarb, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs


def get_abs_max_for_target(d_lbs, d_ubs, target_label):
    return max(
        abs(d_lbs[-1][target_label].item()),
        abs(d_ubs[-1][target_label].item()),
    )


def verify_candidate_base(
    candidate,
    threshold,
    inp1_prop,
    inp2_prop,
    net,
    dataset,
    device,
    refine_bounds_prop,
    log_file,
    back_prop_mode,
):
    """
    Two-stage oracle:
      1. approximate analysis (IAR)
      2. LP analysis if needed

    Returns:
      status, abs_max, rel_dist, iarb
    """
    inp1_correct_label = int(inp1_prop.out_constr.label.item())
    inp2_correct_label = int(inp2_prop.out_constr.label.item())

    delta_eps = convert_candidate_to_delta_eps(dataset, inp1_prop, candidate)

    iarb, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs = run_iar(
        inp1_prop,
        inp2_prop,
        net,
        dataset,
        delta_eps,
        device,
        refine_bounds_prop,
        log_file,
        back_prop_mode,
    )

    abs_max = get_abs_max_for_target(d_lbs, d_ubs, inp1_correct_label)
    if abs_max <= threshold:
        return Status.VERIFIED, abs_max, None, iarb

    relAna = build_relational_analysis(
        inp1_correct_label=inp1_correct_label,
        inp2_correct_label=inp2_correct_label,
        threshold=threshold,
        log_file=log_file,
    )
    status, _, _, rel_dist = relational_analysis_back(iarb, relAna, log_file)
    return status, abs_max, rel_dist, iarb


def verify_candidate_with_rsis(
    candidate,
    threshold,
    inp1_prop,
    inp2_prop,
    net,
    dataset,
    device,
    refine_bounds_prop,
    log_file,
    back_prop_mode,
    left_time,
    RS_mode=None,
    IS_mode=None,
):
    """
    Three-stage oracle:
      1. approximate analysis (IAR)
      2. LP analysis
      3. RS / IS if still not verified
    """
    if RS_mode is not None and IS_mode is not None:
        raise ValueError("Only one of RS_mode or IS_mode should be specified.")

    time_bs_rsis_start = time.time()

    inp1_correct_label = int(inp1_prop.out_constr.label.item())
    inp2_correct_label = int(inp2_prop.out_constr.label.item())

    delta_eps = convert_candidate_to_delta_eps(dataset, inp1_prop, candidate)

    iarb, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs = run_iar(
        inp1_prop,
        inp2_prop,
        net,
        dataset,
        delta_eps,
        device,
        refine_bounds_prop,
        log_file,
        back_prop_mode,
    )

    abs_max = get_abs_max_for_target(d_lbs, d_ubs, inp1_correct_label)
    if abs_max <= threshold:
        return Status.VERIFIED, abs_max, None

    relAna = build_relational_analysis(
        inp1_correct_label=inp1_correct_label,
        inp2_correct_label=inp2_correct_label,
        threshold=threshold,
        log_file=log_file,
    )
    status, _, _, rel_dist = relational_analysis_back(iarb, relAna, log_file)
    if status == Status.VERIFIED:
        return Status.VERIFIED, abs_max, rel_dist
    elif status == Status.ADV_EXAMPLE:
        return Status.ADV_EXAMPLE, None, None
    
    left_time_after_root = left_time - (time.time() - time_bs_rsis_start)
    if left_time_after_root <= 0:
        return Status.UNKNOWN, abs_max, rel_dist

    if RS_mode is not None and RS_mode in ["RS_dual_Z", "RS_random_Z"]:
        rs = RS(
            log_file=log_file,
            RS_mode=RS_mode,
            split_limit=9999,
            relational_prop=RelationalProperty.GLOBAL_ROBUSTNESS,
        )
        status, _, _ = rs.run_iterative_RS_backend(iarb, relAna, left_time_after_root, lp_analysis=True)
    elif IS_mode is not None:
        ind_sp = IS(
            log_file=log_file,
            IS_mode=IS_mode,
            split_limit=9999,
            relational_prop=RelationalProperty.GLOBAL_ROBUSTNESS,
        )
        status, _, _ = ind_sp.run_iterative_IS_backend(iarb, relAna, left_time_after_root, lp_analysis=True)
    else:
        raise ValueError("Either RS_mode or IS_mode must be specified.")

    return status, abs_max, rel_dist


def binary_search_candidate(
    time_budget,
    time_budget_for_one,
    max_iter,
    threshold,
    inp1_prop,
    inp2_prop,
    net,
    dataset,
    ini_d_eps,
    device,
    refine_bounds_prop,
    log_file,
    back_prop_mode,
    RS_mode=None,
    IS_mode=None,
    start_candidate=None,
):
    """
    Unified binary search.

    ACAS:
      continuous search on diff in [0, ini_d_eps]

    MNIST or CIFAR10:
      discrete search on k in [1, ini_d_eps], where eps = k / 256
    """
    low, high, is_discrete = get_initial_search_range(dataset, ini_d_eps, start_candidate)
    if is_discrete and start_candidate is not None:
        low = int(start_candidate) + 1

    if low > high:
        if start_candidate is None:
            return None, None
        if dataset == Dataset.MNIST:
            return Status.VERIFIED, start_candidate / MNIST_DENOM
        elif dataset == Dataset.CIFAR10:
            return Status.VERIFIED, start_candidate / CIFAR10_DENOM
        else:
            return Status.VERIFIED, float(start_candidate)

    best_candidate = start_candidate
    best_status = Status.VERIFIED if start_candidate is not None else None
    start_time_bs = time.time()

    if is_discrete:
        # binary search over integer k
        iter_num = 0
        while low <= high and iter_num < max_iter:
            left_time = time_budget - (time.time() - start_time_bs)
            if left_time <= 0:
                break

            mid = (low + high) // 2
            denom_m_c = MNIST_DENOM if dataset == Dataset.MNIST else CIFAR10_DENOM
            print(f"k_mid={mid}, eps_mid={mid / denom_m_c:.7f}")
            with open(f"{log_file}log.md", "a") as f:
                f.write(f"\n## Binary search (step {iter_num}) starts\n")
                f.write(f"Candidate k: {mid}, corresponding eps: {mid / denom_m_c:.7f}\n")

            if RS_mode is None and IS_mode is None:
                status, abs_max, rel_dist, _ = verify_candidate_base(
                    candidate=mid,
                    threshold=threshold,
                    inp1_prop=inp1_prop,
                    inp2_prop=inp2_prop,
                    net=net,
                    dataset=dataset,
                    device=device,
                    refine_bounds_prop=refine_bounds_prop,
                    log_file=log_file,
                    back_prop_mode=back_prop_mode,
                )
            else:
                time_budget_for_one_candidate = min(time_budget_for_one, left_time)
                status, abs_max, rel_dist = verify_candidate_with_rsis(
                    candidate=mid,
                    threshold=threshold,
                    inp1_prop=inp1_prop,
                    inp2_prop=inp2_prop,
                    net=net,
                    dataset=dataset,
                    device=device,
                    refine_bounds_prop=refine_bounds_prop,
                    log_file=log_file,
                    back_prop_mode=back_prop_mode,
                    left_time=time_budget_for_one_candidate,
                    RS_mode=RS_mode,
                    IS_mode=IS_mode,
                )

            with open(f"{log_file}log.md", "a") as f:
                f.write(
                    f"Binary search (step {iter_num}): "
                    f"status={status}, k_low={low}, k_high={high}, "
                    f"k_mid={mid}, eps_mid={mid / denom_m_c:.7f}, abs_max={abs_max}\n"
                )
                if rel_dist is not None:
                    f.write(f"rel_dist={rel_dist}\n")

            if status == Status.VERIFIED:
                best_candidate = mid
                best_status = Status.VERIFIED
                low = mid + 1
            else:
                high = mid - 1

            iter_num += 1

        max_d_eps = None if best_candidate is None else best_candidate / denom_m_c
        return best_status, max_d_eps

    else:
        # binary search over real-valued diff
        iter_num = 0
        while iter_num < max_iter:
            left_time = time_budget - (time.time() - start_time_bs)
            if left_time <= 0:
                break
            if high - low < TOL:
                break

            mid = (low + high) / 2.0
            print(f"mid={mid:.7f}")
            with open(f"{log_file}log.md", "a") as f:
                f.write(f"\n## Binary search (step {iter_num}) starts\n")
                f.write(f"Candidate diff: {mid:.7f}\n")

            if RS_mode is None and IS_mode is None:
                status, abs_max, rel_dist, _ = verify_candidate_base(
                    candidate=mid,
                    threshold=threshold,
                    inp1_prop=inp1_prop,
                    inp2_prop=inp2_prop,
                    net=net,
                    dataset=dataset,
                    device=device,
                    refine_bounds_prop=refine_bounds_prop,
                    log_file=log_file,
                    back_prop_mode=back_prop_mode,
                )
            else:
                time_budget_for_one_candidate = min(time_budget_for_one, left_time)
                status, abs_max, rel_dist = verify_candidate_with_rsis(
                    candidate=mid,
                    threshold=threshold,
                    inp1_prop=inp1_prop,
                    inp2_prop=inp2_prop,
                    net=net,
                    dataset=dataset,
                    device=device,
                    refine_bounds_prop=refine_bounds_prop,
                    log_file=log_file,
                    back_prop_mode=back_prop_mode,
                    left_time=time_budget_for_one_candidate,
                    RS_mode=RS_mode,
                    IS_mode=IS_mode,
                )

            if status == Status.VERIFIED:
                best_candidate = mid
                best_status = Status.VERIFIED
                low = mid
            else:
                high = mid

            with open(f"{log_file}log.md", "a") as f:
                f.write(
                    f"Binary search (step {iter_num}): "
                    f"status={status}, low={low:.7f}, high={high:.7f}, "
                    f"mid={mid:.7f}, abs_max={abs_max}\n"
                )
                if rel_dist is not None:
                    f.write(f"rel_dist={rel_dist}\n")

            iter_num += 1

        return best_status, best_candidate


def log_base_iar_bounds(log_file, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
    with open(f"{log_file}log.md", "a") as f:
        f.write("\n### BASE IAR bounds\n")
        f.write("Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)\n")
        for layer_idx in range(len(inp1_lbs[-1])):
            f.write(
                f"{layer_idx}: ("
                f"{inp1_lbs[-1][layer_idx]:.7f}, {inp1_ubs[-1][layer_idx]:.7f}, "
                f"{inp2_lbs[-1][layer_idx]:.7f}, {inp2_ubs[-1][layer_idx]:.7f}, "
                f"{d_lbs[-1][layer_idx]:.7f}, {d_ubs[-1][layer_idx]:.7f})\n"
            )


def binary_search_back(
    mode_path_pairs,
    net_name,
    dataset,
    time_budget,
    time_budget_for_one,
    bs_max_iter,
    d_eps,
    i_eps,
    ini_d_eps,
    ini_i_eps,
    net_idx1=None,
    net_idx2=None,
    exe_start=None,
    exe_end=None,
):
    net = util.get_net(net_name, dataset)
    back_prop_mode = "DP"
    refine_bounds_prop = True

    if dataset == Dataset.ACAS:
        eps = None
        count = 10
        base_candidate = float(ini_d_eps)
        thr_id = net_idx2
    elif dataset == Dataset.MNIST:
        eps = ini_i_eps / MNIST_DENOM
        count = 13
        base_candidate = int(ini_d_eps)
        seed_value = int(100*d_eps + i_eps)
        generator.manual_seed(seed_value)
        thr_id = seed_value
    elif dataset == Dataset.CIFAR10:
        eps = ini_i_eps / CIFAR10_DENOM
        count = 16
        base_candidate = int(ini_d_eps)
        seed_value = int(100*d_eps + i_eps)
        generator.manual_seed(seed_value)
        thr_id = seed_value
    else:
        raise NotImplementedError("Dataset not supported for binary search.")

    props, _ = spec.get_specs(
        dataset=dataset,
        eps=eps,
        count=count,
        shuffle=True,
        generator=generator,
    )

    thresholds = get_thresholds_bs(net_name, thr_id=thr_id)
    if len(thresholds) < len(props):
        raise ValueError(
            f"The number of thresholds ({len(thresholds)}) does not match "
            f"the number of properties ({len(props)})."
        )
    
    if exe_start is not None and exe_end is not None:
        props = props[exe_start:exe_end]
        thresholds = thresholds[exe_start:exe_end]

    for local_exe_idx in range(len(props)):
        inp1_prop, inp2_prop = props[local_exe_idx], props[local_exe_idx]
        inp1_correct_label = int(inp1_prop.out_constr.label.item())
        inp2_correct_label = int(inp2_prop.out_constr.label.item())
        curr_threshold = thresholds[local_exe_idx]

        exe_idx = local_exe_idx
        if exe_start is not None and exe_end is not None:
            exe_idx = exe_start + local_exe_idx

        for mode, result_file_path in mode_path_pairs:
            log_file = f"{result_file_path}{exe_idx}/"
            os.makedirs(log_file, exist_ok=True)

            denom_m_c = MNIST_DENOM if dataset == Dataset.MNIST else CIFAR10_DENOM
            with open(f"{log_file}log.md", "w") as f:
                f.write("## Execution arguments:\n")
                f.write(f"Dataset: {dataset}\n")
                f.write(f"Network: {net_name}\n")
                f.write(f"Epsilon: {eps}\n")
                f.write(f"Initial delta epsilon: {ini_d_eps}\n")
                f.write(f"Time budget: {time_budget} seconds\n")
                f.write(f"Threshold: {curr_threshold}\n")
                if dataset in (Dataset.MNIST, Dataset.CIFAR10):
                    f.write(
                        f"Search space: {{k/{denom_m_c} | k = 1, 2, ..., {ini_d_eps}}}\n"
                    )

            print(f"Executing binary search for property {exe_idx} with threshold {curr_threshold}...")

            # ===== BASE analysis at largest candidate =====
            start_time_base_iar = time.time()
            delta_eps_base = convert_candidate_to_delta_eps(dataset, inp1_prop, base_candidate)

            iarb_base, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs = run_iar(
                inp1_prop,
                inp2_prop,
                net,
                dataset,
                delta_eps_base,
                "cpu",
                refine_bounds_prop,
                log_file,
                back_prop_mode,
            )

            log_base_iar_bounds(log_file, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs)

            abs_max = get_abs_max_for_target(d_lbs, d_ubs, inp1_correct_label)
            end_time_base_iar = time.time()
            time_base_iar = end_time_base_iar - start_time_base_iar

            if abs_max <= curr_threshold:
                status = Status.VERIFIED
                with open(f"{log_file}log.md", "a") as f:
                    f.write("\n## BASE Result\n")
                    f.write(f"execution time: IAR = {time_base_iar:.2f} seconds\n")
                    f.write(f"status: {status}\n")
                print(f"Property {exe_idx} is done.")
                continue

            start_time_base_lp = time.time()
            relAna_base = build_relational_analysis(
                inp1_correct_label=inp1_correct_label,
                inp2_correct_label=inp2_correct_label,
                threshold=curr_threshold,
                log_file=log_file,
            )
            status_base_lp, _, _, rel_dist_base_lp = relational_analysis_back(
                iarb_base, relAna_base, log_file
            )
            end_time_base_lp = time.time()
            time_base_lp = end_time_base_lp - start_time_base_lp
            time_base = time_base_iar + time_base_lp

            with open(f"{log_file}log.md", "a") as f:
                f.write("\n## BASE Result\n")
                f.write(
                    f"execution time: IAR + LP analysis = "
                    f"{time_base_iar:.2f} + {time_base_lp:.2f} = {time_base:.2f} seconds\n"
                )
                f.write(f"status: {status_base_lp}\n")
                if rel_dist_base_lp is not None:
                    f.write("relational distance\n")
                    for dim, dist in rel_dist_base_lp.items():
                        f.write(
                            f"Output dim: {dim}, "
                            f"lower bound: {dist[0]:.7f}, upper bound: {dist[1]:.7f}\n"
                        )

            if status_base_lp == Status.VERIFIED:
                print(f"Property {exe_idx} is done.")
                continue

            left_time = time_budget - time_base
            if left_time <= 0:
                with open(f"{log_file}log.md", "a") as f:
                    f.write(
                        "\nTime budget exceeded after BASE analysis. "
                        "No time left for binary search or RS/IS.\n"
                    )
                print(
                    f"Execution {exe_idx}: Time budget exceeded after BASE analysis. "
                    "No time left for binary search or RS/IS."
                )
                continue

            bs_ini_d_eps = get_acasxu_max_input_diff(delta_eps_base) if dataset == Dataset.ACAS else ini_d_eps

            # ===== Binary search by BASE (IAR + LP) =====
            with open(f"{log_file}log.md", "a") as f:
                f.write(
                    f"\n\n# Binary Search by BASE starts "
                    f"(time budget: {left_time:.2f} seconds, max iter: {bs_max_iter})\n"
                )
            start_time_bs = time.time()
            bs_status, max_d_eps = binary_search_candidate(
                time_budget=left_time,
                time_budget_for_one=time_budget_for_one,
                max_iter=bs_max_iter,
                threshold=curr_threshold,
                inp1_prop=inp1_prop,
                inp2_prop=inp2_prop,
                net=net,
                dataset=dataset,
                ini_d_eps=bs_ini_d_eps,
                device="cpu",
                refine_bounds_prop=refine_bounds_prop,
                log_file=log_file,
                back_prop_mode=back_prop_mode,
                RS_mode=None,
                IS_mode=None,
                start_candidate=None,
            )
            end_time_bs = time.time()
            time_bs = end_time_bs - start_time_bs

            with open(f"{log_file}log.md", "a") as f:
                f.write("\n## Binary Search Result\n")
                f.write(f"Binary search time: {time_bs:.2f} seconds\n")
                f.write(f"BS Status: {bs_status}\n")
                f.write(f"Maximum delta epsilon: {max_d_eps}\n")

            print(f"Execution {exe_idx} by Base: Maximum delta epsilon: {max_d_eps}, Status: {bs_status}")

            left_time_after_bs = left_time - time_bs
            if left_time_after_bs <= 0:
                with open(f"{log_file}log.md", "a") as f:
                    f.write(
                        "\nTime budget exceeded after binary search. "
                        "No time left for RS/IS.\n"
                    )
                print(
                    f"Execution {exe_idx}: Time budget exceeded after binary search. "
                    "No time left for RS/IS."
                )
                continue

            # ===== RS / IS =====
            if mode.startswith("RS"):
                RS_mode = mode
                IS_mode = None
                rsis_display_long = f"Relational Split ({mode})"
            elif mode.startswith("IS"):
                IS_mode = mode
                RS_mode = None
                rsis_display_long = f"Individual Split ({mode})"
            else:
                raise ValueError(f"Invalid mode: {mode}")

            print(mode)
            with open(f"{log_file}log.md", "a") as f:
                f.write(f"\n\n# {rsis_display_long} starts\n")
                f.write(f'Time budget: {left_time_after_bs:.2f} seconds\n')

            start_time_rsis = time.time()
            base_start_candidate = convert_eps_to_candidate(dataset, max_d_eps)
            rsis_status, rsis_max_d_eps = binary_search_candidate(
                time_budget=left_time_after_bs,
                time_budget_for_one=time_budget_for_one,
                max_iter=bs_max_iter,
                threshold=curr_threshold,
                inp1_prop=inp1_prop,
                inp2_prop=inp2_prop,
                net=net,
                dataset=dataset,
                ini_d_eps=bs_ini_d_eps,
                device="cpu",
                refine_bounds_prop=refine_bounds_prop,
                log_file=log_file,
                back_prop_mode=back_prop_mode,
                RS_mode=RS_mode,
                IS_mode=IS_mode,
                start_candidate=base_start_candidate,
            )
            end_time_rsis = time.time()
            time_rsis = end_time_rsis - start_time_rsis

            with open(f"{log_file}log.md", "a") as f:
                f.write(f"\n## Binary Search with {mode} Result\n")
                f.write(f"status: {rsis_status}\n")
                f.write(f"Maximum delta epsilon: {rsis_max_d_eps}\n")
                f.write(f"execution time: {time_rsis:.2f} seconds\n")

            print(
                f"Execution {exe_idx} with {mode}: "
                f"Maximum delta epsilon: {rsis_max_d_eps}, Status: {rsis_status}"
            )