import re
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from IPython.display import display
from analysis import *


pd.options.display.max_rows = None
pd.options.display.max_columns = None

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_script_path(path):
    path = Path(path)
    return path if path.is_absolute() else (SCRIPT_DIR / path)

dsns_file = '_dsns_whole.csv'
acasxu = pd.read_csv(SCRIPT_DIR / "acasxu" / f"acasxu{dsns_file}")
mnist4 = pd.read_csv(SCRIPT_DIR / "mnist-256x4" / f"mnist4{dsns_file}")
mnist_conv = pd.read_csv(SCRIPT_DIR / "mnist-conv" / f"mnistconv{dsns_file}")
cifar10 = pd.read_csv(SCRIPT_DIR / "cifar10" / f"cifar{dsns_file}")
gtsrb = pd.read_csv(SCRIPT_DIR / "gtsrb" / f"gtsrb{dsns_file}")

SABRE_DISPLAY_NAME = r'S\textsc{a}BR\textsc{e}' if plt.rcParams.get('text.usetex', False) else 'SᴀBRᴇ'

def filter_instance_info_acasxu(df):
    # collect instance info whose 'base status' is 'UNKNOWN'
    # name: n1_n2_d_input --> (n1, n2, d, input_idx)
    filtered_info = []
    for idx, row in df.iterrows():
        if row['base status'] == 'UNKNOWN':
            name = row['name']
            n1, n2, d, input_idx = re.match(r'(\w+)_(\w+)_(\d+)_(\d+)', name).groups()
            filtered_info.append((int(n1), int(n2), int(d), int(input_idx)))
    return filtered_info

def filter_instance_info(df):
    # collect instance info whose 'base status' is 'UNKNOWN'
    # name: d_e_input --> (d, e, input_idx)
    filtered_info = []
    for idx, row in df.iterrows():
        if row['base status'] == 'UNKNOWN':
            name = row['name']
            d, e, input_idx = re.match(r'(\d+)_(\d+)_(\d+)', name).groups()
            filtered_info.append((int(d), int(e), int(input_idx)))
    return filtered_info

aca_instances = filter_instance_info_acasxu(acasxu)  # list of (n1, n2, d, input_idx)
m4_instances = filter_instance_info(mnist4)  # list of (d, e, input_idx)
mc_instances = filter_instance_info(mnist_conv)
c10_instances = filter_instance_info(cifar10)
gt_instances = filter_instance_info(gtsrb)


def get_thresholds_ratio_from_path(net_name, d_eps=None, i_eps=None, net1_id=None, net2_id=None):
    if 'mnist4' in net_name:
        file_path = resolve_script_path(f"../threshold/threshold_ratio/mnist4/d{d_eps}_e{i_eps}.txt")
    elif 'mnist-conv' in net_name:
        file_path = resolve_script_path(f"../threshold/threshold_ratio/mnist-conv/d{d_eps}_e{i_eps}.txt")
    elif 'cifar10' in net_name:
        file_path = resolve_script_path(f"../threshold/threshold_ratio/cifar10/d{d_eps}_e{i_eps}.txt")
    elif 'acasxu' in net_name:
        file_path = resolve_script_path(f"../threshold/threshold_ratio/acasxu/net_{net1_id}_{net2_id}_d_10.txt")
    else:
        raise ValueError(f"Unknown net_name: {net_name}")

    return read_thresholds_ratio(file_path)


def read_thresholds_ratio(file_path):
    """
    e.g.,
    0.99
    0.80
    ...
    
    """

    thresholds = []
    with open(Path(file_path), 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                thresholds.append(float(line))
    return thresholds

def get_threshold_ratio(net_name, d_val, i_val, input_idx, net1=None, net2=None):
    if net_name == "acasxu":
        thresholds_ratio = get_thresholds_ratio_from_path(net_name, None, None, net1, net2)
    else:
        thresholds_ratio = get_thresholds_ratio_from_path(net_name, d_val, i_val)
    if input_idx >= len(thresholds_ratio):
        raise ValueError(f"Input index {input_idx} is out of range for thresholds of length {len(thresholds_ratio)}")
    return thresholds_ratio[input_idx]

def extract_exp_info_from_csvpath(file_path):
    """
    e.g., './mnist-256x4/DS_dual/d1_e3_3.csv'
    return: (d_val, i_val, input_idx) = (1, 3, 3)
    """
    path = Path(file_path)
    stem_parts = path.stem.split('_')
    d_val = int(stem_parts[0][1:])
    i_val = int(stem_parts[1][1:])
    input_idx = int(stem_parts[2])
    return d_val, i_val, input_idx

def extract_exp_info_from_csvpath_acasxu(file_path):
    """
    e.g., './mnist-256x4/DS_dual/net_1_1_d_50/0.csv'
    return: (net_idx1, net_idx2, d_val, input_idx) = (1, 1, 50, 0)
    """
    path = Path(file_path)
    parent_parts = path.parent.name.split('_')
    net_idx1 = int(parent_parts[1])
    net_idx2 = int(parent_parts[2])
    d_val = int(parent_parts[4])
    input_idx = int(path.stem)
    return net_idx1, net_idx2, d_val, input_idx

def get_interval_width(df, lb_col, ub_col):
    return df[ub_col] - df[lb_col]

def get_abs_max(df, lb_col, ub_col):
    return df[[lb_col, ub_col]].abs().max(axis=1)  # e.g., lb=-3, ub=2 -> 3

def get_csv_path(dir_path):
    csv_paths = []
    for root, dirs, files in os.walk(resolve_script_path(dir_path)):
        for filename in files:
            if filename.endswith('.csv'):
                csv_paths.append(os.path.join(root, filename))
                
    # sort by d, e
    # e.g., d1_e2_0.csv, d1_e2_1.csv, ..., d1_e3_0.csv, d1_e3_1.csv, ..., d2_e2_0.csv, ...
    ordered_csv_paths = sorted(csv_paths, key=lambda x: (int(re.search(r'd(\d+)_e', x).group(1)),
                                                          int(re.search(r'e(\d+)_', x).group(1)),
                                                          int(re.search(r'_(\d+)\.csv', x).group(1))))
    return ordered_csv_paths

def get_csv_path_acasxu(dir_path):
    csv_paths = []
    for root, dirs, files in os.walk(resolve_script_path(dir_path)):
        for filename in files:
            if filename.endswith('.csv'):
                csv_paths.append(os.path.join(root, filename))
                
    # sort by net_id1, net_id2, d, e
    # e.g., net_1_1_d_50/0.csv, net_1_1_d_50/1.csv, ..., net_1_1_d_100/0.csv, net_1_1_d_100/1.csv, ..., net_1_2_d_50/0.csv, ...
    ordered_csv_paths = sorted(csv_paths, key=lambda x: (int(re.search(r'net_(\d+)_', x).group(1)),
                                                          int(re.search(r'net_\d+_(\d+)_', x).group(1)),
                                                          int(re.search(r'd_(\d+)', x).group(1)),
                                                          int(re.search(r'/(\d+)\.csv', x).group(1))))
    return ordered_csv_paths

def csv_to_df(csv_path):
    df = pd.read_csv(csv_path)
    return df

def get_parent_row(row, seen_names):
    name = row['name']
    if "_" not in name:
        return None
    if name in seen_names:
        return name
    seen_names.add(name)
    return "_".join(name.split("_")[:-1])

def get_dfs(dataset_name, csv_path_ds, csv_path_dsrnd, csv_path_ns, csv_path_nsind):
    df_ds, df_dsrnd, df_ns, df_nsind = None, None, None, None
    if dataset_name == "acasxu":
        n1_ds, n2_ds, d_ds, idx_ds = extract_exp_info_from_csvpath_acasxu(csv_path_ds)
        n1_dsrnd, n2_dsrnd, d_dsrnd, idx_dsrnd = extract_exp_info_from_csvpath_acasxu(csv_path_dsrnd)
        n1_ns, n2_ns, d_ns, idx_ns = extract_exp_info_from_csvpath_acasxu(csv_path_ns)
        n1_nsind, n2_nsind, d_nsind, idx_nsind = extract_exp_info_from_csvpath_acasxu(csv_path_nsind)
        if not (n1_ds, n2_ds, d_ds, idx_ds) == (n1_dsrnd, n2_dsrnd, d_dsrnd, idx_dsrnd) == (n1_ns, n2_ns, d_ns, idx_ns) == (n1_nsind, n2_nsind, d_nsind, idx_nsind):
            print(f"ds: n1={n1_ds}, n2={n2_ds}, d={d_ds}, idx={idx_ds}")
            print(f"dsrnd: n1={n1_dsrnd}, n2={n2_dsrnd}, d={d_dsrnd}, idx={idx_dsrnd}")
            print(f"ns: n1={n1_ns}, n2={n2_ns}, d={d_ns}, idx={idx_ns}")
            print(f"nsind: n1={n1_nsind}, n2={n2_nsind}, d={d_nsind}, idx={idx_nsind}")
        assert (n1_ds, n2_ds, d_ds, idx_ds) == (n1_dsrnd, n2_dsrnd, d_dsrnd, idx_dsrnd) == (n1_ns, n2_ns, d_ns, idx_ns) == (n1_nsind, n2_nsind, d_nsind, idx_nsind), "Experiment parameters do not match among the CSV files."
        if (n1_ds, n2_ds, d_ds, idx_ds) in aca_instances:
            df_ds = csv_to_df(csv_path_ds)
            df_dsrnd = csv_to_df(csv_path_dsrnd)
            df_ns = csv_to_df(csv_path_ns)
            df_nsind = csv_to_df(csv_path_nsind)
    else:
        d_ds, i_ds, idx_ds = extract_exp_info_from_csvpath(csv_path_ds)
        d_dsrnd, i_dsrnd, idx_dsrnd = extract_exp_info_from_csvpath(csv_path_dsrnd)
        d_ns, i_ns, idx_ns = extract_exp_info_from_csvpath(csv_path_ns)
        d_nsind, i_nsind, idx_nsind = extract_exp_info_from_csvpath(csv_path_nsind)
        if not (d_ds, i_ds, idx_ds) == (d_dsrnd, i_dsrnd, idx_dsrnd) == (d_ns, i_ns, idx_ns) == (d_nsind, i_nsind, idx_nsind):
            print(f"ds: d={d_ds}, i={i_ds}, idx={idx_ds}")
            print(f"dsrnd: d={d_dsrnd}, i={i_dsrnd}, idx={idx_dsrnd}")
            print(f"ns: d={d_ns}, i={i_ns}, idx={idx_ns}")
            print(f"nsind: d={d_nsind}, i={i_nsind}, idx={idx_nsind}")
        assert (d_ds, i_ds, idx_ds) == (d_dsrnd, i_dsrnd, idx_dsrnd) == (d_ns, i_ns, idx_ns) == (d_nsind, i_nsind, idx_nsind), "Experiment parameters do not match among the CSV files."
        if dataset_name == "mnist4" and (d_ds, i_ds, idx_ds) in m4_instances:
            df_ds = csv_to_df(csv_path_ds)
            df_dsrnd = csv_to_df(csv_path_dsrnd)
            df_ns = csv_to_df(csv_path_ns)
            df_nsind = csv_to_df(csv_path_nsind)
        elif dataset_name == "mnist-conv" and (d_ds, i_ds, idx_ds) in mc_instances:
            df_ds = csv_to_df(csv_path_ds)
            df_dsrnd = csv_to_df(csv_path_dsrnd)
            df_ns = csv_to_df(csv_path_ns)
            df_nsind = csv_to_df(csv_path_nsind)
        elif dataset_name == "cifar10" and (d_ds, i_ds, idx_ds) in c10_instances:
            df_ds = csv_to_df(csv_path_ds)
            df_dsrnd = csv_to_df(csv_path_dsrnd)
            df_ns = csv_to_df(csv_path_ns)
            df_nsind = csv_to_df(csv_path_nsind)
        elif dataset_name == "gtsrb" and (d_ds, i_ds, idx_ds) in gt_instances:
            df_ds = csv_to_df(csv_path_ds)
            df_dsrnd = csv_to_df(csv_path_dsrnd)
            df_ns = csv_to_df(csv_path_ns)
            df_nsind = csv_to_df(csv_path_nsind)

    return df_ds, df_dsrnd, df_ns, df_nsind

def transition_of_subproblems(dataset_name, csv_path_ds, csv_path_dsrnd, csv_path_ns, csv_path_nsind, save_dir):
    df_ds, df_dsrnd, df_ns, df_nsind = get_dfs(dataset_name, csv_path_ds, csv_path_dsrnd, csv_path_ns, csv_path_nsind)
    if dataset_name == "acasxu":
        n1_ds, n2_ds, d_ds, idx_ds = extract_exp_info_from_csvpath_acasxu(csv_path_ds)
    else:
        d_ds, i_ds, idx_ds = extract_exp_info_from_csvpath(csv_path_ds)
    # ----------------------------
    # aggregate number of subproblems per level
    # ----------------------------
    def aggregate_num_subproblems(df):
        return (
            df.groupby("level", as_index=False)
            .agg(num_subproblems=("name", "count"))
            .sort_values("level")
        )

    agg_ds = aggregate_num_subproblems(df_ds)
    agg_dsrnd = aggregate_num_subproblems(df_dsrnd)
    agg_ns = aggregate_num_subproblems(df_ns)
    agg_nsind = aggregate_num_subproblems(df_nsind)

    # ----------------------------
    # plot
    # ----------------------------
    plt.figure(figsize=(10, 6))

    method_info = [
        (agg_ds, "DS", "blue", "o"),
        (agg_dsrnd, "DS_random", "red", "x"),
        (agg_ns, "NS_rel", "gray", "^"),
        (agg_nsind, "NS_ind", "green", "s"),
    ]

    legend_elems = []

    for agg_df, label, clr, marker in method_info:
        plt.plot(
            agg_df["level"],
            agg_df["num_subproblems"],
            linestyle="--",
            linewidth=1.5,
            color=clr,
            alpha=0.8,
        )
        plt.scatter(
            agg_df["level"],
            agg_df["num_subproblems"],
            s=40,
            color=clr,
            marker=marker,
            alpha=0.8,
        )

        legend_elems.append(
            Line2D(
                [0], [0],
                marker=marker,
                color=clr,
                linestyle="--",
                linewidth=1.5,
                markersize=7,
                label=label,
            )
        )

    plt.xlabel("Split level")
    plt.ylabel("Number of subproblems")

    if dataset_name == "acasxu":
        plt.title(f"Number of subproblems by split level (n1={n1_ds}, n2={n2_ds}, idx={idx_ds})")
    else:
        plt.title(f"Number of subproblems by split level (d={d_ds}, i={i_ds}, idx={idx_ds})")

    ax = plt.gca()
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    plt.legend(handles=legend_elems, loc="best", frameon=True)
    plt.grid(True)

    # save figure
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if dataset_name == "acasxu":
            plt.savefig(os.path.join(save_dir, f"{n1_ds}_{n2_ds}_{idx_ds}_num_problems.png"))
        else:
            plt.savefig(os.path.join(save_dir, f"d{d_ds}_e{i_ds}_{idx_ds}_num_problems.png"))

    plt.show()
    plt.close()

def draw_line_graph(csv_path_ds, csv_path_dsrnd, csv_path_ns, csv_path_nsind, save_dir, dataset_name, threshold_line=False):
    method_specs = [
        ("DS", csv_path_ds, "blue", "o"),
        ("DS_random", csv_path_dsrnd, "red", "x"),
        ("NS_rel", csv_path_ns, "gray", "^"),
        ("NS_ind", csv_path_nsind, "green", "s"),
    ]
    active_methods = [(label, path, color, marker) for label, path, color, marker in method_specs if path is not None]
    if len(active_methods) == 0:
        raise ValueError("All CSV paths are None. Provide at least one CSV path.")

    base_label, base_path, _, _ = active_methods[0]
    if dataset_name == "acasxu":
        n1_ds, n2_ds, d_ds, idx_ds = extract_exp_info_from_csvpath_acasxu(base_path)
    else:
        d_ds, i_ds, idx_ds = extract_exp_info_from_csvpath(base_path)

    method_dfs = []
    for label, path, color, marker in active_methods:
        if dataset_name == "acasxu":
            n1, n2, d, idx = extract_exp_info_from_csvpath_acasxu(path)
            if (n1, n2, d, idx) != (n1_ds, n2_ds, d_ds, idx_ds):
                raise ValueError(
                    f"Experiment parameters do not match: {label} has "
                    f"(n1={n1}, n2={n2}, d={d}, idx={idx}), expected "
                    f"(n1={n1_ds}, n2={n2_ds}, d={d_ds}, idx={idx_ds})."
                )
        else:
            d, i, idx = extract_exp_info_from_csvpath(path)
            if (d, i, idx) != (d_ds, i_ds, idx_ds):
                raise ValueError(
                    f"Experiment parameters do not match: {label} has "
                    f"(d={d}, i={i}, idx={idx}), expected "
                    f"(d={d_ds}, i={i_ds}, idx={idx_ds})."
                )

        temp_df = csv_to_df(path).copy()
        abs_max = get_abs_max(temp_df, 'lb', 'ub')
        temp_df['range_ratio'] = abs_max / abs_max.iloc[0]
        method_dfs.append((temp_df, label, color, marker))

    plt.figure(figsize=(10,6))
    legend_elems = []
    for temp_df, label, clr, shape in method_dfs:
        seen_names = set()
        for row_idx, row in temp_df.iterrows():
            temp_df.loc[row_idx, 'parent'] = get_parent_row(row, seen_names)

        plt.scatter(temp_df['level'], temp_df['range_ratio'], s=30, color=clr, alpha=0.6, label=label, marker=shape)
        
        for _, row in temp_df.iterrows():
            if row["parent"] is not None:
                if row["parent"] == "DS" or row["parent"] == "NS":
                    parent = temp_df.iloc[0]  # first row
                else:
                    parent_candidates = temp_df[
                        (temp_df["name"] == row["parent"]) &
                        (temp_df["level"] == row["level"] - 1)
                    ]
                    if parent_candidates.empty:
                        continue
                    parent = parent_candidates.iloc[0]
                plt.plot([parent["level"], row["level"]], [parent["range_ratio"], row["range_ratio"]], color=clr, linestyle='--', linewidth=1, alpha=0.4)
        
        legend_elems.append(
                    Line2D([0], [0], marker=shape, color='w', markerfacecolor=clr,
                    markeredgecolor=clr, label=f"{label} (points)")
                    )
        legend_elems.append(
                    Line2D([0], [0], color=clr, linestyle='--', linewidth=1,
                    label=f"{label} (lines)")
                    )

    if threshold_line:
        if dataset_name == "acasxu":
            threshold_ratio = get_threshold_ratio(dataset_name, None, None, idx_ds, net1=n1_ds, net2=n2_ds)
        else:
            threshold_ratio = get_threshold_ratio(dataset_name, d_ds, i_ds, idx_ds)
        plt.axhline(y=threshold_ratio, color='orange', linestyle='-', label='Threshold', linewidth=1)
        legend_elems.append(Line2D([0], [0], color='orange', linestyle='-', linewidth=1,
                            label=f"Threshold ({threshold_ratio:.3f})"))
            
    plt.xlabel("Split level")
    plt.ylabel("Difference ratio")
    if dataset_name == "acasxu":
        plt.title(f"Difference ratio transition(n1={n1_ds}, n2={n2_ds}, idx={idx_ds})")
    else:
        plt.title(f"Difference ratio transition(d={d_ds}, i={i_ds}, idx={idx_ds})")

    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)
    
    plt.legend(handles=legend_elems, loc="best", frameon=True)

    plt.grid(True)
    # save figure
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if dataset_name == "acasxu":
            plt.savefig(os.path.join(save_dir, f"{n1_ds}_{n2_ds}_{idx_ds}.png"))
        else:
            plt.savefig(os.path.join(save_dir, f"d{d_ds}_e{i_ds}_{idx_ds}.png"))
    plt.show()

    plt.close()
    return

import pandas as pd


def get_leaf_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leaf nodes = rows at the maximum depth.
    """
    max_level = df["level"].max()
    return df[df["level"] == max_level].copy()


def get_time_budget(net_name, cifd=None):
    if net_name == 'acasxu':
        time_budget = 420  # seconds
    elif net_name == 'mnist4' or net_name == 'mnist-conv':
        time_budget = 600  # seconds
    elif net_name == 'cifar10':
        assert cifd == 1 or cifd == 2 or cifd == 3, f"cifd should be 1 or 2 or 3. Got {cifd}"
        time_budget = 1800 if cifd == 1 else 3600 if cifd == 2 else 7200  # seconds
    elif net_name == 'gtsrb':
        assert cifd == 1 or cifd == 2 or cifd == 3, f"cifd should be 1 or 2 or 3. Got {cifd}"
        time_budget = 1800 if cifd == 1 else 3600 if cifd == 2 else 7200  # seconds
    else:
        raise ValueError("Unknown net_name")
    return time_budget

def get_status_from_summary_df(net_name, method_name, n1, n2, d, e, idx):
    if net_name == "acasxu":
        summary_df = acasxu
        row = f"{n1}_{n2}_{d}_{idx}"
        time_budget = get_time_budget(net_name)
    elif net_name == "mnist4":
        summary_df = mnist4
        row = f"{d}_{e}_{idx}"
        time_budget = get_time_budget(net_name)
    elif net_name == "mnist-conv":
        summary_df = mnist_conv
        row = f"{d}_{e}_{idx}"
        time_budget = get_time_budget(net_name)
    elif net_name == "cifar10":
        summary_df = cifar10
        row = f"{d}_{e}_{idx}"
        time_budget = get_time_budget(net_name, d)
    elif net_name == "gtsrb":
        summary_df = gtsrb
        row = f"{d}_{e}_{idx}"
        time_budget = get_time_budget(net_name, d)

    else:
        raise ValueError(f"Unknown net_name: {net_name}")
    
    method_name_map = {
        "DS": "DSZ",
        "DS_random": "RndZ",
        "NS_rel": "NS",
        "NS_ind": "NSInd",
    }
    
    time = summary_df.loc[summary_df['name'] == row, f'{method_name_map[method_name]} time'].values[0]
    status = summary_df.loc[summary_df['name'] == row, f'{method_name_map[method_name]} status'].values[0]
    if time > time_budget:
        return "UNKNOWN"
    else:
        return status


def get_status_from_leaf_df(df: pd.DataFrame, net_name: str, cifd: int = None) -> str:
    """
    Priority:
        ADV_EXAMPLE if any leaf is ADV_EXAMPLE
        UNKNOWN     if any leaf is UNKNOWN
        VERIFIED    otherwise
    """
    leaf_df = get_leaf_df(df)
    statuses = set(leaf_df["status"].astype(str))
    time_budget = get_time_budget(df, net_name, cifd)
    time_exceeded = leaf_df["time"].max() > time_budget
    if time_exceeded:
        return "UNKNOWN"

    if "ADV_EXAMPLE" in statuses or "Status.ADV_EXAMPLE" in statuses:
        return "ADV_EXAMPLE"
    elif "UNKNOWN" in statuses or "Status.UNKNOWN" in statuses:
        return "UNKNOWN"
    else:
        return "VERIFIED"


def get_instance_method_statuses(dataset_name, csv_path_ds, csv_path_dsrnd, csv_path_ns, csv_path_nsind):
    """
    Return leaf-based final statuses for the 4 approaches on one instance.
    """
    df_ds, df_dsrnd, df_ns, df_nsind = get_dfs(
        dataset_name, csv_path_ds, csv_path_dsrnd, csv_path_ns, csv_path_nsind
    )

    if any(df is None for df in [df_ds, df_dsrnd, df_ns, df_nsind]):
        return None
    
    if dataset_name == "acasxu":
        n1, n2, d, idx = extract_exp_info_from_csvpath_acasxu(csv_path_ds)
        e = None
        cifd = None
    else:
        d, e, idx = extract_exp_info_from_csvpath(csv_path_ds)
        n1, n2 = None, None

    return {
        "DS": get_status_from_summary_df(dataset_name, "DS", n1, n2, d, e, idx),
        "DS_random": get_status_from_summary_df(dataset_name, "DS_random", n1, n2, d, e, idx),
        "NS_rel": get_status_from_summary_df(dataset_name, "NS_rel", n1, n2, d, e, idx),
        "NS_ind": get_status_from_summary_df(dataset_name, "NS_ind", n1, n2, d, e, idx),
    }


def filter_instance_paths_by_status(
    dataset_name,
    csv_paths_ds,
    csv_paths_dsrnd,
    csv_paths_ns,
    csv_paths_nsind,
    target_status="VERIFIED",
    mode="any",   # "any" or "all"
):
    """
    Filter instances based on leaf-based final statuses.

    mode="any":
        keep instance if any approach has target_status

    mode="all":
        keep instance if all approaches have target_status

    Returns filtered path lists:
        filtered_ds, filtered_dsrnd, filtered_ns, filtered_nsind
    """
    filtered_ds = []
    filtered_dsrnd = []
    filtered_ns = []
    filtered_nsind = []

    for csv_path_ds, csv_path_dsrnd, csv_path_ns, csv_path_nsind in zip(
        csv_paths_ds, csv_paths_dsrnd, csv_paths_ns, csv_paths_nsind
    ):
        statuses = get_instance_method_statuses(
            dataset_name, csv_path_ds, csv_path_dsrnd, csv_path_ns, csv_path_nsind
        )

        if statuses is None:
            continue

        values = list(statuses.values())

        if mode == "any":
            keep = any(s == target_status for s in values)
        elif mode == "all":
            keep = all(s == target_status for s in values)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if keep:
            filtered_ds.append(csv_path_ds)
            filtered_dsrnd.append(csv_path_dsrnd)
            filtered_ns.append(csv_path_ns)
            filtered_nsind.append(csv_path_nsind)

    return filtered_ds, filtered_dsrnd, filtered_ns, filtered_nsind

def per_instance_level_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    For one instance / one method:
    compute mean range_ratio for each level.

    Returns columns:
        level, mean_range_ratio
    """
    df = df.copy()
    abs_max = get_abs_max(df, "lb", "ub")
    df["range_ratio"] = abs_max / abs_max.iloc[0]

    result = (
        df.groupby("level", as_index=False)
          .agg(mean_range_ratio=("range_ratio", "mean"))
          .sort_values("level")
    )
    return result

def per_instance_level_min(df: pd.DataFrame) -> pd.DataFrame:
    """
    For one instance / one method:
    compute min range_ratio for each level.

    Returns columns:
        level, min_range_ratio
    """
    df = df.copy()
    abs_max = get_abs_max(df, "lb", "ub")
    df["range_ratio"] = abs_max / abs_max.iloc[0]

    result = (
        df.groupby("level", as_index=False)
          .agg(min_range_ratio=("range_ratio", "min"))
          .sort_values("level")
    )
    return result


def summarize_across_instances(level_mean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns:
        instance_id, level, mean_range_ratio

    Output columns:
        level, n_instances, median, q1, q3, mean
    """
    summary = (
        level_mean_df.groupby("level", as_index=False)
        .agg(
            n_instances=("instance_id", "count"),
            median=("mean_range_ratio", "median"),
            q1=("mean_range_ratio", lambda x: x.quantile(0.25)),
            q3=("mean_range_ratio", lambda x: x.quantile(0.75)),
            mean=("mean_range_ratio", "mean"),
        )
        .sort_values("level")
    )
    return summary


def collect_level_means_across_instances(dataset_name, csv_paths_ds, csv_paths_dsrnd, csv_paths_ns, csv_paths_nsind):
    """
    For all instances, collect per-instance per-level mean_range_ratio
    for each method.

    Returns:
        ds_all, dsrnd_all, ns_all, nsind_all
    Each dataframe has:
        instance_id, level, mean_range_ratio
    """
    rows_ds = []
    rows_dsrnd = []
    rows_ns = []
    rows_nsind = []

    for csv_path_ds, csv_path_dsrnd, csv_path_ns, csv_path_nsind in zip(
        csv_paths_ds, csv_paths_dsrnd, csv_paths_ns, csv_paths_nsind
    ):
        df_ds, df_dsrnd, df_ns, df_nsind = get_dfs(
            dataset_name, csv_path_ds, csv_path_dsrnd, csv_path_ns, csv_path_nsind
        )

        # skipped instances return None
        if any(df is None for df in [df_ds, df_dsrnd, df_ns, df_nsind]):
            continue

        if dataset_name == "acasxu":
            n1, n2, d, idx = extract_exp_info_from_csvpath_acasxu(csv_path_ds)
            instance_id = f"net_{n1}_{n2}_d_{d}_{idx}"
        else:
            d, e, idx = extract_exp_info_from_csvpath(csv_path_ds)
            instance_id = f"d{d}_e{e}_{idx}"

        for df_method, rows in zip(
            [df_ds, df_dsrnd, df_ns, df_nsind],
            [rows_ds, rows_dsrnd, rows_ns, rows_nsind]
        ):
            temp = per_instance_level_min(df_method)
            temp["instance_id"] = instance_id
            rows.append(temp)

    ds_all = pd.concat(rows_ds, ignore_index=True) if rows_ds else pd.DataFrame(columns=["instance_id", "level", "min_range_ratio"])
    dsrnd_all = pd.concat(rows_dsrnd, ignore_index=True) if rows_dsrnd else pd.DataFrame(columns=["instance_id", "level", "min_range_ratio"])
    ns_all = pd.concat(rows_ns, ignore_index=True) if rows_ns else pd.DataFrame(columns=["instance_id", "level", "min_range_ratio"])
    nsind_all = pd.concat(rows_nsind, ignore_index=True) if rows_nsind else pd.DataFrame(columns=["instance_id", "level", "min_range_ratio"])

    return ds_all, dsrnd_all, ns_all, nsind_all


def plot_merged_line_graph(dataset_name, csv_paths_ds, csv_paths_dsrnd, csv_paths_ns, csv_paths_nsind, save_dir=None):
    """
    Merge all instances:
      - per instance: average range_ratio by level
      - across instances: compute median and IQR
      - plot one curve per method with IQR shading
    """

    # ---- font + style settings ----
    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 2.5,
    })

    ds_all, dsrnd_all, ns_all, nsind_all = collect_level_means_across_instances(
        dataset_name, csv_paths_ds, csv_paths_dsrnd, csv_paths_ns, csv_paths_nsind
    )

    ds_summary = summarize_across_instances(ds_all)
    dsrnd_summary = summarize_across_instances(dsrnd_all)
    ns_summary = summarize_across_instances(ns_all)
    nsind_summary = summarize_across_instances(nsind_all)

    plt.figure(figsize=(10, 6))

    
    method_info = [
        (ds_summary, SABRE_DISPLAY_NAME, "blue"),
        # (dsrnd_summary, "DS_random", "red"),
        # (ns_summary, "NS_rel", "gray"),
        (nsind_summary, "ClasIS", "green"),
    ]

    dataset_dict = {
        "acasxu": "ACAS Xu",
        "mnist4": "MNIST-F",
        "mnist-conv": "MNIST-C",
        "cifar10": "CIFAR",
    }
    dataset_display_name = dataset_dict.get(dataset_name, dataset_name)

    for summary_df, label, color in method_info:
        if summary_df.empty:
            continue

        x = summary_df["level"]
        # y = summary_df["mean"]   # use median as central curve
        y = summary_df["median"]   # use median as central curve
        y1 = summary_df["q1"]
        y2 = summary_df["q3"]

        plt.plot(x, y, marker="o", linewidth=2, label=label, color=color)
        plt.fill_between(x, y1, y2, alpha=0.2, color=color)

    plt.xlabel("Split level")
    plt.ylabel("Relational distance ratio")
    # plt.title(f"Transition of relational distance ratio by split level ({dataset_display_name})")
    plt.title(f"{dataset_display_name}")
    plt.grid(True)
    plt.legend(loc="best", frameon=True)

    ax = plt.gca()
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_merged_range_ratio.png"), bbox_inches="tight")

    plt.show()
    plt.close()

    # return ds_summary, dsrnd_summary, ns_summary, nsind_summary

def plot_line_graph_acasxu_merged():
    folder_path = "./acasxu/"
    dataset_name = "acasxu"

    csv_dir_ds = f"{folder_path}DS_dual_Z_threshold/"
    csv_dir_dsrnd = f"{folder_path}DS_random_Z_threshold/"
    csv_dir_nsind = f"{folder_path}NS_dual_ind_threshold/"
    csv_dir_ns = f"{folder_path}NS_dual_threshold/"

    csv_paths_ds = get_csv_path_acasxu(csv_dir_ds)
    csv_paths_dsrnd = get_csv_path_acasxu(csv_dir_dsrnd)
    csv_paths_nsind = get_csv_path_acasxu(csv_dir_nsind)
    csv_paths_ns = get_csv_path_acasxu(csv_dir_ns)

    csv_paths_ds, csv_paths_dsrnd, csv_paths_ns, csv_paths_nsind = \
        filter_instance_paths_by_status(
            dataset_name=dataset_name,
            csv_paths_ds=csv_paths_ds,
            csv_paths_dsrnd=csv_paths_dsrnd,
            csv_paths_ns=csv_paths_ns,
            csv_paths_nsind=csv_paths_nsind,
            target_status="VERIFIED",
            mode="any"
        )

    return plot_merged_line_graph(
        dataset_name=dataset_name,
        csv_paths_ds=csv_paths_ds,
        csv_paths_dsrnd=csv_paths_dsrnd,
        csv_paths_ns=csv_paths_ns,
        csv_paths_nsind=csv_paths_nsind,
        save_dir=None,  # or "./acasxu/figures/"
    )

def plot_line_graph_merged(folder_path, dataset_name):

    csv_dir_ds = f"{folder_path}DS_dual_Z_threshold/"
    csv_dir_dsrnd = f"{folder_path}DS_random_Z_threshold/"
    csv_dir_nsind = f"{folder_path}NS_dual_ind_threshold/"
    csv_dir_ns = f"{folder_path}NS_dual_threshold/"

    csv_paths_ds = get_csv_path(csv_dir_ds)
    csv_paths_dsrnd = get_csv_path(csv_dir_dsrnd)
    csv_paths_nsind = get_csv_path(csv_dir_nsind)
    csv_paths_ns = get_csv_path(csv_dir_ns)

    csv_paths_ds, csv_paths_dsrnd, csv_paths_ns, csv_paths_nsind = \
        filter_instance_paths_by_status(
            dataset_name=dataset_name,
            csv_paths_ds=csv_paths_ds,
            csv_paths_dsrnd=csv_paths_dsrnd,
            csv_paths_ns=csv_paths_ns,
            csv_paths_nsind=csv_paths_nsind,
            target_status="VERIFIED",
            mode="any"
        )

    return plot_merged_line_graph(
        dataset_name=dataset_name,
        csv_paths_ds=csv_paths_ds,
        csv_paths_dsrnd=csv_paths_dsrnd,
        csv_paths_ns=csv_paths_ns,
        csv_paths_nsind=csv_paths_nsind,
        save_dir=None,  # or "./acasxu/figures/"
    )

'''
we will compare Sabre and ClasIS, so we can ignore DS_random and NS_rel for now.
per instance:
    - compute mean range_ratio by level for each method
    - get difference mean_range_ratio(approach1 - approach2) by level
        - if either mean_range_ratio is NaN, stop computing further levels
aggregate across instances:
    - for each level, compute median and IQR of the difference_mean_range_ratio across instances
plot:
    - one curve for median difference_mean_range_ratio by level, with IQR shading
'''
def per_instance_level_diff(df_ds: pd.DataFrame, df_nsind: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-instance difference:
        diff = SABRE - ClasIS

    Stop at first level where either side is NaN.

    Returns:
        level, diff_range_ratio
    """
    ds_mean = per_instance_level_min(df_ds)
    nsind_mean = per_instance_level_min(df_nsind)

    merged = pd.merge(
        ds_mean, nsind_mean,
        on="level",
        how="inner",
        suffixes=("_ds", "_nsind")
    ).sort_values("level")

    diffs = []
    for _, row in merged.iterrows():
        v1 = row["min_range_ratio_ds"]
        v2 = row["min_range_ratio_nsind"]

        if pd.isna(v1) or pd.isna(v2):
            break  # stop further levels

        diffs.append({
            "level": row["level"],
            "diff_range_ratio": v1 - v2
        })

    return pd.DataFrame(diffs)

def collect_level_diffs_across_instances(
    dataset_name,
    csv_paths_ds,
    csv_paths_dsrnd,
    csv_paths_ns,
    csv_paths_nsind
):
    """
    Collect per-instance level differences.

    Returns:
        dataframe with columns:
            instance_id, level, diff_range_ratio
    """
    rows = []

    for csv_path_ds, csv_path_dsrnd, csv_path_ns, csv_path_nsind in zip(csv_paths_ds, csv_paths_dsrnd, csv_paths_ns, csv_paths_nsind):

        df_ds, df_dsrnd, df_ns, df_nsind = get_dfs(
            dataset_name,
            csv_path_ds,
            csv_path_dsrnd,
            csv_path_ns,
            csv_path_nsind
        )
        if df_ds is None or df_nsind is None:
            continue

        # instance id
        if dataset_name == "acasxu":
            n1, n2, d, idx = extract_exp_info_from_csvpath_acasxu(csv_path_ds)
            instance_id = f"net_{n1}_{n2}_d_{d}_{idx}"
        else:
            d, e, idx = extract_exp_info_from_csvpath(csv_path_ds)
            instance_id = f"d{d}_e{e}_{idx}"
        
        # if d == 1:  # for CIFAR10, we only have d=1,2
        #     continue

        # currently we only compare SABRE and ClasIS, so we ignore DS_random and NS_rel for the diff computation
        diff_df = per_instance_level_diff(df_ds, df_nsind)
        diff_df["instance_id"] = instance_id

        rows.append(diff_df)

    if rows:
        return pd.concat(rows, ignore_index=True)
    else:
        return pd.DataFrame(columns=["instance_id", "level", "diff_range_ratio"])
    
def summarize_diff_across_instances(diff_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate diff across instances.

    Output:
        level, n_instances, median, q1, q3, mean
    """
    return (
        diff_df.groupby("level", as_index=False)
        .agg(
            n_instances=("instance_id", "count"),
            median=("diff_range_ratio", "median"),
            q1=("diff_range_ratio", lambda x: x.quantile(0.25)),
            q3=("diff_range_ratio", lambda x: x.quantile(0.75)),
            mean=("diff_range_ratio", "mean"),
        )
        .sort_values("level")
    )

def plot_diff_curve(
    dataset_name,
    csv_paths_ds,
    csv_paths_dsrnd,
    csv_paths_ns,
    csv_paths_nsind,
    save_dir=None
):
    """
    Plot SABRE - ClasIS difference curve with IQR shading.
    """

    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 2.5,
    })

    diff_all = collect_level_diffs_across_instances(
        dataset_name,
        csv_paths_ds,
        csv_paths_dsrnd,
        csv_paths_ns,
        csv_paths_nsind
    )

    summary = summarize_diff_across_instances(diff_all)

    if summary.empty:
        print("No data to plot.")
        return

    x = summary["level"]
    y = summary["mean"]
    # y = summary["median"]
    y1 = summary["q1"]
    y2 = summary["q3"]

    plt.figure(figsize=(10, 6))

    plt.plot(x, y, marker="o", label="SaBRe - ClasIS (mean)")
    plt.fill_between(x, y1, y2, alpha=0.2)

    plt.axhline(0, linestyle="--")  # important baseline

    dataset_dict = {
        "acasxu": "ACAS Xu",
        "mnist4": "MNIST-F",
        "mnist-conv": "MNIST-C",
        "cifar10": "CIFAR",
    }
    dataset_display_name = dataset_dict.get(dataset_name, dataset_name)

    plt.xlabel("Split level")
    plt.ylabel("Difference in relational distance ratio")
    plt.title(f"{dataset_display_name} (SaBRe - ClasIS)")
    plt.grid(True)
    plt.legend()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"{dataset_name}_diff_curve.png"),
            bbox_inches="tight"
        )

    plt.show()
    plt.close()

def plot_diff_acasxu():
    folder_path = "./acasxu/"
    dataset_name = "acasxu"

    csv_dir_ds = f"{folder_path}DS_dual_Z_threshold/"
    csv_dir_dsrnd = f"{folder_path}DS_random_Z_threshold/"
    csv_dir_nsind = f"{folder_path}NS_dual_ind_threshold/"
    csv_dir_ns = f"{folder_path}NS_dual_threshold/"

    csv_paths_ds = get_csv_path_acasxu(csv_dir_ds)
    csv_paths_dsrnd = get_csv_path_acasxu(csv_dir_dsrnd)
    csv_paths_nsind = get_csv_path_acasxu(csv_dir_nsind)
    csv_paths_ns = get_csv_path_acasxu(csv_dir_ns)

    # reuse your filtering
    csv_paths_ds, csv_paths_dsrnd, csv_paths_ns, csv_paths_nsind = filter_instance_paths_by_status(
        dataset_name=dataset_name,
        csv_paths_ds=csv_paths_ds,
        csv_paths_dsrnd=csv_paths_dsrnd,
        csv_paths_ns=csv_paths_ns,
        csv_paths_nsind=csv_paths_nsind,
        target_status="VERIFIED",
        mode="any"
    )

    plot_diff_curve(
        dataset_name,
        csv_paths_ds,
        csv_paths_dsrnd,
        csv_paths_ns,
        csv_paths_nsind
    )

def plot_diff(folder_path, dataset_name):
    csv_dir_ds = f"{folder_path}DS_dual_Z_threshold/"
    csv_dir_dsrnd = f"{folder_path}DS_random_Z_threshold/"
    csv_dir_nsind = f"{folder_path}NS_dual_ind_threshold/"
    csv_dir_ns = f"{folder_path}NS_dual_threshold/"

    csv_paths_ds = get_csv_path(csv_dir_ds)
    csv_paths_dsrnd = get_csv_path(csv_dir_dsrnd)
    csv_paths_nsind = get_csv_path(csv_dir_nsind)
    csv_paths_ns = get_csv_path(csv_dir_ns)

    # reuse your filtering
    # csv_paths_ds, csv_paths_dsrnd, csv_paths_ns, csv_paths_nsind = filter_instance_paths_by_status(
    #     dataset_name=dataset_name,
    #     csv_paths_ds=csv_paths_ds,
    #     csv_paths_dsrnd=csv_paths_dsrnd,
    #     csv_paths_ns=csv_paths_ns,
    #     csv_paths_nsind=csv_paths_nsind,
    #     target_status="VERIFIED",
    #     mode="any"
    # )

    plot_diff_curve(
        dataset_name,
        csv_paths_ds,
        csv_paths_dsrnd,
        csv_paths_ns,
        csv_paths_nsind
    )


def get_depth(df: pd.DataFrame) -> int:
    return int(df["level"].max())

def summarize_instance(
    dataset_name,
    csv_path_ds,
    csv_path_dsrnd,
    csv_path_ns,
    csv_path_nsind,
    filter_d=None
):
    df_ds, df_dsrnd, df_ns, df_nsind = get_dfs(
        dataset_name,
        csv_path_ds,
        csv_path_dsrnd,
        csv_path_ns,
        csv_path_nsind
    )

    if any(df is None for df in [df_ds, df_dsrnd, df_ns, df_nsind]):
        return None

    # instance id
    if dataset_name == "acasxu":
        n1, n2, d, idx = extract_exp_info_from_csvpath_acasxu(csv_path_ds)
        instance_id = f"net_{n1}_{n2}_d_{d}_{idx}"
        e = None
    else:
        d, e, idx = extract_exp_info_from_csvpath(csv_path_ds)
        instance_id = f"d{d}_e{e}_{idx}"
        n1, n2 = None, None
        if filter_d is not None and d != filter_d:
            return None
        # if d != 1:
        #     return None

    result = {
        "instance_id": instance_id,

        # DS
        "DS_status": get_status_from_summary_df(dataset_name, "DS", n1, n2, d, e, idx),
        "DS_depth": get_depth(df_ds),

        # DS_random
        "DS_random_status": get_status_from_summary_df(dataset_name, "DS_random", n1, n2, d, e, idx),
        "DS_random_depth": get_depth(df_dsrnd),

        # NS_rel
        "NS_rel_status": get_status_from_summary_df(dataset_name, "NS_rel", n1, n2, d, e, idx),
        "NS_rel_depth": get_depth(df_ns),

        # NS_ind
        "NS_ind_status": get_status_from_summary_df(dataset_name, "NS_ind", n1, n2, d, e, idx),
        "NS_ind_depth": get_depth(df_nsind),
    }

    return result

def collect_instance_summaries(
    dataset_name,
    csv_paths_ds,
    csv_paths_dsrnd,
    csv_paths_ns,
    csv_paths_nsind,
    filter_d=None
):
    rows = []

    for paths in zip(csv_paths_ds, csv_paths_dsrnd, csv_paths_ns, csv_paths_nsind):
        summary = summarize_instance(dataset_name, *paths, filter_d=filter_d)
        if summary is not None:
            rows.append(summary)

    return pd.DataFrame(rows)

def status_depth_df_acasxu():
    folder_path = "./acasxu/"
    dataset_name = "acasxu"

    csv_dir_ds = f"{folder_path}DS_dual_Z_threshold/"
    csv_dir_dsrnd = f"{folder_path}DS_random_Z_threshold/"
    csv_dir_nsind = f"{folder_path}NS_dual_ind_threshold/"
    csv_dir_ns = f"{folder_path}NS_dual_threshold/"

    csv_paths_ds = get_csv_path_acasxu(csv_dir_ds)
    csv_paths_dsrnd = get_csv_path_acasxu(csv_dir_dsrnd)
    csv_paths_nsind = get_csv_path_acasxu(csv_dir_nsind)
    csv_paths_ns = get_csv_path_acasxu(csv_dir_ns)

    # filter by status
    csv_paths_ds, csv_paths_dsrnd, csv_paths_ns, csv_paths_nsind = filter_instance_paths_by_status(
        dataset_name=dataset_name,
        csv_paths_ds=csv_paths_ds,
        csv_paths_dsrnd=csv_paths_dsrnd,
        csv_paths_ns=csv_paths_ns,
        csv_paths_nsind=csv_paths_nsind,
        target_status="VERIFIED",
        mode="any"
    )

    df_summary = collect_instance_summaries(
        dataset_name,
        csv_paths_ds,
        csv_paths_dsrnd,
        csv_paths_ns,
        csv_paths_nsind
    )

    return df_summary

def status_depth_df(folder_path, dataset_name, filter_d=None):
    if dataset_name == "gtsrb":
        csv_dir_ds = f"{folder_path}RS_dual_Z_threshold/"
        csv_dir_dsrnd = f"{folder_path}RS_random_Z_threshold/"
        csv_dir_nsind = f"{folder_path}IS_dual_ind_threshold/"
        csv_dir_ns = f"{folder_path}IS_dual_threshold/"
    else:
        csv_dir_ds = f"{folder_path}DS_dual_Z_threshold/"
        csv_dir_dsrnd = f"{folder_path}DS_random_Z_threshold/"
        csv_dir_nsind = f"{folder_path}NS_dual_ind_threshold/"
        csv_dir_ns = f"{folder_path}NS_dual_threshold/"

    csv_paths_ds = get_csv_path(csv_dir_ds)
    csv_paths_dsrnd = get_csv_path(csv_dir_dsrnd)
    csv_paths_nsind = get_csv_path(csv_dir_nsind)
    csv_paths_ns = get_csv_path(csv_dir_ns)

    # filter by status
    csv_paths_ds, csv_paths_dsrnd, csv_paths_ns, csv_paths_nsind = filter_instance_paths_by_status(
        dataset_name=dataset_name,
        csv_paths_ds=csv_paths_ds,
        csv_paths_dsrnd=csv_paths_dsrnd,
        csv_paths_ns=csv_paths_ns,
        csv_paths_nsind=csv_paths_nsind,
        target_status="VERIFIED",
        mode="any"
    )

    df_summary = collect_instance_summaries(
        dataset_name,
        csv_paths_ds,
        csv_paths_dsrnd,
        csv_paths_ns,
        csv_paths_nsind,
        filter_d=filter_d
    )

    return df_summary



def get_global_max_level(df: pd.DataFrame):
    depth_cols = [c for c in df.columns if c.endswith("_depth")]
    return int(df[depth_cols].max().max())

def build_solved_curve_from_summary(df: pd.DataFrame, method_prefix: str, max_level: int):
    status_col = f"{method_prefix}_status"
    depth_col = f"{method_prefix}_depth"

    df_method = df[["instance_id", status_col, depth_col]].copy()

    # only VERIFIED
    df_verified = df_method[df_method[status_col] == "VERIFIED"]

    rows = []
    for l in range(max_level + 1):
        count = (df_verified[depth_col] <= l).sum()
        rows.append({"level": l, "count": count})

    return pd.DataFrame(rows)

def build_bar_data(df_summary: pd.DataFrame):

    max_level = get_global_max_level(df_summary)

    df_ds = build_solved_curve_from_summary(df_summary, "DS", max_level)
    df_nsind = build_solved_curve_from_summary(df_summary, "NS_ind", max_level)
    df_ns = build_solved_curve_from_summary(df_summary, "NS_rel", max_level)

    df_merged = df_ds.rename(columns={"count": "DS"}) \
        .merge(df_nsind.rename(columns={"count": "NS_ind"}), on="level") \
        .merge(df_ns.rename(columns={"count": "NS"}), on="level")

    return df_merged.sort_values("level")

def plot_bar_solved(df_bar: pd.DataFrame, dataset_name: str):

    levels = df_bar["level"].values
    x = np.arange(len(levels))

    width = 0.25  # bar width

    # ---- font settings (global) ----
    base_font_size = 26
    plt.rcParams.update({
        "font.size": base_font_size,
        "axes.labelsize": base_font_size + 2,
        "axes.titlesize": base_font_size + 2,
        "xtick.labelsize": base_font_size,
        "ytick.labelsize": base_font_size,
        "legend.fontsize": base_font_size,
    })

    plt.figure(figsize=(12, 6))

    # plt.bar(x - width, df_bar["DS"], width, label="SABRE")
    # plt.bar(x, df_bar["NS_ind"], width, label="ClasIS")
    # plt.bar(x + width, df_bar["NS"], width, label="DualIS")

    order = ["NS_ind", "NS", "DS"]
    labels = {
        "NS_ind": "ClasIS",
        "NS": "DualIS",
        "DS": SABRE_DISPLAY_NAME,
    }

    for i, key in enumerate(order):
        plt.bar(x + (i - 1) * width, df_bar[key], width, label=labels[key])

    plt.xticks(x, levels)

    plt.xlabel("Split level")
    plt.ylabel("# solved instances")
    # plt.title(dataset_name)

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

def drop_trailing_constant_rows(df: pd.DataFrame):
    cols = ["DS", "NS_ind", "NS"]

    # detect change compared to previous row
    changed = df[cols].diff().fillna(0).ne(0).any(axis=1)

    # always keep first row
    changed.iloc[0] = True

    # find last row where change occurs
    last_change_idx = changed[changed].index[-1]

    # keep up to that row
    return df.loc[:last_change_idx].copy()

def plot_bar_solved_all(filter_d=None):
    aca_df = aca_df = status_depth_df_acasxu()
    m4_df = status_depth_df(
        folder_path = "./mnist-256x4/",
        dataset_name = "mnist4",
        filter_d=filter_d
    )
    mc_df = status_depth_df(
        folder_path = "./mnist-conv/",
        dataset_name = "mnist-conv",
        filter_d=filter_d
    )
    cif_df = status_depth_df(
        folder_path = "./cifar10/",
        dataset_name = "cifar10",
        filter_d=filter_d
    )
    gt_df = status_depth_df(
        folder_path = "./gtsrb/",
        dataset_name = "gtsrb",
        filter_d=filter_d
    )


    filtered_bar_data_aca = drop_trailing_constant_rows(build_bar_data(aca_df))
    filtered_bar_data_m4 = drop_trailing_constant_rows(build_bar_data(m4_df))
    filtered_bar_data_mc = drop_trailing_constant_rows(build_bar_data(mc_df))
    filtered_bar_data_cif = drop_trailing_constant_rows(build_bar_data(cif_df))
    filtered_bar_data_gt = drop_trailing_constant_rows(build_bar_data(gt_df))

    plot_bar_solved(filtered_bar_data_aca, "ACAS Xu")
    plot_bar_solved(filtered_bar_data_m4, "MNIST-F")
    plot_bar_solved(filtered_bar_data_mc, "MNIST-C")
    plot_bar_solved(filtered_bar_data_cif, "CIFAR")
    plot_bar_solved(filtered_bar_data_gt, "GTSRB")


# plot_bar_solved_all()
