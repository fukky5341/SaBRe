import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from typing import Any
from util import method_name_map, method_to_status_col, method_to_time_col


def read_csvfolder(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".csv"):
                file_paths.append(os.path.join(root, filename))
    file_paths.sort()
    return file_paths


def get_time_budget(df, net_name):
    if net_name == 'acasxu':
        time_budget = 420  # seconds
    elif net_name == 'mnist4' or net_name == 'mnist-conv':
        time_budget = 600  # seconds
    elif net_name == 'cifar10':
        # for net_1_*, time budget is 1800 seconds, 
        # for net_2_*, time budget is 3600 seconds,
        # for other nets, time budget is 7200 seconds
        time_budget = df['name'].apply(lambda x: 1800 if x.startswith('1_') \
                                       else 3600 if x.startswith('2_') else 7200)
    elif net_name == 'gtsrb':
        # for net_1_*, time budget is 1800 seconds, 
        # for net_2_*, time budget is 3600 seconds,
        # for other nets, time budget is 7200 seconds
        time_budget = df['name'].apply(lambda x: 1800 if x.startswith('1_') \
                                       else 3600 if x.startswith('2_') else 7200)
    else:
        raise ValueError("Unknown net_name")
    return time_budget


def sort_df_either_solved(df, net_name, methods):
    time_budget = get_time_budget(df, net_name)

    solved_mask = np.zeros(len(df), dtype=bool)
    for method in methods:
        solved_mask |= (df[method_to_status_col[method]] != 'UNKNOWN') & (df[method_to_time_col[method]] <= time_budget)
    df_solved = df[solved_mask]
    return df_solved


def sort_df_all_solved(df, net_name, methods):
    time_budget = get_time_budget(df, net_name)

    solved_mask = np.ones(len(df), dtype=bool)
    for method in methods:
        solved_mask &= (df[method_to_status_col[method]] != 'UNKNOWN') & (df[method_to_time_col[method]] <= time_budget)

    df_solved = df[solved_mask]
    return df_solved


def extract_exp_info_from_path_acasxu(file_path):
    """
    e.g., './acasxu/DS_dual/net_1_1_d_3/2.md'
    return: (net_id, d_val, input_idx) = (1, 3, 2)
    """
    parts = file_path.split('/')
    net_id1 = int(parts[3].split('_')[1])
    net_id2 = int(parts[3].split('_')[2])
    d_val = int(parts[3].split('_')[4])
    input_idx = int(parts[4].split('.')[0])
    return net_id1, net_id2, d_val, input_idx


def extract_exp_info_from_path(file_path):
    """
    e.g., './mnist-256x4/DS_dual/d1_e2_0.md'
    return: (d_val, i_val, input_idx) = (1, 2, 0)
    """
    parts = file_path.split('/')
    d_val = int(parts[3].split('_')[0][1:])
    i_val = int(parts[3].split('_')[1][1:])
    input_idx = int(parts[3].split('_')[2].split('.')[0])
    return d_val, i_val, input_idx


def get_subproblems_num(folder_path, df, acasxu=False):
    target_indicies = df.loc[df['base status'] != 'VERIFIED', 'name'].tolist()
    csv_files = read_csvfolder(folder_path)
    subproblems_num = 0
    for file_path in csv_files:
        if acasxu:
            net_id1, net_id2, d_val, input_idx = extract_exp_info_from_path_acasxu(file_path)
            name = f"{net_id1}_{net_id2}_{d_val}_{input_idx}"
        else:
            d_val, i_val, input_idx = extract_exp_info_from_path(file_path)
            name = f"{d_val}_{i_val}_{input_idx}"
        if name not in target_indicies:
            continue
        df_log = pd.read_csv(file_path)
        subproblems_num += df_log['name'].nunique()
    ave_subproblems_num = subproblems_num / len(target_indicies)
    return ave_subproblems_num


def random_sample_base_solved_df(df, n_solved):
    # set random seed
    random.seed(1234)
    random_indices = random.sample(list(df.index), n_solved)
    # add base status column at third from left, if index in random_indices, then VERIFIED else UNKNOWN
    new_df = df.copy()
    new_df.insert(2, 'base status', new_df.index.to_series().apply(lambda x: 'VERIFIED' if x in random_indices else 'UNKNOWN'))
    return new_df

def random_sample_base_solved_df_cifar(df, n_solved):
    # add base status column at third from left, if index in random_indices, then VERIFIED else UNKNOWN
    new_df = df.copy()
    new_df_1_2 = new_df[new_df['name'].str.startswith(('1_', '2_'))]
    new_df_other = new_df[~new_df['name'].str.startswith(('1_', '2_'))]
    # set random seed
    random.seed(1234)
    random_indices = random.sample(list(new_df_1_2.index), n_solved)
    new_df_1_2.insert(2, 'base status', new_df_1_2.index.to_series().apply(lambda x: 'VERIFIED' if x in random_indices else 'UNKNOWN'))
    new_df_other.insert(2, 'base status', 'UNKNOWN')
    new_df = pd.concat([new_df_1_2, new_df_other], ignore_index=True)
    return new_df


def analyze_with_base(method, acasxu_df, mnist4_df, mnist_conv_df, cifar10_df, gtsrb_df, subproblems_num_dict=None):
    dataset_latex_map = {
        "acasxu": "ACAS Xu",
        "mnist4": "MNIST-F",
        "mnist-conv": "MNIST-C",
        "cifar10": "CIFAR",
        "gtsrb": "GTSRB",
    }
    solved_label = r"$s^{\#}$"
    subproblems_label = r"$p^{\#}$"
    delta_time_label = r"$\Delta T$"
    raven_label = "RaVeN"
    tool_label = "SaBRe"

    rows = []
    for net_name, df in [('acasxu', acasxu_df), ('mnist4', mnist4_df), ('mnist-conv', mnist_conv_df), ('cifar10', cifar10_df), ('gtsrb', gtsrb_df)]:
        time_budget = get_time_budget(df, net_name)
        # -- base solved --
        base_solved = df[(df['base status'] == 'VERIFIED') & (df['base time'] <= time_budget)]
        # -- method solved --
        # method_solved = df[(df[method_to_status_col[method]] == 'VERIFIED') & (df[method_to_time_col[method]] <= time_budget)]
        method_solved = df[(df['base status'] != 'VERIFIED') & (df[method_to_status_col[method]] != 'UNKNOWN') & (df[method_to_time_col[method]] <= time_budget)]
        # -- average time ratio (base solved) --
        base_solved_time_ratio = (base_solved['base time'] / time_budget)*100
        # -- average time ratio (method solved) --
        method_solved_time_ratio = (method_solved[method_to_time_col[method]] / time_budget)*100
        # -- average time (base solved) --
        base_solved_time = base_solved['base time']
        # -- average time (method solved) --
        method_solved_time = method_solved[method_to_time_col[method]]
        row = {
            "dataset": net_name,
            "method": method_name_map[method],
            "base_solved": base_solved.shape[0],
            "method_solved": method_solved.shape[0],
            "base_subproblem": 1,
            "subproblems": subproblems_num_dict.get(net_name, pd.NA) if subproblems_num_dict is not None else pd.NA,
            "base_time_ratio": base_solved_time_ratio.mean(),
            "method_time_ratio": method_solved_time_ratio.mean(),
        }
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df = result_df[[
        "dataset",
        "method",
        "base_solved",
        "method_solved",
        "base_subproblem",
        "subproblems",
        "base_time_ratio",
        "method_time_ratio",
    ]]
    print("\\begin{tabular}{lccccccccc}")
    print("\\toprule")
    print(
        "\\multirow{2}{*}{Dataset} & "
        f"\\multicolumn{{2}}{{c}}{{{solved_label}}} & & "
        f"\\multicolumn{{2}}{{c}}{{{subproblems_label}}} & & "
        f"\\multicolumn{{2}}{{c}}{{{delta_time_label} (\\%)}} \\\\"
    )
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){5-6} \\cmidrule(lr){8-9}")
    print(f"& {raven_label} & {tool_label} & & {raven_label} & {tool_label} & & {raven_label} & {tool_label} \\\\")
    print("\\midrule")
    for _, row in result_df.iterrows():
        subproblem_text = "-" if pd.isna(row["subproblems"]) else f"{float(row['subproblems']):.1f}"
        print(
            f"{dataset_latex_map.get(row['dataset'], row['dataset'])} & {int(row['base_solved'])} & {int(row['method_solved'])} && "
            f"{int(row['base_subproblem'])} & {subproblem_text} && "
            f"{float(row['base_time_ratio']):.2f} & {float(row['method_time_ratio']):.2f} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")

    display_columns = pd.MultiIndex.from_tuples(
        [
            (solved_label, raven_label),
            (solved_label, tool_label),
            (subproblems_label, raven_label),
            (subproblems_label, tool_label),
            (delta_time_label, raven_label),
            (delta_time_label, tool_label),
        ]
    )
    display_rows: list[list[Any]] = []
    display_index: list[str] = []
    for _, row in result_df.iterrows():
        display_index.append(dataset_latex_map.get(row["dataset"], row["dataset"]))
        subproblem_text = "-" if pd.isna(row["subproblems"]) else f"{float(row['subproblems']):.1f}"
        display_rows.append(
            [
                str(int(row["base_solved"])),
                str(int(row["method_solved"])),
                "1",
                subproblem_text,
                f"{float(row['base_time_ratio']):.2f}",
                f"{float(row['method_time_ratio']):.2f}",
            ]
        )

    display_df = pd.DataFrame(display_rows, index=display_index, columns=display_columns)
    display_df.index.name = "Dataset"
    return display_df


def analyze_result(df, net_name, methods):
    if 'base status' in df.columns:
        df = df[df['base status'] != 'VERIFIED']
    time_budget = get_time_budget(df, net_name)
    res = {}
    all_solved = sort_df_all_solved(df, net_name, methods)
    either_solved = sort_df_either_solved(df, net_name, methods)
    for method in methods:
        solved = df[(df[method_to_status_col[method]] != "UNKNOWN") & (df[method_to_time_col[method]] <= time_budget)]
        time_ratio = solved[method_to_time_col[method]] / time_budget
        all_time = all_solved[method_to_time_col[method]].clip(upper=time_budget)
        all_time.loc[all_solved[method_to_status_col[method]] == "UNKNOWN"] = time_budget
        all_time_ratio = all_time / time_budget
        either_time = either_solved[method_to_time_col[method]].clip(upper=time_budget)
        either_time.loc[either_solved[method_to_status_col[method]] == "UNKNOWN"] = time_budget
        either_time_ratio = either_time / time_budget

        res[method] = {
            'num_solved': solved.shape[0],
            'mean_time_ratio_solved': time_ratio.mean(),
            'mean_time_ratio_all': all_time_ratio.mean(),
            'mean_time_ratio_either': either_time_ratio.mean()
        }

    for method in methods:
        print(f"Method: {method}")
        print(f"  #Solved: {res[method]['num_solved']}")
        print(f"  Mean time ratio (solved): {res[method]['mean_time_ratio_solved']}")
        print(f"  Mean time ratio (all): {res[method]['mean_time_ratio_all']}")
        print(f"  Mean time ratio (either): {res[method]['mean_time_ratio_either']}")


def analyze_ds_ns_result(df, net_name):
    time_budget = get_time_budget(df, net_name)

    solved_ds = df[(df["DS status"] != "UNKNOWN") & (df["DS time"] <= time_budget)]
    solved_ns = df[(df["NS status"] != "UNKNOWN") & (df["NS time"] <= time_budget)]
    solved_nsind = df[(df["NSInd status"] != "UNKNOWN") & (df["NSInd time"] <= time_budget)]
    solved_common = df[
        ((df["DS status"] != "UNKNOWN") & (df["DS time"] <= time_budget)) &
        ((df["NS status"] != "UNKNOWN") & (df["NS time"] <= time_budget)) &
        ((df["NSInd status"] != "UNKNOWN") & (df["NSInd time"] <= time_budget))
    ]
    solved_one_can = df[
        ((df["DS status"] != "UNKNOWN") & (df["DS time"] <= time_budget)) |
        ((df["NS status"] != "UNKNOWN") & (df["NS time"] <= time_budget)) |
        ((df["NSInd status"] != "UNKNOWN") & (df["NSInd time"] <= time_budget))
    ]
    # time ratio
    time_ratio_ds = solved_ds['DS time'] / time_budget
    time_ratio_ns = solved_ns['NS time'] / time_budget
    time_ratio_nsind = solved_nsind['NSInd time'] / time_budget
    time_ratio_common_ds = solved_common['DS time'] / time_budget
    time_ratio_common_ns = solved_common['NS time'] / time_budget
    time_ratio_common_nsind = solved_common['NSInd time'] / time_budget
    time_ratio_either_ds = solved_one_can['DS time'] / time_budget
    time_ratio_either_ns = solved_one_can['NS time'] / time_budget
    time_ratio_either_nsind = solved_one_can['NSInd time'] / time_budget

    print("Analysis of DS/NS/NSInd Result")
    print(f"Total instances: {len(df)}")
    print(f"#Solved")
    print(f"DS: {solved_ds.shape[0]}, NS: {solved_ns.shape[0]}, NSInd: {solved_nsind.shape[0]}")
    print(f"common: {solved_common.shape[0]}, either: {solved_one_can.shape[0]}")
    print(f"Time ratio (= mean time / time budget)")
    print(f"DS: {time_ratio_ds.mean()}, NS: {time_ratio_ns.mean()}, NSInd: {time_ratio_nsind.mean()}")
    print(f"Time (common)")
    print(f"DS: {time_ratio_common_ds.mean()}, NS: {time_ratio_common_ns.mean()}, NSInd: {time_ratio_common_nsind.mean()}")
    print(f"Time (either)")
    print(f"DS: {time_ratio_either_ds.mean()}, NS: {time_ratio_either_ns.mean()}, NSInd: {time_ratio_either_nsind.mean()}")


def get_solvedNumRatio_timeAveRatio(df, net_name, method):
    time_budget = get_time_budget(df, net_name)

    if method == 'NS':
        status_col = 'NS status'
        time_col = 'NS time'
    elif method == 'NSInd':
        status_col = 'NSInd status'
        time_col = 'NSInd time'
    elif method == 'DS_dual_Z':
        status_col = 'DSZ status'
        time_col = 'DSZ time'
    elif method == 'DS_random_Z':
        status_col = 'RndZ status'
        time_col = 'RndZ time'
    elif method == 'Random':
        status_col = 'Random status'
        time_col = 'Random time'
    elif method == 'Dual':
        status_col = 'Dual status'
        time_col = 'Dual time'
    else:
        raise ValueError("Unknown method")

    solved_df = df[(df[status_col] != "UNKNOWN") & (df[time_col] <= time_budget)]
    solved_num = solved_df.shape[0]
    solved_ratio = solved_num / len(df)
    time_df = df[time_col].clip(upper=time_budget)
    time_df.loc[df[status_col] == "UNKNOWN"] = time_budget
    time_ratio_df = (time_df / time_budget)*100
    time_ave_ratio = time_ratio_df.mean()

    return solved_num, solved_ratio, time_ave_ratio


def get_table_Nsolved_timeRatio(acasxu_df, mnist4_df, mnist_conv_df, cifar_df, gtsrb_df, methods):
    dataset_specs = [
        ("acasxu", acasxu_df, "./acasxu", "ACAS Xu"),
        ("mnist4", mnist4_df, "./mnist-256x4", "MNIST-F"),
        ("mnist-conv", mnist_conv_df, "./mnist-conv", "MNIST-C"),
        ("cifar10", cifar_df, "./cifar10", "CIFAR"),
        ("gtsrb", gtsrb_df, "./gtsrb", "GTSRB"),
    ]

    dataset_latex_map = {
        "acasxu": "ACAS Xu",
        "mnist4": "MNIST-F",
        "mnist-conv": "MNIST-C",
        "cifar10": "CIFAR",
        "gtsrb": "GTSRB",
    }
    metric_latex_map = {
        "solved": r"$s^{\#}$",
        "subproblems": r"$p^{\#}$",
        "deltaTime": r"$\Delta T$",
    }
    method_latex_map = {
        "NSInd": "ClasIS",
        "NS": "DualIS",
        "DS_dual_Z": "SaBRe",
        "DS_random_Z": "RandRS",
    }

    method_path_map = {
        "NS": ("NS_dual", "IS_Dual"),
        "NSInd": ("NS_dual_ind", "IS_Dual_Ind"),
        "DS_dual_Z": ("DS_dual_Z", "RS_dual_Z"),
        "DS_random_Z": ("DS_random_Z", "RS_random_Z"),
    }

    filtered_dfs = {
        name: df[df["base status"] != "VERIFIED"].copy()
        for name, df, _, _ in dataset_specs
    }

    rows: list[dict[str, Any]] = []
    for method in methods:
        row: dict[str, Any] = {"method": method_name_map[method]}
        tmp_method, gtsrb_method = method_path_map.get(method, (method, method))

        for dataset_name, df, folder_prefix, _display_name in dataset_specs:
            threshold_method = gtsrb_method if dataset_name == "gtsrb" else tmp_method
            either_df = sort_df_either_solved(filtered_dfs[dataset_name], dataset_name, methods)
            solved_num, _, _ = get_solvedNumRatio_timeAveRatio(filtered_dfs[dataset_name], dataset_name, method)

            if dataset_name == "acasxu":
                subproblems_num = get_subproblems_num(f"{folder_prefix}/{threshold_method}_threshold", either_df, acasxu=True)
            else:
                subproblems_num = get_subproblems_num(f"{folder_prefix}/{threshold_method}_threshold", either_df)

            _, _, time_ave_ratio = get_solvedNumRatio_timeAveRatio(either_df, dataset_name, method)
            row[f"{dataset_name}_solved"] = solved_num
            row[f"{dataset_name}_subproblems"] = subproblems_num
            row[f"{dataset_name}_deltaTime"] = time_ave_ratio

        rows.append(row)

    result_df = pd.DataFrame(rows)
    ordered_columns = ["method"]
    for dataset_name, _, _, _ in dataset_specs:
        ordered_columns.extend(
            [
                f"{dataset_name}_solved",
                f"{dataset_name}_subproblems",
                f"{dataset_name}_deltaTime",
            ]
        )
    result_df = result_df[ordered_columns]

    print("\\begin{tabular}{lccccccccccccccc}")
    print("\\toprule")
    print(
        "\\multirow{2}{*}{Method} & "
        f"\\multicolumn{{3}}{{c}}{{{dataset_latex_map['acasxu']}}} & "
        f"\\multicolumn{{3}}{{c}}{{{dataset_latex_map['mnist4']}}} & "
        f"\\multicolumn{{3}}{{c}}{{{dataset_latex_map['mnist-conv']}}} & "
        f"\\multicolumn{{3}}{{c}}{{{dataset_latex_map['cifar10']}}} & "
        f"\\multicolumn{{3}}{{c}}{{{dataset_latex_map['gtsrb']}}} \\\\"
    )
    print("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13} \\cmidrule(lr){14-16}")
    print(
        f"& {metric_latex_map['solved']} & {metric_latex_map['subproblems']} & {metric_latex_map['deltaTime']} "
        f"& {metric_latex_map['solved']} & {metric_latex_map['subproblems']} & {metric_latex_map['deltaTime']} "
        f"& {metric_latex_map['solved']} & {metric_latex_map['subproblems']} & {metric_latex_map['deltaTime']} "
        f"& {metric_latex_map['solved']} & {metric_latex_map['subproblems']} & {metric_latex_map['deltaTime']} "
        f"& {metric_latex_map['solved']} & {metric_latex_map['subproblems']} & {metric_latex_map['deltaTime']} \\\\"
    )
    print("\\midrule")

    for method, (_, row) in zip(methods, result_df.iterrows()):
        cells: list[str] = []
        for dataset_name, _, _, _ in dataset_specs:
            solved_val = row[f"{dataset_name}_solved"]
            subproblems_val = row[f"{dataset_name}_subproblems"]
            delta_time_val = row[f"{dataset_name}_deltaTime"]
            solved_text = "-" if pd.isna(solved_val) else str(int(round(float(solved_val))))
            subproblems_text = "-" if pd.isna(subproblems_val) else f"{float(subproblems_val):.1f}"
            delta_time_text = "-" if pd.isna(delta_time_val) else f"{float(delta_time_val):.2f}"
            cells.extend([solved_text, subproblems_text, delta_time_text])

        method_label = method_latex_map.get(method, row["method"])
        print(f"{method_label} & " + " & ".join(cells) + r" \\")

    print("\\bottomrule")
    print("\\end{tabular}")

    display_columns = pd.MultiIndex.from_tuples(
        [
            (dataset_latex_map[dataset_name], metric_latex_map[metric])
            for dataset_name, _, _, _ in dataset_specs
            for metric in ("solved", "subproblems", "deltaTime")
        ]
    )

    display_rows: list[list[Any]] = []
    display_index: list[str] = []
    for method, (_, row) in zip(methods, result_df.iterrows()):
        method_label = method_latex_map.get(method, row["method"])
        display_index.append(method_label)
        row_cells: list[Any] = []
        for dataset_name, _, _, _ in dataset_specs:
            solved_val = row[f"{dataset_name}_solved"]
            subproblems_val = row[f"{dataset_name}_subproblems"]
            delta_time_val = row[f"{dataset_name}_deltaTime"]
            solved_text = "-" if pd.isna(solved_val) else str(int(round(float(solved_val))))
            subproblems_text = "-" if pd.isna(subproblems_val) else f"{float(subproblems_val):.1f}"
            delta_time_text = "-" if pd.isna(delta_time_val) else f"{float(delta_time_val):.2f}"
            row_cells.extend([solved_text, subproblems_text, delta_time_text])
        display_rows.append(row_cells)

    display_df = pd.DataFrame(display_rows, index=display_index, columns=display_columns)
    display_df.index.name = "Method"
    return display_df


def get_table_Nsolved_timeRatio_rand_dual(acasxu_df, mnist4_df, mnist_conv_df, cifar_df, methods):
    acasxu_df = acasxu_df[acasxu_df['base status'] != 'VERIFIED']
    mnist4_df = mnist4_df[mnist4_df['base status'] != 'VERIFIED']
    mnist_conv_df = mnist_conv_df[mnist_conv_df['base status'] != 'VERIFIED']
    cifar_df = cifar_df[cifar_df['base status'] != 'VERIFIED']
    print("all instances")
    temp_res = []
    for method in methods:
        ac_solved_num, ac_solved_ratio, ac_time_ave_ratio = get_solvedNumRatio_timeAveRatio(acasxu_df, 'acasxu', method)
        m4_solved_num, m4_solved_ratio, m4_time_ave_ratio = get_solvedNumRatio_timeAveRatio(mnist4_df, 'mnist4', method)
        mc_solved_num, mc_solved_ratio, mc_time_ave_ratio = get_solvedNumRatio_timeAveRatio(mnist_conv_df, 'mnist-conv', method)
        ci_solved_num, ci_solved_ratio, ci_time_ave_ratio = get_solvedNumRatio_timeAveRatio(cifar_df, 'cifar10', method)
        name = method_name_map[method]
        temp_res.append([ac_solved_num, m4_solved_num, mc_solved_num, ci_solved_num])
        # print(f"    {name} & {solved_ratio:.2f} ({solved_num}) & {time_ave_ratio:.2f} \\\\")
        print(f"    {name} & {ac_solved_num} & {ac_time_ave_ratio:.2f} & {m4_solved_num} & {m4_time_ave_ratio:.2f} & {mc_solved_num} & {mc_time_ave_ratio:.2f} & {ci_solved_num} & {ci_time_ave_ratio:.2f} \\\\")
    print("instances either solved")
    acasxu_either = sort_df_either_solved(acasxu_df, 'acasxu', methods)
    mnist4_either = sort_df_either_solved(mnist4_df, 'mnist4', methods)
    mnistconv_either = sort_df_either_solved(mnist_conv_df, 'mnist-conv', methods)
    cifar10_either = sort_df_either_solved(cifar_df, 'cifar10', methods)
    for method, solved_nums in zip(methods, temp_res):
        _, _, ac_time_ave_ratio = get_solvedNumRatio_timeAveRatio(acasxu_either, 'acasxu', method)
        _, _, m4_time_ave_ratio = get_solvedNumRatio_timeAveRatio(mnist4_either, 'mnist4', method)
        _, _, mc_time_ave_ratio = get_solvedNumRatio_timeAveRatio(mnistconv_either, 'mnist-conv', method)
        _, _, ci_time_ave_ratio = get_solvedNumRatio_timeAveRatio(cifar10_either, 'cifar10', method)
        name = method_name_map[method]
        print(f"    {name} & {solved_nums[0]} & {ac_time_ave_ratio:.2f} & {solved_nums[1]} & {m4_time_ave_ratio:.2f} & {solved_nums[2]} & {mc_time_ave_ratio:.2f} & {solved_nums[3]} & {ci_time_ave_ratio:.2f} \\\\")
    print("instances all solved")
    acasxu_all = sort_df_all_solved(acasxu_df, 'acasxu', methods)
    mnist4_all = sort_df_all_solved(mnist4_df, 'mnist4', methods)
    mnistconv_all = sort_df_all_solved(mnist_conv_df, 'mnist-conv', methods)
    cifar10_all = sort_df_all_solved(cifar_df, 'cifar10', methods)
    for method, solved_nums in zip(methods, temp_res):
        _, _, ac_time_ave_ratio = get_solvedNumRatio_timeAveRatio(acasxu_all, 'acasxu', method)
        _, _, m4_time_ave_ratio = get_solvedNumRatio_timeAveRatio(mnist4_all, 'mnist4', method)
        _, _, mc_time_ave_ratio = get_solvedNumRatio_timeAveRatio(mnistconv_all, 'mnist-conv', method)
        _, _, ci_time_ave_ratio = get_solvedNumRatio_timeAveRatio(cifar10_all, 'cifar10', method)
        name = method_name_map[method]
        print(f"    {name} & {solved_nums[0]} & {ac_time_ave_ratio:.2f} & {solved_nums[1]} & {m4_time_ave_ratio:.2f} & {solved_nums[2]} & {mc_time_ave_ratio:.2f} & {solved_nums[3]} & {ci_time_ave_ratio:.2f} \\\\")
    return

def get_solved_num_on_d(mnist4_df, mnist_conv_df, cifar_df):
    df_list = [mnist4_df, mnist_conv_df, cifar_df]
    dataset_names = ['mnist4', 'mnist-conv', 'cifar10']
    methods = ['NSInd', 'NS', 'DSZ']

    for df, dataset_name in zip(df_list, dataset_names):
        print(f"{dataset_name}:")
        df_1 = df[(df['name'].str.startswith("1_")) & (df['base status'] == 'UNKNOWN')]
        df_2 = df[(df['name'].str.startswith("2_")) & (df['base status'] == 'UNKNOWN')]
        df_3 = df[(df['name'].str.startswith("3_")) & (df['base status'] == 'UNKNOWN')]
        for i, df_temp in enumerate([df_1, df_2, df_3]):
            print(f"  d={i+1}:")
            if df_temp.empty:
                print("    No instances")
                continue
            time_budget = get_time_budget(df_temp, dataset_name)
            for method in methods:
                solved_num = len(df_temp[(df_temp[f'{method} status'] != 'UNKNOWN') & (df_temp[f'{method} time'] <= time_budget)])
                print(f"  {method}: {solved_num}")
        print("")

def get_solved_num_and_timeratio_on_d(df, dataset_name):
    methods = ['NSInd', 'NS', 'DSZ']
    latex_map = {
        "NSInd": "ClasIS",
        "NS": "DualIS",
        "DSZ": "SaBRe"
    }
    d_values = [1, 2, 3]

    rows: list[dict[str, Any]] = []

    for d in d_values:
        df_temp = df[
            (df['name'].str.startswith(f"{d}_", na=False)) &
            (df['base status'] == 'UNKNOWN')
        ]

        if df_temp.empty:
            for m in methods:
                row = next((r for r in rows if r["method"] == latex_map[m]), None)
                if row is None:
                    row = {"method": latex_map[m]}
                    rows.append(row)
                row[f"d{d}_solved"] = 0
                row[f"d{d}_subproblems"] = pd.NA
                row[f"d{d}_deltaTime"] = pd.NA
            continue

        time_budget = get_time_budget(df_temp, dataset_name)
        filtered_df = df_temp[
            ((df_temp['NSInd status'] != 'UNKNOWN') & (df_temp['NSInd time'] <= time_budget)) |
            ((df_temp['NS status'] != 'UNKNOWN') & (df_temp['NS time'] <= time_budget)) |
            ((df_temp['DSZ status'] != 'UNKNOWN') & (df_temp['DSZ time'] <= time_budget))
        ]

        for m in methods:
            row = next((r for r in rows if r["method"] == latex_map[m]), None)
            if row is None:
                row = {"method": latex_map[m]}
                rows.append(row)

            if m == 'NS':
                tmp_method = 'NS_dual'
            elif m == 'NSInd':
                tmp_method = 'NS_dual_ind'
            elif m == 'DSZ':
                tmp_method = 'DS_dual_Z'
            else:
                tmp_method = m
            
            if dataset_name == 'acasxu':
                num_subproblems = get_subproblems_num(f"./{dataset_name}/{tmp_method}_threshold", filtered_df, acasxu=True)
            else:
                num_subproblems = get_subproblems_num(f"./{dataset_name}/{tmp_method}_threshold", filtered_df)

            solved_df = df_temp[
                (df_temp[f'{m} status'] != 'UNKNOWN') &
                (df_temp[f'{m} time'] <= time_budget)
            ]
            solved = len(solved_df)

            if solved > 0:
                ratio = (filtered_df[f'{m} time'] / time_budget).mean() * 100
                row[f"d{d}_solved"] = solved
                row[f"d{d}_subproblems"] = float(f"{num_subproblems:.1f}")
                row[f"d{d}_deltaTime"] = float(f"{ratio:.2f}")
            else:
                row[f"d{d}_solved"] = 0
                row[f"d{d}_subproblems"] = pd.NA
                row[f"d{d}_deltaTime"] = pd.NA

    if not rows:
        return pd.DataFrame(columns=["method"])

    result_df = pd.DataFrame(rows)
    result_df = result_df[
        [
            "method",
            "d1_solved",
            "d1_subproblems",
            "d1_deltaTime",
            "d2_solved",
            "d2_subproblems",
            "d2_deltaTime",
            "d3_solved",
            "d3_subproblems",
            "d3_deltaTime",
        ]
    ]

    print("\\begin{tabular}{lccccccccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{Method} & \\multicolumn{2}{c}{$d=1$} & \\multicolumn{2}{c}{$d=2$} & \\multicolumn{2}{c}{$d=3$} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
    print("& \\solved & \\subproblems & \\deltaTime & \\solved & \\subproblems & \\deltaTime & \\solved & \\subproblems & \\deltaTime \\\\")
    print("\\midrule")

    for _, row in result_df.iterrows():
        row_cells: list[str] = []
        for d in d_values:
            solved_val = row[f"d{d}_solved"]
            subproblems_val = row[f"d{d}_subproblems"]
            delta_time_val = row[f"d{d}_deltaTime"]
            solved_text = "-" if pd.isna(solved_val) else str(int(round(float(solved_val))))
            subproblems_text = "-" if pd.isna(subproblems_val) else f"{float(subproblems_val):.1f}"
            delta_time_text = "-" if pd.isna(delta_time_val) else f"{float(delta_time_val):.2f}"
            row_cells.extend([solved_text, subproblems_text, delta_time_text])

        print(f"{row['method']} & " + " & ".join(row_cells) + r" \\")

    print("\\bottomrule")
    print("\\end{tabular}")
    return result_df
