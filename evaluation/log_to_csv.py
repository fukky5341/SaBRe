import re
import pandas as pd
import os


method_to_time_col = {
    'NS': 'NS time',
    'NSInd': 'NSInd time',
    'DS_dual_Z': 'DSZ time',
    'DS_random_Z': 'RndZ time',
    'RS_dual_Z': 'DSZ time',
    'RS_random_Z': 'RndZ time',
    'IS_dual': 'NS time',
    'IS_dual_ind': 'NSInd time',
    'IS': 'NS time',
    'ISInd': 'NSInd time'
}

method_to_status_col = {
    'NS': 'NS status',
    'NSInd': 'NSInd status',
    'DS_dual_Z': 'DSZ status',
    'DS_random_Z': 'RndZ status',
    'RS_dual_Z': 'DSZ status',
    'RS_random_Z': 'RndZ status',
    'IS_dual': 'NS status',
    'IS_dual_ind': 'NSInd status',
    'IS': 'NS status',
    'ISInd': 'NSInd status'
}


def read_folder(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".md"):
                file_paths.append(os.path.join(root, filename))
    file_paths.sort()
    return file_paths


def extract_exp_info_from_path_acasxu(file_path):
    """
    e.g., '../nnRelationalVerify/experiment_result/acasxu/DS_dual/net_1_1_d_3/2/log.md'
    return: (net_id, d_val, input_idx) = (1, 3, 2)
    """
    parts = file_path.split('/')
    net_id1 = int(parts[5].split('_')[1])
    net_id2 = int(parts[5].split('_')[2])
    d_val = int(parts[5].split('_')[4])
    input_idx = int(parts[6])
    return net_id1, net_id2, d_val, input_idx


def extract_exp_info_from_path(file_path):
    """
    e.g., '../nnRelationalVerify/result/mnist-256x4/DS_dual_Z_threshold/d1_e2/0/log.md'
    return: (d_val, i_val, input_idx) = (1, 2, 0)
    """
    # print(f"file_path: {file_path}")
    parts = file_path.split('/')
    d_val = int(parts[5].split('_')[0][1:])
    i_val = int(parts[5].split('_')[1][1:])
    input_idx = int(parts[6])
    return d_val, i_val, input_idx


def extract_exp_info_from_csvpath(file_path):
    """
    e.g., './mnist-256x4/DS_dual/abcd/d1_e3_3.csv'
    return: (d_val, i_val, input_idx) = (1, 3, 3)
    """
    parts = file_path.split('/')
    d_val = int(parts[4].split('_')[0][1:])
    i_val = int(parts[4].split('_')[1][1:])
    input_idx = int(parts[4].split('_')[2].split('.')[0])
    return d_val, i_val, input_idx


"""
log to csv for each method
"""


def extract_blocks(file_path):
    with open(file_path, 'r') as f:
        text = f.read()

    # Define the regex pattern to match the blocks
    # start with "## Summary of splitting at", then end with a single empty line
    pattern = re.compile(
        r'## Summary of splitting.*?\n'
        r'(?=\n)',  # end with a single empty line
        re.DOTALL
    )

    base_pattern = re.compile(
        r'## BASE Result.*?\n'
        r'(?=\n)',  # end with a single empty line
        re.DOTALL
    )

    matches = pattern.findall(text)
    base_block = base_pattern.findall(text)
    return base_block, matches


def extract_base_result_from_block(base_block_text):
    time_status_pattern = re.compile(
        r'execution time:.*?=\s+([\d.]+)\s+seconds.*?\n'
        r'status:\s+Status\.([A-Z]+)',
        re.DOTALL,
    )
    dim_pattern = re.compile(
        r'Output dim:\s+(\d+),\s+lower bound:\s+([-\d.]+),\s+upper bound:\s+([-\d.]+)',
    )

    time_status_match = time_status_pattern.search(base_block_text)
    if not time_status_match:
        return []

    time, status = time_status_match.groups()
    dim_match = dim_pattern.search(base_block_text)
    if dim_match:
        dim, lower, upper = dim_match.groups()
        dim = int(dim)
        lower = float(lower)
        upper = float(upper)
    else:
        dim = float("nan")
        lower = float("nan")
        upper = float("nan")

    return [['BASE', status, 0, 0, float(time), dim, lower, upper]]


def extract_bounds_from_blocks(base_block, exe_blocks):
    # base_block: list
    # base results: [[dim, lower_bound, upper_bound], ...]
    # ds result of each execution (exe result): [[dim, lower_bound, upper_bound], ...]
    # ds results of all executions (ds results): [[exe result], ...]

    base_result = []
    split_result = []

    # Extract base block results
    # name, status, split level, split count, time, output dim, lower bound, upper bound
    if base_block:
        base_result.extend(extract_base_result_from_block(base_block[0]))

    # Extract execution blocks results
    # divide into ABCD, DSM, DSZ
    # name, status, split level, split count, time, output dim, lower bound, upper bound
    if exe_blocks:
        for level, block in enumerate(exe_blocks, start=1):
            name_status_split_time_dim_lb_ub = re.compile(
                r'([A-Za-z0-9_]+),\s+status:\s+Status\.([A-Z]+),\s+split count:\s+(\d+),\s+time:\s+([\d.]+)\s*\n'
                r'Output dim:\s+(\d+),\s+lower bound:\s+(-?\d+(?:\.\d+)?),\s+upper bound:\s+(-?\d+(?:\.\d+)?)')
            temp_result = []
            for temp in name_status_split_time_dim_lb_ub.finditer(block):
                name, status, split, time, dim, lower, upper = temp.groups()
                temp_result.append([name, status, int(level), int(split), float(time), int(dim), float(lower), float(upper)])
            if temp_result:
                split_result.append(temp_result)

    return base_result, split_result


def convert_base_result_to_df(result):
    """
    result: [["BASE", status, 0, 0, time, dim, lower_bound, upper_bound], ...]
    df: pd.DataFrame with columns ["name", "status", "level", "split", "time", "dim", "lower_bound", "upper_bound"]
    """
    df = pd.DataFrame(result, columns=["name", "status", "level", "split", "time", "dim", "lb", "ub"])
    return df


def convert_ds_results_to_dfs(results):
    """
    execution result: [[name, status, time, dim, lower_bound, upper_bound], ...]
    results: [exe_result, ...]
    df: ["name", "status", "level", "split", "time", "dim", "lower_bound", "upper_bound"]
    dfs: vertically merged df
    """
    columns = ["name", "status", "level", "split", "time", "dim", "lb", "ub"]
    if not results:
        return pd.DataFrame(columns=columns)

    dfs = []
    for exe_result in results:
        df = pd.DataFrame(exe_result, columns=columns)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def path_to_result_df(file_path):
    base_block, exe_blocks = extract_blocks(file_path)
    base_result, split_result = extract_bounds_from_blocks(base_block, exe_blocks)
    base_df = convert_base_result_to_df(base_result)
    split_df = convert_ds_results_to_dfs(split_result)
    if split_df.empty:
        return base_df
    merged_df = pd.concat([base_df, split_df], ignore_index=True)
    return merged_df


"""
log to csv of merged results for each dataset
"""

def get_rsis_result(file_path):
    # d_val, i_val, input_idx = extract_exp_info_from_path(file_path)
    # res_name = f"{d_val}_{i_val}_{input_idx}"

    with open(file_path, 'r') as f:
        text = f.read()

    # base result pattern
    # e.g.,
    # ## BASE Result
    # execution time: IAR + RelationalAnalysis = 1.09 + 1.22 = 2.31 seconds
    # status: Status.VERIFIED

    base_pattern = re.compile(
        r'##\s+BASE\s+Result.*?\n(.*?)(?=\n\n|$)',
        re.DOTALL
    )
    
    base_time_status = re.compile(
        r'execution time:.*?=\s+([\d.]+)\s+seconds.*?\n'
        r'status:\s+Status\.([A-Z]+)',
        re.DOTALL,
    )

    base_match = base_pattern.findall(text)
    if base_match:
        res_match = base_time_status.search(base_match[0])
        if res_match:
            base_time, base_status = res_match.groups()
            if base_status != "UNKNOWN":
                return {
                    "base_status": base_status,
                    "base_time": float(base_time),
                    "status": base_status,
                    "dsns_time": float(base_time),
                    "total_time": float(base_time)
                }
    else:
        print(f"No base result found in {file_path}")
        return None

    # result pattern
    # e.g.,
    # ## NS Result
    # status: Status.UNKNOWN
    # execution time: (base) + (ns) = 53.42 + 550.88 = 604.29 seconds
    # or
    # ## DS Result
    # status: Status.VERIFIED
    # execution time: (base) + (ds) = 56.44 + 541.53 = 597.96 seconds

    pattern = re.compile(
        r'##\s+(?:RS|IS|DS|NS)\s+Result.*?\n(.*?)(?=\n\n|$)',
        re.DOTALL
    )

    status_time = re.compile(
        r'status:\s+Status\.([A-Z]+).*?execution\s+time:\s+\(base\)\s+\+\s+\((?:rs|is|ds|ns)\)\s+=\s+([\d.]+)\s+\+\s+([\d.]+)\s+=\s+([\d.]+)\s+seconds',
        re.DOTALL
    )

    match = pattern.findall(text)
    # print(match[0])
    if match:
        # type_ds_ns = match[0][0]  # 'DS' or 'NS'
        res_match = status_time.search(match[0])
        if res_match:
            status, base_time, ds_time, total_time = res_match.groups()
            return {
                "base_status": base_status,
                "base_time": float(base_time),
                "status": status,
                "dsns_time": float(ds_time),
                "total_time": float(total_time)
            }
    return None


def folder_path_to_df_with_base(folder_path, dsns_name, acasxu=False):
    file_paths = read_folder(folder_path)
    file_paths.sort()
    all_dfs = []
    dsns_status = method_to_status_col[dsns_name]
    dsns_time = method_to_time_col[dsns_name]

    for file_path in file_paths:
        res = get_rsis_result(file_path)
        if acasxu:
            net_id1, net_id2, d_val, input_idx = extract_exp_info_from_path_acasxu(file_path)
            res_name = f"{net_id1}_{net_id2}_{d_val}_{input_idx}"
        else:
            d_val, i_val, input_idx = extract_exp_info_from_path(file_path)
            res_name = f"{d_val}_{i_val}_{input_idx}"
        if res:
            data = [{
                "name": res_name,
                "base status": res["base_status"],
                "base time": res["base_time"],
                dsns_status: res["status"],
                dsns_time: res["total_time"]
            }]
            df = pd.DataFrame(data)
            all_dfs.append(df)
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return None


def folder_path_to_df(folder_path, dsns_name, acasxu=False):
    file_paths = read_folder(folder_path)
    file_paths.sort()
    all_dfs = []
    dsns_status = method_to_status_col[dsns_name]
    dsns_time = method_to_time_col[dsns_name]

    for file_path in file_paths:
        res = get_rsis_result(file_path)
        if acasxu:
            net_id1, net_id2, d_val, input_idx = extract_exp_info_from_path_acasxu(file_path)
            res_name = f"{net_id1}_{net_id2}_{d_val}_{input_idx}"
        else:
            d_val, i_val, input_idx = extract_exp_info_from_path(file_path)
            res_name = f"{d_val}_{i_val}_{input_idx}"
        if res:
            data = [{
                "name": res_name,
                dsns_status: res["status"],
                dsns_time: res["total_time"]
            }]
            df = pd.DataFrame(data)
            all_dfs.append(df)
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return None

def log_to_csv():
    # acasxu
    acasxu_folder_path = "../nnRelationalVerify/result/acasxu/"
    methods = ['DS_dual_Z', 'NS', 'NSInd', 'DS_random_Z']
    acasxu_df = None
    for i, method in enumerate(methods):
        if method == 'NS':
            dsns_folder = 'NS_dual_threshold/'
        elif method == 'NSInd':
            dsns_folder = 'NS_dual_ind_threshold/'
        else:
            dsns_folder = f"{method}_threshold/"
        if i == 0:
            df = folder_path_to_df_with_base(f"{acasxu_folder_path}{dsns_folder}", method, acasxu=True)
        else:
            df = folder_path_to_df(f"{acasxu_folder_path}{dsns_folder}", method, acasxu=True)
        if acasxu_df is None:
            acasxu_df = df
        else:
            acasxu_df = acasxu_df.merge(df, on="name")
    dir_path_ac = "./acasxu"
    if not os.path.exists(dir_path_ac):
        os.makedirs(dir_path_ac)
    acasxu_df.to_csv("./acasxu/acasxu_dsns_whole.csv", index=False)

    # mnist4
    mnist4_folder_path = "../nnRelationalVerify/result/mnist-256x4/"
    methods = ['DS_dual_Z', 'NS', 'NSInd', 'DS_random_Z']
    mnist4_df = None
    for i, method in enumerate(methods):
        if method == 'NS':
            dsns_folder = 'NS_dual_threshold/'
        elif method == 'NSInd':
            dsns_folder = 'NS_dual_ind_threshold/'
        else:
            dsns_folder = f"{method}_threshold/"
        if i == 0:
            df = folder_path_to_df_with_base(f"{mnist4_folder_path}{dsns_folder}", method, acasxu=False)
        else:
            df = folder_path_to_df(f"{mnist4_folder_path}{dsns_folder}", method)
        if mnist4_df is None:
            mnist4_df = df
        else:
            mnist4_df = mnist4_df.merge(df, on="name")
    dir_path_m4 = "./mnist-256x4/"
    if not os.path.exists(dir_path_m4):
        os.makedirs(dir_path_m4)
    mnist4_df.to_csv("./mnist-256x4/mnist4_dsns_whole.csv", index=False)

    # mnist conv
    mnistconv_folder_path = "../nnRelationalVerify/result/mnist-conv/"
    methods = ['DS_dual_Z', 'NS', 'NSInd', 'DS_random_Z']
    mnistconv_df = None
    for i, method in enumerate(methods):
        if method == 'NS':
            dsns_folder = 'NS_dual_threshold/'
        elif method == 'NSInd':
            dsns_folder = 'NS_dual_ind_threshold/'
        else:
            dsns_folder = f"{method}_threshold/"
        if i == 0:
            df = folder_path_to_df_with_base(f"{mnistconv_folder_path}{dsns_folder}", method, acasxu=False)
        else:
            df = folder_path_to_df(f"{mnistconv_folder_path}{dsns_folder}", method)
        if mnistconv_df is None:
            mnistconv_df = df
        else:
            mnistconv_df = mnistconv_df.merge(df, on="name")
    dir_path_mc = "./mnist-conv/"
    if not os.path.exists(dir_path_mc):
        os.makedirs(dir_path_mc)
    mnistconv_df.to_csv("./mnist-conv/mnistconv_dsns_whole.csv", index=False)

    # cifar
    cifar_folder_path = "../nnRelationalVerify/result/cifar10/"
    methods = ['DS_dual_Z', 'NS', 'NSInd', 'DS_random_Z']
    cifar_df = None
    for i, method in enumerate(methods):
        if method == 'NS':
            dsns_folder = 'NS_dual_threshold/'
        elif method == 'NSInd':
            dsns_folder = 'NS_dual_ind_threshold/'
        else:
            dsns_folder = f"{method}_threshold/"
        if i == 0:
            df = folder_path_to_df_with_base(f"{cifar_folder_path}{dsns_folder}", method, acasxu=False)
        else:
            df = folder_path_to_df(f"{cifar_folder_path}{dsns_folder}", method)
        if cifar_df is None:
            cifar_df = df
        else:
            cifar_df = cifar_df.merge(df, on="name")
    dir_path_c = "./cifar10/"
    if not os.path.exists(dir_path_c):
        os.makedirs(dir_path_c)
    cifar_df.to_csv("./cifar10/cifar_dsns_whole.csv", index=False)

    # gtsrb
    gtsrb_folder_path = "../nnRelationalVerify/result/gtsrb/"
    methods = ['RS_dual_Z', 'IS', 'ISInd', 'RS_random_Z']
    gtsrb_df = None
    for i, method in enumerate(methods):
        if method == 'IS':
            dsns_folder = 'IS_dual_threshold/'
        elif method == 'ISInd':
            dsns_folder = 'IS_dual_ind_threshold/'
        else:
            dsns_folder = f"{method}_threshold/"
        if i == 0:
            df = folder_path_to_df_with_base(f"{gtsrb_folder_path}{dsns_folder}", method, acasxu=False)
        else:
            df = folder_path_to_df(f"{gtsrb_folder_path}{dsns_folder}", method)
        if gtsrb_df is None:
            gtsrb_df = df
        else:
            gtsrb_df = gtsrb_df.merge(df, on="name")
    dir_path_g = "./gtsrb/"
    if not os.path.exists(dir_path_g):
        os.makedirs(dir_path_g)
    gtsrb_df.to_csv("./gtsrb/gtsrb_dsns_whole.csv", index=False)

    # acasxu
    acasxu_folder = '../nnRelationalVerify/result/acasxu/'
    for dsns_mode in ['DS_dual_Z_threshold', 'DS_random_Z_threshold', 'NS_dual_threshold', 'NS_dual_ind_threshold']:
        acasxu_folder_dsns = f"{acasxu_folder}{dsns_mode}/"
        acasxu_res_files = read_folder(acasxu_folder_dsns)
        save_folder = f"./acasxu/{dsns_mode}/"
        os.makedirs(save_folder, exist_ok=True)
        # print(len(acasxu_res_files))
        for res_file in acasxu_res_files:
            n1, n2, d_vals, input_idxs = extract_exp_info_from_path_acasxu(res_file)
            # print(n1, n2, d_vals, input_idxs)
            split_df = path_to_result_df(res_file)
            save_folder_nnd = f"{save_folder}net_{n1}_{n2}_d_{d_vals}/"
            os.makedirs(save_folder_nnd, exist_ok=True)
            save_path = f"{save_folder_nnd}{input_idxs}.csv"
            split_df.to_csv(save_path, index=False)

    # mnist4
    d_list = [1, 2, 3]
    e_list = [2, 3, 4]
    split_list = ['DS_dual_Z_threshold', 'DS_random_Z_threshold', 'NS_dual_threshold', 'NS_dual_ind_threshold']

    for d_val in d_list:
        for i_val in e_list:
            for split in split_list:
                folder_path = f"../nnRelationalVerify/result/mnist-256x4/{split}/d{d_val}_e{i_val}/"
                if not os.path.exists(folder_path):
                    continue
                logs = read_folder(folder_path)
                for log in logs:
                    d_, i_, input_idx = extract_exp_info_from_path(log)
                    split_df = path_to_result_df(log)
                    save_dir = f"./mnist-256x4/{split}/"
                    os.makedirs(save_dir, exist_ok=True)
                    split_df.to_csv(f"{save_dir}d{d_}_e{i_}_{input_idx}.csv", index=False)

    # mnist conv
    d_list = [1, 2, 3]
    e_list = [2, 3, 4]
    split_list = ['DS_dual_Z_threshold', 'DS_random_Z_threshold', 'NS_dual_threshold', 'NS_dual_ind_threshold']

    for d_val in d_list:
        for i_val in e_list:
            for split in split_list:
                folder_path = f"../nnRelationalVerify/result/mnist-conv/{split}/d{d_val}_e{i_val}/"
                if not os.path.exists(folder_path):
                    continue
                logs = read_folder(folder_path)
                for log in logs:
                    d_, i_, input_idx = extract_exp_info_from_path(log)
                    split_df = path_to_result_df(log)
                    save_dir = f"./mnist-conv/{split}/"
                    os.makedirs(save_dir, exist_ok=True)
                    split_df.to_csv(f"{save_dir}d{d_}_e{i_}_{input_idx}.csv", index=False)

    # cifar
    d_list = [1, 2, 3]
    e_list = [2, 3, 4]
    split_list = ['DS_dual_Z_threshold', 'DS_random_Z_threshold', 'NS_dual_threshold', 'NS_dual_ind_threshold']

    for d_val in d_list:
        for i_val in e_list:
            for split in split_list:
                # print(f"Processing d={d_val}, e={i_val}, split={split}")
                folder_path = f"../nnRelationalVerify/result/cifar10/{split}/d{d_val}_e{i_val}/"
                if not os.path.exists(folder_path):
                    continue
                logs = read_folder(folder_path)
                for log in logs:
                    d_, i_, input_idx = extract_exp_info_from_path(log)
                    split_df = path_to_result_df(log)
                    save_dir = f"./cifar10/{split}/"
                    os.makedirs(save_dir, exist_ok=True)
                    split_df.to_csv(f"{save_dir}d{d_}_e{i_}_{input_idx}.csv", index=False)

    # gtsrb
    d_list = [1, 2, 3]
    e_list = [2, 3, 4]
    split_list = ['RS_random_Z_threshold', 'RS_dual_Z_threshold', 'IS_dual_threshold', 'IS_dual_ind_threshold']

    for d_val in d_list:
        for i_val in e_list:
            for split in split_list:
                # print(f"Processing d={d_val}, e={i_val}, split={split}")
                folder_path = f"../nnRelationalVerify/result/gtsrb/{split}/d{d_val}_e{i_val}/"
                if not os.path.exists(folder_path):
                    continue
                logs = read_folder(folder_path)
                for log in logs:
                    d_, i_, input_idx = extract_exp_info_from_path(log)
                    split_df = path_to_result_df(log)
                    save_dir = f"./gtsrb/{split}/"
                    os.makedirs(save_dir, exist_ok=True)
                    split_df.to_csv(f"{save_dir}d{d_}_e{i_}_{input_idx}.csv", index=False)