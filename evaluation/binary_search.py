import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter

SABRE_DISPLAY_NAME = r'S\textsc{a}BR\textsc{e}' if plt.rcParams.get('text.usetex', False) else 'SᴀBRᴇ'


def parse_binary_search_results(log_text: str):
    # Capture each "## ... Result" block until the next "##" header or end of text
    block_pattern = re.compile(
        r"^##\s+(Binary Search(?: with [^\n]+)? Result)\s*\n(.*?)(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    results = []

    for m in block_pattern.finditer(log_text):
        raw_block_name = m.group(1).strip()
        block_body = m.group(2)

        # status
        status_match = re.search(
            r"^(?:BS Status|status):\s*(.+)$",
            block_body,
            re.MULTILINE,
        )
        status = status_match.group(1).strip() if status_match else None
        if status == "None":
            status = None

        # epsilon
        eps_match = re.search(
            r"^Maximum delta epsilon:\s*(.+)$",
            block_body,
            re.MULTILINE,
        )
        epsilon = eps_match.group(1).strip() if eps_match else None
        if epsilon == "None":
            epsilon = None
        else:
            try:
                epsilon = float(epsilon)
            except ValueError:
                pass

        # time
        time_match = re.search(
            r"^(?:Binary search time|execution time):\s*([\d.]+)\s*seconds$",
            block_body,
            re.MULTILINE,
        )
        time_seconds = float(time_match.group(1)) if time_match else None

        # parsed block type
        if "with RS_dual_Z" in raw_block_name:
            result_type = "RS_dual_Z"
        elif "with RS_random_Z" in raw_block_name:
            result_type = "RS_random_Z"
        elif "with IS_dual_ind" in raw_block_name:
            result_type = "IS_dual_ind"
        elif "with IS_dual" in raw_block_name:
            result_type = "IS_dual"
        else:
            result_type = "BASE"

        results.append({
            "result_type": result_type,
            "status": status,
            "epsilon": epsilon,
            "time": time_seconds,
        })

    return results



def extract_binary_search_results_from_log(
    log_dir: str,
    is_acasxu: bool = False,
    set_id: int = None
):
    method_dirs = ["RS_dual_Z", "RS_random_Z", "IS_dual", "IS_dual_ind"]
    all_results = []

    for method_dir in method_dirs:
        log_dir_method = Path(log_dir) / method_dir

        if is_acasxu:
            log_files = sorted(
                log_dir_method.glob("*/*/log.md"),
                key=lambda p: (p.parent.parent.name, int(p.parent.name)),
            )
        else:
            log_files = sorted(
                log_dir_method.glob("*/log.md"),
                key=lambda p: int(p.parent.name),
            )

        for log_path in log_files:

            if is_acasxu:
                set_name = log_path.parent.parent.name   # net_1_1_d_1
                experiment_id = int(log_path.parent.name)
            else:
                set_name = f"set_{set_id}"
                experiment_id = int(log_path.parent.name)

            with open(log_path, "r", encoding="utf-8") as f:
                text = f.read()

            results = parse_binary_search_results(text)

            for r in results:
                row = r.copy()
                row["experiment_id"] = experiment_id
                row["source_method_dir"] = method_dir
                row["set_name"] = set_name
                all_results.append(row)

    df = pd.DataFrame(all_results)

    if df.empty:
        return df

    # ---- grouping keys ----
    group_keys = ["set_name", "experiment_id", "result_type"]
    sort_keys = ["set_name", "experiment_id", "source_method_dir"]

    # ---- BASE handling ----
    df_base = (
        df[df["result_type"] == "BASE"]
        .sort_values(sort_keys)
        .drop_duplicates(subset=group_keys, keep="first")
    )

    df_methods = df[df["result_type"] != "BASE"]

    df_clean = pd.concat([df_base, df_methods], ignore_index=True)

    df_clean = (
        df_clean
        .sort_values(group_keys)
        .groupby(group_keys, as_index=False)
        .last()
    )

    # ---- wide format ----
    index_cols = ["set_name", "experiment_id"]

    df_wide = (
        df_clean
        .set_index(index_cols + ["result_type"])[["status", "epsilon", "time"]]
        .unstack("result_type")
    )

    df_wide.columns = [
        f"{result_type}_{metric}"
        for metric, result_type in df_wide.columns
    ]

    df_wide = df_wide.reset_index()

    # ---- reorder ----
    desired_order = ["BASE", "RS_dual_Z", "RS_random_Z", "IS_dual_ind", "IS_dual"]
    metrics = ["status", "epsilon", "time"]

    ordered_cols = index_cols.copy()
    for method in desired_order:
        for metric in metrics:
            col = f"{method}_{metric}"
            if col in df_wide.columns:
                ordered_cols.append(col)

    return df_wide[ordered_cols]

methods = ["RS_dual_Z", "RS_random_Z", "IS_dual_ind", "IS_dual"]

def extract_binary_search_results_acasxu_all(root_dir: str):

    root = Path(root_dir)

    acas_dirs = sorted([
        d for d in root.glob("acasxu*")
        if d.is_dir()
    ])

    all_dfs = []

    for acas_dir in acas_dirs:
        df = extract_binary_search_results_from_log(
            log_dir=str(acas_dir),
            is_acasxu=True
        )

        if df.empty:
            continue

        # distinguish different runs
        df["group"] = acas_dir.name

        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)

def extract_set_id_from_dir(dir_name: str, dataset_name: str) -> int:
    prefix = f"{dataset_name}_"
    if not dir_name.startswith(prefix):
        raise ValueError(f"Unexpected directory name: {dir_name}")
    return int(dir_name[len(prefix):])


def extract_binary_search_results_all(root_dir: str, dataset_name: str):

    root = Path(root_dir)

    all_dirs = sorted([
        d for d in root.glob(f"{dataset_name}_*")
        if d.is_dir()
    ])

    all_dfs = []

    for dir in all_dirs:

        set_id = extract_set_id_from_dir(dir.name, dataset_name)

        df = extract_binary_search_results_from_log(
            log_dir=str(dir),
            is_acasxu=False,
            set_id=set_id
        )

        if df.empty:
            continue

        # keep both numeric + readable id
        df["set_id"] = set_id
        df["group"] = dir.name

        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)

def either_solved_df(df: pd.DataFrame):
    status_cols = [f"{method}_status" for method in methods]

    solved_mask = df[status_cols].isin(["VERIFIED", "Status.VERIFIED"]).any(axis=1)

    return df[solved_mask].copy()

def all_solved_df(df: pd.DataFrame):
    status_cols = [f"{method}_status" for method in methods]

    solved_mask = df[status_cols].isin(["VERIFIED", "Status.VERIFIED"]).all(axis=1)

    return df[solved_mask].copy()

def drop_time_col(df):
    time_cols = [f"{method}_time" for method in methods]
    time_cols += "BASE_time"
    existing_time_cols = [col for col in time_cols if col in df.columns]
    return df.drop(columns=existing_time_cols)

def count_solved(df):
    base_epsilon_col = f"BASE_epsilon"
    for method in methods:
        status_col = f"{method}_status"
        epsilon_col = f"{method}_epsilon"
        if status_col in df.columns:
            # except row where BASE_epsilon = method_epsilon (i.e. no improvement)
            if epsilon_col in df.columns and base_epsilon_col in df.columns:
                solved_count = df[
                    (df[status_col].isin(['VERIFIED', 'Status.VERIFIED'])) &
                    (df[epsilon_col] != df[base_epsilon_col])
                ].shape[0]
                print(f"{method}: {solved_count} solved with improvement")

def plot_epsilon(df, is_acasxu=False, dataset_display_name=""):
    denom = 256 if not is_acasxu else 1
    epsilon_cols = [
        "BASE_epsilon",
        "RS_dual_Z_epsilon",
        "RS_random_Z_epsilon",
        "IS_dual_ind_epsilon",
        "IS_dual_epsilon",
    ]
    label_mapping = {
        "BASE_epsilon": "RaVeN",
        "RS_dual_Z_epsilon": "SaBRe",
        "RS_random_Z_epsilon": "RandRS",
        "IS_dual_epsilon": "DualIS",
        "IS_dual_ind_epsilon": "ClasIS",
    }

    # ---- font settings (global) ----
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    df_plot = df.copy()
    df_plot[epsilon_cols] = df_plot[epsilon_cols].fillna(0) * denom
    df_plot = df_plot.sort_values("experiment_id")

    x = df_plot["experiment_id"].values

    # small offsets to avoid overlap
    offsets = np.linspace(-0.2, 0.2, len(epsilon_cols))

    plt.figure(figsize=(10,5))

    # ---- vertical lines ----
    for xi in x:
        plt.axvline(x=xi, linestyle="--", linewidth=0.8, alpha=0.4)

    for i, col in enumerate(epsilon_cols):
        plt.scatter(
            x + offsets[i],
            df_plot[col],
            s=60,
            label=label_mapping.get(col, col),
        )

    plt.xlabel("instance")
    plt.ylabel("maximum distance")
    plt.title(f"{dataset_display_name}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_epsilon_improved_only(df, is_acasxu=False, dataset_display_name=""):
    denom = 1 if is_acasxu else 256

    epsilon_cols = [
        "BASE_epsilon",
        "RS_dual_Z_epsilon",
        "RS_random_Z_epsilon",
        "IS_dual_ind_epsilon",
        "IS_dual_epsilon",
    ]

    compare_cols = [
        "RS_dual_Z_epsilon",
        "RS_random_Z_epsilon",
        "IS_dual_ind_epsilon",
        "IS_dual_epsilon",
    ]

    label_mapping = {
        "BASE_epsilon": "RaVeN",
        "RS_dual_Z_epsilon": "SaBRe",
        "RS_random_Z_epsilon": "RandRS",
        "IS_dual_epsilon": "DualIS",
        "IS_dual_ind_epsilon": "ClasIS",
    }

    # ---- font settings (global) ----
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    df_plot = df.copy()
    df_plot[epsilon_cols] = df_plot[epsilon_cols].fillna(0)

    # keep only rows where at least one non-base method is strictly better than BASE
    improved_mask = np.zeros(len(df_plot), dtype=bool)
    for col in compare_cols:
        improved_mask |= (df_plot[col] > df_plot["BASE_epsilon"])

    df_plot = df_plot[improved_mask].copy()

    if df_plot.empty:
        print("No improved instances found.")
        return

    # rescale for plotting
    df_plot[epsilon_cols] = df_plot[epsilon_cols] * denom

    # # x-axis: 1, 2, 3, ..., number of improved instances
    x = np.arange(1, len(df_plot) + 1)
    # x = df_plot["experiment_id"].values

    offsets = np.linspace(-0.2, 0.2, len(epsilon_cols))

    plt.figure(figsize=(10, 5))

    # ---- vertical lines ----
    for xi in x:
        plt.axvline(x=xi, linestyle="--", linewidth=0.8, alpha=0.4)
    
    for i, col in enumerate(epsilon_cols):
        plt.scatter(
            x + offsets[i],
            df_plot[col].values,
            s=60,
            label=label_mapping.get(col, col),
        )

    plt.xlabel("instance")
    plt.ylabel("maximum distance")
    plt.title(f"{dataset_display_name}")
    plt.xticks(x)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ====== Below is the code for statistical analysis (Wilcoxon signed-rank test and Cohen's d) ======

# ===== CONFIG =====
epsilon_cols = [
    "BASE_epsilon",
    "IS_dual_ind_epsilon",
    "IS_dual_epsilon",
    "RS_random_Z_epsilon",
    "RS_dual_Z_epsilon",
]

label_mapping = {
    "BASE_epsilon": "RaVeN",
    "RS_dual_Z_epsilon": "SaBRe",
    "RS_random_Z_epsilon": "RandRS",
    "IS_dual_ind_epsilon": "ClasIS",
    "IS_dual_epsilon": "DualIS",
}

# ===== FUNCTIONS =====
def wilcoxon_test(x1, x2):
    # Wilcoxon requires non-identical pairs
    diff = x1 - x2
    nonzero = diff != 0
    
    if nonzero.sum() == 0:
        return np.nan, np.nan
    
    stat, p = wilcoxon(x1[nonzero], x2[nonzero])
    return stat, p


def cohens_d_paired(x1, x2):
    diff = x1 - x2
    if diff.std(ddof=1) == 0:
        return np.nan
    return diff.mean() / diff.std(ddof=1)


def mean_diff(x1, x2):
    return (x1 - x2).mean()


def statistical_analysis_df(df, show_details=True):
    # ===== PREPROCESS =====
    df_proc = df.copy()

    # Replace NaN with 0 (your assumption)
    df_proc[epsilon_cols] = df_proc[epsilon_cols].fillna(0.0)

    # ===== PAIRWISE ANALYSIS =====
    results = []

    for c1, c2 in combinations(epsilon_cols, 2):
        x1 = df_proc[c1].values
        x2 = df_proc[c2].values
        
        stat, p = wilcoxon_test(x1, x2)
        d = cohens_d_paired(x1, x2)
        md = mean_diff(x1, x2)
        
        results.append({
            "method_1": label_mapping[c1],
            "method_2": label_mapping[c2],
            "wilcoxon_stat": stat,
            "p_value": p,
            "cohens_d": d,
            "mean_diff": md,
            "n": len(x1)
        })

    results_df = pd.DataFrame(results)

    # ===== MULTIPLE TEST CORRECTION (Holm) =====
    pvals = results_df["p_value"].values
    _, p_corrected, _, _ = multipletests(pvals, method="holm")
    results_df["p_corrected"] = p_corrected

    # ===== SORT (optional: strongest effects first) =====
    results_df = results_df.sort_values(by="cohens_d", ascending=False)

    if show_details:
        print(results_df)

    return results_df


# convert df to table
'''
approach | method_1 | method_2 | method_3 | method_4
method_1 | = | *** | * | *** 
method_2 | xxx | = | * | x 
method_3 | x | x | = | xx
method_4 | xxx | * | ** | =
'''

latex_name_mapping = {
    "RaVeN": r"\raven",
    "SaBRe": r"\tool",
    "RandRS": r"\RSrandom",
    "ClasIS": r"\baseline",
    "DualIS": r"\is",
}

def get_symbol(d, p):
    if pd.isna(d) or pd.isna(p):
        return "-"
    
    abs_d = abs(d)
    
    # strength
    if p < 0.01 and abs_d > 0.8:
        sym = "***"
    elif p < 0.05 and abs_d > 0.5:
        sym = "**"
    elif p < 0.2:
        sym = "*"
    else:
        return "-"
    
    # direction
    if d > 0:
        return sym
    else:
        return sym.replace("*", "x")  # worse
    
def get_symbol(d, p):
    if pd.isna(d) or pd.isna(p):
        return "-"
    
    abs_d = abs(d)

    if p < 0.05:
        if abs_d > 0.8:
            sym = "***"
        elif abs_d > 0.4:
            sym = "**"
        elif abs_d > 0.2:
            sym = "*"
        else:
            return "-"
    else:
        return "-"
    
    return sym if d > 0 else sym.replace("*", "x")

def statistical_analysis_table(df):
    results_df = statistical_analysis_df(df)
    
    if results_df is None:
        raise ValueError("statistical_analysis_df returned None. Did you forget to return results_df?")
    
    methods = ["RaVeN", "ClasIS", "DualIS", "RandRS", "SaBRe"]
    star_matrix = pd.DataFrame(index=methods, columns=methods)

    for _, row in results_df.iterrows():
        m1 = row["method_1"]
        m2 = row["method_2"]
        d = row["cohens_d"]
        # p = row["p_corrected"]
        p = row["p_value"]
        
        sym = get_symbol(d, p)
        
        star_matrix.loc[m1, m2] = sym
        
        if sym == "-":
            star_matrix.loc[m2, m1] = "-"
        else:
            reverse = sym.replace("*", "#").replace("x", "*").replace("#", "x")
            star_matrix.loc[m2, m1] = reverse

    for m in methods:
        star_matrix.loc[m, m] = "="

    star_matrix = star_matrix.fillna("-")

    star_matrix_latex = star_matrix.copy()

    star_matrix_latex.index = [
        latex_name_mapping[m] for m in star_matrix.index
    ]

    star_matrix_latex.columns = [
        latex_name_mapping[m] for m in star_matrix.columns
    ]

    # # move raven row to bottom
    # raven_row = star_matrix_latex.loc[r"\raven"]
    # star_matrix_latex = star_matrix_latex.drop(index=r"\raven")
    # star_matrix_latex.loc[r"\raven"] = raven_row

    # # move column to right
    # cols = list(star_matrix_latex.columns)
    # cols.remove(r"\raven")
    # cols.append(r"\raven")
    # star_matrix_latex = star_matrix_latex[cols]

    # ===== PRINT TABLE =====
    print(star_matrix)

    # ===== LATEX =====
    latex_table = star_matrix_latex.to_latex(
        escape=False,
        column_format="l" + "c" * len(star_matrix_latex.columns)
    )

    print("\n=== LATEX ===\n")
    print(latex_table)

    return star_matrix, latex_table


def cohens_d_heatmap_with_p(df, annot_mode="symbol"):
    """
    annot_mode:
        "symbol" -> show d + stars
        "pvalue" -> show d + p-value
    """

    results_df = statistical_analysis_df(df, show_details=False)

    method_order = ["RaVeN", "ClasIS", "DualIS", "RandRS", "SaBRe"]

    d_matrix = pd.DataFrame(np.nan, index=method_order, columns=method_order)
    p_matrix = pd.DataFrame(np.nan, index=method_order, columns=method_order)

    for _, row in results_df.iterrows():
        m1, m2 = row["method_1"], row["method_2"]
        d, p = row["cohens_d"], row["p_value"]

        if m1 in d_matrix.index and m2 in d_matrix.columns:
            d_matrix.loc[m1, m2] = d
            d_matrix.loc[m2, m1] = -d

            p_matrix.loc[m1, m2] = p
            p_matrix.loc[m2, m1] = p

    for m in method_order:
        d_matrix.loc[m, m] = 0.0
        p_matrix.loc[m, m] = np.nan

    display_method_order = [
        SABRE_DISPLAY_NAME if method == "SaBRe" else method
        for method in method_order
    ]
    d_matrix.index = display_method_order
    d_matrix.columns = display_method_order
    p_matrix.index = display_method_order
    p_matrix.columns = display_method_order

    # ===== dynamic color scale =====
    vmax = 1.2 * np.nanmax(np.abs(d_matrix.values))
    vmin = -vmax

    # ===== annotation =====
    def get_star(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "-"

    annot = d_matrix.copy().astype(str)

    for i in range(len(method_order)):
        for j in range(len(method_order)):
            d = d_matrix.iloc[i, j]
            p = p_matrix.iloc[i, j]

            if i == j:
                annot.iloc[i, j] = "0"
                continue

            if annot_mode == "symbol":
                # star = get_star(p)
                star = None
                if star:
                    annot.iloc[i, j] = f"{d:.2f} (p:{get_star(p)})"
                else:
                    annot.iloc[i, j] = f"{d:.2f}"
            elif annot_mode == "pvalue":
                annot.iloc[i, j] = f"{d:.2f}\n(p={p:.1e})"
            else:
                annot.iloc[i, j] = f"{d:.2f}"

    # ===== plot =====
    plt.figure(figsize=(8, 5))
    base_font_size = 16
    plt.rcParams.update({
        "font.size": base_font_size + 2,
        "figure.titlesize": base_font_size + 4,
        "axes.titlesize": base_font_size + 2,
        "axes.labelsize": base_font_size + 2,
        "xtick.labelsize": base_font_size,
        "ytick.labelsize": base_font_size,
        "legend.fontsize": base_font_size - 1,
    })
    sns.heatmap(
        d_matrix,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        center=0,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        square=False,
        cbar_kws={"label": "Cohen's d", 
                  "orientation": "horizontal",
                  "pad": 0.01},
        annot_kws={
            "color": "black",
            "fontsize": 18,
            # "fontweight": "bold"
        }
    )

    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    # return d_matrix


def statistical_analysis_merge():
    log_dir = '../result/binary_search'
    # acasxu
    aca_df = extract_binary_search_results_acasxu_all(log_dir)
    # m4
    m4_df = extract_binary_search_results_all(log_dir, 'mnist-256x4')
    # mc
    mc_df = extract_binary_search_results_all(log_dir, 'mnist-conv')
    # cifar10
    cif_df = extract_binary_search_results_all(log_dir, 'cifar10')
    # gtsrb
    gtsrb_df = extract_binary_search_results_all(log_dir, 'gtsrb')

    df_all = pd.concat([aca_df, m4_df, mc_df, cif_df, gtsrb_df], ignore_index=True)

    # star_matrix, latex_table = statistical_analysis_table(df_all)
    d_matrix = cohens_d_heatmap_with_p(df_all)


def summarize_epsilon(df: pd.DataFrame):
    """
    Compute mean and max epsilon per method.
    NaN is treated as 0.
    """

    # pick epsilon columns
    epsilon_cols = [c for c in df.columns if c.endswith("_epsilon")]

    # replace NaN → 0
    df_eps = df[epsilon_cols].fillna(0)

    results = []

    for col in epsilon_cols:
        method = col.replace("_epsilon", "")

        mean_val = df_eps[col].mean()
        max_val = df_eps[col].max()

        results.append({
            "method": method,
            "mean_epsilon": mean_val,
            "max_epsilon": max_val
        })

    return pd.DataFrame(results).sort_values("mean_epsilon", ascending=False)

header = r"""
\begin{tabular}{lcccccccc}
\toprule
\multirow{2}{*}{Method} & 
\multicolumn{2}{c}{\acasxu} & 
\multicolumn{2}{c}{\mnistF} & 
\multicolumn{2}{c}{\mnistC} & 
\multicolumn{2}{c}{\cifar} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
 & mean & max 
 & mean$^{\dagger}$ & max$^{\dagger}$ 
 & mean$^{\dagger}$ & max$^{\dagger}$ 
 & mean$^{\dagger}$ & max$^{\dagger}$ \\
\midrule
"""

def df_to_dict(df):
    return df.set_index("method").to_dict(orient="index")

def make_latex_table(aca_sum, m4_sum, mc_sum, cif_sum):
    # convert to dict
    aca = df_to_dict(aca_sum)
    m4  = df_to_dict(m4_sum)
    mc  = df_to_dict(mc_sum)
    cif = df_to_dict(cif_sum)

    # use consistent method ordering (from aca)
    methods = ["RS_dual_Z", "RS_random_Z", "IS_dual", "IS_dual_ind", "BASE"]

    def fmt_aca(x):
        return f"{x:.2f}"

    def fmt(x):
        return f"{100 * x:.2f}"

    lines = []
    lines.append(header)

    name_map = {
        "BASE": r"\raven",
        "RS_dual_Z": r"\tool",
        "RS_random_Z": r"\RSrandom",
        "IS_dual": r"\is",
        "IS_dual_ind": r"\baseline",
    }

    for m in methods:
        row = (
            f"{name_map.get(m, m)} & "
            f"{fmt_aca(aca[m]['mean_epsilon'])} & {fmt_aca(aca[m]['max_epsilon'])} & "
            f"{fmt(m4[m]['mean_epsilon'])} & {fmt(m4[m]['max_epsilon'])} & "
            f"{fmt(mc[m]['mean_epsilon'])} & {fmt(mc[m]['max_epsilon'])} & "
            f"{fmt(cif[m]['mean_epsilon'])} & {fmt(cif[m]['max_epsilon'])} \\\\"
        )
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


def all_summarize_epsilon():
    log_dir = '../result/binary_search'
    # acasxu
    aca_df = extract_binary_search_results_acasxu_all(log_dir)
    aca_sum = summarize_epsilon(aca_df)
    # m4
    m4_df = extract_binary_search_results_all(log_dir, 'mnist-256x4')
    m4_sum = summarize_epsilon(m4_df)
    # mc
    mc_df = extract_binary_search_results_all(log_dir, 'mnist-conv')
    mc_sum = summarize_epsilon(mc_df)
    # cifar10
    cif_df = extract_binary_search_results_all(log_dir, 'cifar10')
    cif_sum = summarize_epsilon(cif_df)

    # latex table
    latex_table = make_latex_table(aca_sum, m4_sum, mc_sum, cif_sum)
    print(latex_table)



from matplotlib.ticker import LogLocator

def plot_epsilon_ratio_boxplot(
    df_list,
    dataset_list,
    base_method="RS_dual_Z",
    compare_methods=None,
    base_font_size=18,
    box_plot=True,
    cap=20,
    log_scale=True,
    same_marker_color=True,
    show_points=True,
    showfliers=False,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    if compare_methods is None:
        compare_methods = ["RS_random_Z", "IS_dual_ind", "IS_dual", "BASE"]

    method_name_map = {
        "RS_dual_Z": SABRE_DISPLAY_NAME,
        "RS_random_Z": "RandRS",
        "IS_dual_ind": "ClasIS",
        "IS_dual": "DualIS",
        "BASE": "RaVeN",
    }

    net_name_map = {
        "acasxu": "ACAS Xu",
        "mnist4": "MNIST-F",
        "mnist-conv": "MNIST-C",
        "cifar10": "CIFAR",
    }

    sns.set(style="whitegrid")
    plt.rcParams.update({
        "font.size": base_font_size + 2,
        "figure.titlesize": base_font_size + 4,
        "axes.titlesize": base_font_size + 2,
        "axes.labelsize": base_font_size + 2,
        "xtick.labelsize": base_font_size,
        "ytick.labelsize": base_font_size,
        "legend.fontsize": base_font_size - 1,
    })

    n = len(df_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for df, dataset_name, ax in zip(df_list, dataset_list, axes):
        df_plot = df.copy()

        epsilon_cols = [f"{base_method}_epsilon"] + [f"{m}_epsilon" for m in compare_methods]
        epsilon_cols = [c for c in epsilon_cols if c in df_plot.columns]
        df_plot[epsilon_cols] = df_plot[epsilon_cols].fillna(0)

        # keep only rows where at least one compared method is strictly better than BASE
        improved_mask = np.zeros(len(df_plot), dtype=bool)
        base_eps_col_for_filter = "BASE_epsilon"
        if base_eps_col_for_filter in df_plot.columns:
            base_eps = df_plot[base_eps_col_for_filter].fillna(0)
            for method in compare_methods + [base_method]:  # also consider BASE for improvement (if it has epsilon)
                col_eps = f"{method}_epsilon"
                if col_eps in df_plot.columns:
                    improved_mask |= (df_plot[col_eps] > base_eps)

            df_plot = df_plot[improved_mask].copy()
        
        # print(f"{dataset_name}")
        # print(df_plot[:10])

        records = []

        base_eps_col = f"{base_method}_epsilon"
        base_status_col = f"{base_method}_status"

        for method in compare_methods:
            eps_col = f"{method}_epsilon"
            status_col = f"{method}_status"

            if eps_col not in df_plot.columns or status_col not in df_plot.columns:
                continue

            num = df_plot[base_eps_col].fillna(0)
            den = df_plot[eps_col].fillna(0)

            for i, (num_val, den_val) in enumerate(zip(num, den)):
                if den_val == 0 and num_val == 0:
                    continue
                if den_val == 0:
                    ratio = cap
                else:
                    ratio = min(num_val / den_val, cap)

                records.append({
                    "Instance": i,
                    "Method": method,
                    "Display Method": method_name_map.get(method, method),
                    "Ratio": ratio,
                    "Base Status": df_plot.iloc[i][base_status_col] if base_status_col in df_plot.columns else None,
                    "Method Status": df_plot.iloc[i][status_col],
                    "Base Eps": num_val,
                    "Method Eps": den_val,
                })

        df_long = pd.DataFrame(records)

        if df_long.empty:
            ax.set_title(net_name_map.get(dataset_name, dataset_name))
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("")
            if ax == axes[0]:
                ax.set_ylabel(f"{method_name_map.get(base_method, base_method)} / method")
            else:
                ax.set_ylabel("")
            continue

        if log_scale:
            df_long = df_long[df_long["Ratio"] > 0]

        if df_long.empty:
            ax.set_title(net_name_map.get(dataset_name, dataset_name))
            ax.text(0.5, 0.5, "No positive ratio data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("")
            if ax == axes[0]:
                ax.set_ylabel(f"{method_name_map.get(base_method, base_method)} / method")
            else:
                ax.set_ylabel("")
            continue

        color = "C0" if same_marker_color else None

        if box_plot:
            sns.boxplot(
                data=df_long,
                x="Display Method",
                y="Ratio",
                ax=ax,
                color=color,
                showcaps=True,
                boxprops={"alpha": 0.4},
                whiskerprops={"linewidth": 1, "alpha": 0.8},
                showfliers=showfliers,
                zorder=1,
                width=0.6,
                medianprops={"color": "black", "linewidth": 2, "zorder": 10},
            )

        if show_points:
            sns.stripplot(
                data=df_long,
                x="Display Method",
                y="Ratio",
                ax=ax,
                color=color if color is not None else "C0",
                size=5,
                alpha=0.6,
                jitter=0.2,
                zorder=2,
            )

        ax.axhline(1.0, ls="--", lw=1.2, color="black", alpha=0.7)

        if log_scale:
            ax.set_yscale("log")

        if ax == axes[0]:
            ax.set_ylabel(f"{method_name_map.get(base_method, base_method)} / method")
        else:
            ax.set_ylabel("")

        if log_scale:
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2,3,4,5,6,7,8,9]))
            ax.grid(True, which="major", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.4)

        ax.set_xlabel("")
        ax.set_title(net_name_map.get(dataset_name, dataset_name))
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    return fig

def plot_epsilon_scatter(
    df_list,
    dataset_list,
    base_method="RS_dual_Z",
    compare_methods=None,
    base_font_size=18,
    jitter_linear_frac=0.01,
    jitter_log_sigma=0.06,
    method_offset_frac=0.01,
    log_scale=False,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    if compare_methods is None:
        compare_methods = ["RS_random_Z", "IS_dual_ind", "IS_dual", "BASE"]

    base_method_display_name = SABRE_DISPLAY_NAME if base_method == "RS_dual_Z" else base_method
    method_name_map = {
        "RS_dual_Z": SABRE_DISPLAY_NAME,
        "RS_random_Z": "RandRS",
        "IS_dual_ind": "ClasIS",
        "IS_dual": "DualIS",
        "BASE": "RaVeN",
    }

    net_name_map = {
        "acasxu": "ACAS Xu",
        "mnist4": "MNIST-F",
        "mnist-conv": "MNIST-C",
        "cifar10": "CIFAR",
        "gtsrb": "GTSRB",
    }

    colors = {
        "RS_random_Z": "#e76400",
        "IS_dual_ind": "#1b9e77",
        "IS_dual": "#7369FF",
        "BASE": "#ff1f1f",
    }

    # palette = sns.color_palette("deep")

    # colors = {
    #     "RS_random_Z": palette[1],
    #     "IS_dual_ind": palette[2],
    #     "IS_dual": palette[3],
    #     "BASE": palette[4],
    # }

    markers = {
        "RS_random_Z": "o",
        "IS_dual_ind": "s",
        "IS_dual": "^",
        "BASE": "D",
    }

    sns.set(style="whitegrid")
    plt.rcParams.update({
        "font.size": base_font_size + 2,
        "axes.titlesize": base_font_size + 2,
        "axes.labelsize": base_font_size + 2,
        "xtick.labelsize": base_font_size,
        "ytick.labelsize": base_font_size,
        "legend.fontsize": base_font_size - 1,
    })

    n = len(df_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharex=False, sharey=False)

    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    # small deterministic offsets to separate methods a bit
    method_offsets = {
        "RS_random_Z": -1.5,
        "IS_dual_ind": -0.5,
        "IS_dual": 0.5,
        "BASE": 1.5,
    }

    for df, dataset_name, ax in zip(df_list, dataset_list, axes):
        base_eps_col = f"{base_method}_epsilon"

        plotted_any = False
        max_val = 0.0

        scale = 1.0 if dataset_name == "acasxu" else 256.0

        # compute global max first for adaptive linear jitter / offsets
        for method in compare_methods:
            eps_col = f"{method}_epsilon"
            if eps_col not in df.columns or base_eps_col not in df.columns:
                continue
            x0 = df[eps_col].fillna(0).to_numpy(dtype=float) * scale
            y0 = df[base_eps_col].fillna(0).to_numpy(dtype=float) * scale
            max_val = max(max_val, x0.max(initial=0), y0.max(initial=0))

        max_val = max(max_val, 1e-6)
        linear_jitter = jitter_linear_frac * max_val
        method_offset = method_offset_frac * max_val

        for method in compare_methods:
            eps_col = f"{method}_epsilon"
            if eps_col not in df.columns or base_eps_col not in df.columns:
                continue

            x = df[eps_col].fillna(0).to_numpy(dtype=float) * scale
            y = df[base_eps_col].fillna(0).to_numpy(dtype=float) * scale

            if log_scale:
                # only positive points are valid on log scale
                mask = (x > 0) & (y > 0)
                x = x[mask]
                y = y[mask]

                if len(x) == 0:
                    continue

                # multiplicative jitter for log scale
                x = x * np.exp(rng.normal(0, jitter_log_sigma, size=len(x)))
                y = y * np.exp(rng.normal(0, jitter_log_sigma, size=len(y)))

                # small multiplicative offset by method
                offset = method_offsets.get(method, 0.0) * jitter_log_sigma * 0.15
                x = x * np.exp(offset)
                y = y * np.exp(-offset * 0.3)

            else:
                # additive jitter for linear scale
                x = x + rng.normal(0, linear_jitter, size=len(x))
                y = y + rng.normal(0, linear_jitter, size=len(y))

                # small additive offset by method
                offset = method_offsets.get(method, 0.0) * method_offset
                x = x + offset
                y = y - 0.3 * offset

                x = np.clip(x, 0, None)
                y = np.clip(y, 0, None)

            ax.scatter(
                x,
                y,
                alpha=0.65,
                s=42,
                color=colors.get(method, "C0"),
                marker=markers.get(method, "o"),
                label=method_name_map.get(method, method),
                edgecolors="none",
            )
            plotted_any = True

        if not plotted_any:
            ax.set_title(net_name_map.get(dataset_name, dataset_name))
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Compared method (*/256)")
            ax.set_ylabel(f"{base_method_display_name} (*/256)")
            continue

        # diagonal y = x
        if log_scale:
            min_pos = max(1e-3, min(
                [
                    np.min(df[f"{m}_epsilon"].fillna(0).to_numpy(dtype=float) * scale[
                        (df[f"{m}_epsilon"].fillna(0).to_numpy(dtype=float) * scale) > 0
                    ])
                    for m in compare_methods
                    if f"{m}_epsilon" in df.columns and np.any((df[f"{m}_epsilon"].fillna(0).to_numpy(dtype=float) * scale) > 0)
                ] + [
                    np.min(df[base_eps_col].fillna(0).to_numpy(dtype=float) * scale[
                        (df[base_eps_col].fillna(0).to_numpy(dtype=float) * scale) > 0
                    ])
                ]
            ))
            ax.plot([min_pos, max_val], [min_pos, max_val], "k--", linewidth=1.2, alpha=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.2, alpha=0.8)

        with_scale = " (*/256)" if dataset_name != "acasxu" else ""

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"$\\varepsilon^*$ of Compared method")
        ax.set_ylabel(f"$\\varepsilon^*$ of {base_method_display_name}")
        ax.set_title(net_name_map.get(dataset_name, dataset_name))

        if dataset_name != "acasxu" and not log_scale:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.grid(True, linestyle="--", alpha=0.7)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"$\\varepsilon^*$ of Compared method")
        ax.set_ylabel(f"$\\varepsilon^*$ of {base_method_display_name}")
        ax.set_title(net_name_map.get(dataset_name, dataset_name))
        ax.grid(True, linestyle="--", alpha=0.7)

        if dataset_name != "acasxu":
            ax.annotate(
                "(/256)",
                xy=(1, 0), xycoords="axes fraction",
                xytext=(45, -25), textcoords="offset points",
                ha="right", va="bottom",
                fontsize=base_font_size
            )

            ax.annotate(
                "(/256)",
                xy=(0, 1), xycoords="axes fraction",
                xytext=(-40, 20), textcoords="offset points",
                ha="left", va="top",
                fontsize=base_font_size
            )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=True)

    plt.tight_layout()
    return fig

def plot_epsilon_scatter_grouped(
    df_list,
    dataset_list,
    base_method="RS_dual_Z",
    base_font_size=18,
    jitter=0.005,
    log_scale=False,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ===== method groups =====
    group_1 = ["BASE", "RS_random_Z"]       # RAVEN, RandRS
    group_2 = ["IS_dual_ind", "IS_dual"]    # ClasIS, DualIS

    method_name_map = {
        "RS_dual_Z": "SaBRe",
        "RS_random_Z": "RandRS",
        "IS_dual_ind": "ClasIS",
        "IS_dual": "DualIS",
        "BASE": "RaVeN",
    }

    colors = {
        "BASE": "C4",
        "RS_random_Z": "C1",
        "IS_dual_ind": "C2",
        "IS_dual": "C3",
    }

    markers = {
        "BASE": "D",
        "RS_random_Z": "o",
        "IS_dual_ind": "s",
        "IS_dual": "^",
    }

    net_name_map = {
        "acasxu": "ACAS Xu",
        "mnist4": "MNIST-F",
        "mnist-conv": "MNIST-C",
        "cifar10": "CIFAR",
    }

    sns.set(style="whitegrid")
    plt.rcParams.update({
        "font.size": base_font_size + 2,
        "axes.titlesize": base_font_size + 2,
        "axes.labelsize": base_font_size + 2,
    })

    n = len(df_list)

    # 2 columns per dataset
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))

    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for row_idx, (df, dataset_name) in enumerate(zip(df_list, dataset_list)):
        base_eps_col = f"{base_method}_epsilon"

        if dataset_name == "acasxu":
            scale = 1.0
        else:
            scale = 256.0

        for col_idx, methods in enumerate([group_1, group_2]):
            ax = axes[row_idx][col_idx]

            max_val = 0.0

            for method in methods:
                eps_col = f"{method}_epsilon"

                if eps_col not in df.columns:
                    continue

                x = df[eps_col].fillna(0).to_numpy() * scale
                y = df[base_eps_col].fillna(0).to_numpy() * scale

                max_val = max(max_val, x.max(initial=0), y.max(initial=0))

                # ===== jitter =====
                if log_scale:
                    mask = (x > 0) & (y > 0)
                    x, y = x[mask], y[mask]
                    x *= np.exp(rng.normal(0, jitter, size=len(x)))
                    y *= np.exp(rng.normal(0, jitter, size=len(y)))
                else:
                    x += rng.normal(0, jitter, size=len(x))
                    y += rng.normal(0, jitter, size=len(y))
                    x = np.clip(x, 0, None)
                    y = np.clip(y, 0, None)

                ax.scatter(
                    x,
                    y,
                    color=colors[method],
                    marker=markers[method],
                    alpha=0.65,
                    s=45,
                    label=method_name_map[method],
                )

            max_val = max(max_val, 1e-6)

            # diagonal
            ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.2)

            if log_scale:
                ax.set_xscale("log")
                ax.set_yscale("log")

            ax.set_aspect("equal", adjustable="box")

            # labels
            x_suffix = "RaVeN / RandRS" if col_idx == 0 else "ClasIS / DualIS"
            with_scale = f" (x256)" if dataset_name != "acasxu" else ""
            if row_idx == n - 1:
                ax.set_xlabel(f"{x_suffix}{with_scale}")
            if col_idx == 0:
                ax.set_ylabel(f"SaBRe{with_scale}")

            ax.set_title(f"{net_name_map.get(dataset_name)}")

            ax.grid(True, linestyle="--", alpha=0.7)

            ax.legend(frameon=True)

    plt.tight_layout()
    return fig

def plot_epsilon_ratio_boxplot_all(cap=20, log_scale=False):
    log_dir = '../result/binary_search'
    # acasxu
    aca_df = extract_binary_search_results_acasxu_all(log_dir)
    # m4
    m4_df = extract_binary_search_results_all(log_dir, 'mnist-256x4')
    # mc
    mc_df = extract_binary_search_results_all(log_dir, 'mnist-conv')
    # cifar10
    cif_df = extract_binary_search_results_all(log_dir, 'cifar10')
    # gtsrb
    gtsrb_df = extract_binary_search_results_all(log_dir, 'gtsrb')

    # plot_epsilon_ratio_boxplot(
    #     df_list=[mc_df],
    #     dataset_list=["mnist-conv"],
    #     base_method="RS_dual_Z",
    #     compare_methods=["RS_random_Z", "IS_dual_ind", "IS_dual", "BASE"],
    #     box_plot=True,
    #     cap=cap,
    #     log_scale=log_scale,
    #     same_marker_color=True
    # )

    # plot_epsilon_scatter_grouped(
    #     df_list=[aca_df, m4_df, mc_df, cif_df],
    #     dataset_list=["acasxu", "mnist4", "mnist-conv", "cifar10"],
    #     base_method="RS_dual_Z",
    #     base_font_size=18,
    # )

    # plot_epsilon_scatter_grouped(
    #     df_list=[aca_df, m4_df],
    #     dataset_list=["acasxu", "mnist4"],
    #     base_method="RS_dual_Z",
    #     base_font_size=18,
    # )

    # plot_epsilon_scatter_grouped(
    #     df_list=[mc_df, cif_df],
    #     dataset_list=["mnist-conv", "cifar10"],
    #     base_method="RS_dual_Z",
    #     base_font_size=18,
    # )

    plot_epsilon_scatter(
        df_list=[aca_df, m4_df, mc_df, cif_df, gtsrb_df],
        dataset_list=["acasxu", "mnist4", "mnist-conv", "cifar10", "gtsrb"],
        base_method="RS_dual_Z",
        compare_methods=["BASE", "IS_dual_ind", "IS_dual", "RS_random_Z", ],
        base_font_size=18,
    )

def count_epsilon_better(base_method="RS_dual_Z", compare_methods=None):

    log_dir = '../result/binary_search'
    # acasxu
    aca_df = extract_binary_search_results_acasxu_all(log_dir)
    # m4
    m4_df = extract_binary_search_results_all(log_dir, 'mnist-256x4')
    # mc
    mc_df = extract_binary_search_results_all(log_dir, 'mnist-conv')
    # cifar10
    cif_df = extract_binary_search_results_all(log_dir, 'cifar10')
    # gtsrb
    gtsrb_df = extract_binary_search_results_all(log_dir, 'gtsrb')

    df_list = [aca_df, m4_df, mc_df, cif_df, gtsrb_df]
    dataset_list = ["acasxu", "mnist4", "mnist-conv", "cifar10", "gtsrb"]

    if compare_methods is None:
        compare_methods = ["RS_random_Z", "IS_dual_ind", "IS_dual", "BASE"]

    method_name_map = {
        "RS_dual_Z": "SaBRe",
        "RS_random_Z": "RandRS",
        "IS_dual_ind": "ClasIS",
        "IS_dual": "DualIS",
        "BASE": "RaVeN",
    }

    net_name_map = {
        "acasxu": "ACAS Xu",
        "mnist4": "MNIST-F",
        "mnist-conv": "MNIST-C",
        "cifar10": "CIFAR",
        "gtsrb": "GTSRB",
    }

    results = []

    for df, dataset_name in zip(df_list, dataset_list):
        base_eps_col = f"{base_method}_epsilon"

        for method in compare_methods:
            eps_col = f"{method}_epsilon"

            if eps_col not in df.columns or base_eps_col not in df.columns:
                continue

            base_eps = df[base_eps_col].fillna(0)
            method_eps = df[eps_col].fillna(0)

            better_count = (method_eps < base_eps).sum()
            worse_count = (method_eps > base_eps).sum()
            tie_count = (method_eps == base_eps).sum()
            total_count = len(df)

            results.append({
                "Dataset": net_name_map.get(dataset_name, dataset_name),
                "Method": method_name_map.get(method, method),
                "Better Count": better_count,
                "Worse Count": worse_count,
                "Tie Count": tie_count,
                "Total Count": total_count,
                "Better Ratio": better_count / total_count if total_count > 0 else 0.0,
            })

    return pd.DataFrame(results)
