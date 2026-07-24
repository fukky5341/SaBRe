import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from analysis import *
from util import method_name_map, method_to_time_col, \
    method_to_status_col, get_time_budget

dsns_file = '_dsns_whole.csv'
acasxu = pd.read_csv(f"./acasxu/acasxu{dsns_file}")
mnist4 = pd.read_csv(f"./mnist-256x4/mnist4{dsns_file}")
mnist_conv = pd.read_csv(f"./mnist-conv/mnistconv{dsns_file}")
cifar10 = pd.read_csv(f"./cifar10/cifar{dsns_file}")
gtsrb = pd.read_csv(f"./gtsrb/gtsrb{dsns_file}")


def show_color_marker_samples():
    np.random.seed(0)
    x = np.linspace(0, 10, 20)
    y_base = np.sin(x)

    palettes = {
        "tab10": plt.cm.tab10.colors,
        "Set2": sns.color_palette("Set2"),
        "Dark2": sns.color_palette("Dark2"),
        "Pastel1": sns.color_palette("Pastel1"),
        "viridis": plt.cm.viridis(np.linspace(0, 1, 10)),
        "matplotlib default (C0~C9)": ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'],
    }

    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '>', '<']

    fig, axes = plt.subplots(len(palettes), 1, figsize=(10, 3 * len(palettes)))
    if len(palettes) == 1:
        axes = [axes]

    for ax, (name, colors) in zip(axes, palettes.items()):
        ax.set_title(f"{name} palette", fontsize=12, weight='bold')
        for i, (color, marker) in enumerate(zip(colors, markers)):
            ax.scatter(
                x + i * 0.1,                      # 少し横にずらす
                y_base + np.random.randn(len(x)) * 0.05,
                color=color,
                marker=marker,
                s=100,
                label=f"{marker}",
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
            )
        ax.legend(title="marker", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()


def get_subproblems_num_list(folder_path, df, acasxu=False):
    target_indicies = df.loc[df['base status'] != 'VERIFIED', 'name'].tolist()
    csv_files = read_csvfolder(folder_path)
    # subproblems_num = 0
    subproblems_num_list = []
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
        # subproblems_num += df_log['name'].nunique()
        subproblems_num_list.append(df_log['name'].nunique())
    return subproblems_num_list


def plot_subproblem_histogram(acasxu_l, mnist4_l, mnist_conv_l, cifar10_l, gtsrb_l, bins, base_font_size=16):
    dataset_names = ['ACAS Xu', 'MNIST-F', 'MNIST-C', 'CIFAR', 'GTSRB']

    plt.rcParams.update({
        'font.size': base_font_size,
        'axes.titlesize': base_font_size + 2,
        'axes.labelsize': base_font_size,
        'xtick.labelsize': base_font_size,
        'ytick.labelsize': base_font_size,
    })

    # 各データセットごとに histogram（ビンごとのカウント）を計算
    hist_counts = {}
    for name, values in zip(dataset_names, [acasxu_l, mnist4_l, mnist_conv_l, cifar10_l, gtsrb_l]):
        counts, _ = np.histogram(values, bins=bins)
        hist_counts[name] = counts  # shape: (n_bins,)

    n_datasets = len(dataset_names)
    n_bins = len(bins) - 1
    x_gap = 2
    x = np.arange(n_datasets) * x_gap

    # labels for each bin
    bin_labels = []
    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]
        if i == n_bins - 1:
            # 最後のビンは "3000+" のように表記
            bin_labels.append(f"{left}+")
        else:
            bin_labels.append(f"{left}-{right}")

    # colors
    colors = ['C9', 'C1', 'C4', 'C3', 'C6', 'C2', 'C0', 'C5', 'C7', 'C8']

    plt.figure(figsize=(8, 6))

    # bottom positions for stacked bars
    bottom = np.zeros(n_datasets)

    # plot stacked bars for each bin
    for b in range(n_bins):
        counts_for_bin = np.array([hist_counts[name][b] for name in dataset_names])
        plt.bar(
            x,
            counts_for_bin,
            bottom=bottom,
            color=colors[b % len(colors)],
            edgecolor='black',
            label=bin_labels[b],
            alpha=0.8,
            width=1.5
        )
        bottom += counts_for_bin

    # axes, labels, etc.
    plt.xticks(x, dataset_names, rotation=45, ha='right')
    plt.ylabel("Number of instances")
    plt.xlabel("Dataset")
    plt.title("Instances Distribution")
    # plt.grid(axis='y', linestyle='--', alpha=0.3)

    # legend
    plt.legend(
        title="#Subproblem",
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.,
        frameon=False
    )

    plt.tight_layout()
    # plt.show()

    return plt


def get_subproblems_distribution(acasxu, mnist4, mnist_conv, cifar10, gtsrb, method = "DS_random_Z_threshold", bins=[0, 10, 30, 60, 100, 200, 350, 600], base_font_size=18):
    acasxu_subproblems_num_list = get_subproblems_num_list(f"./acasxu/{method}", acasxu, acasxu=True)
    mnist4_subproblems_num_list = get_subproblems_num_list(f"./mnist-256x4/{method}", mnist4)
    mnist_conv_subproblems_num_list = get_subproblems_num_list(f"./mnist-conv/{method}", mnist_conv)
    cifar10_subproblems_num_list = get_subproblems_num_list(f"./cifar10/{method}", cifar10)
    if 'DS_random_Z_threshold' == method:
        gtsrb_method = 'RS_random_Z_threshold'
    else:
        raise NotImplementedError(f"Method {method} not implemented for GTSRB")
    gtsrb_subproblems_num_list = get_subproblems_num_list(f"./gtsrb/{gtsrb_method}", gtsrb)
    fig = plot_subproblem_histogram(
        acasxu_subproblems_num_list,
        mnist4_subproblems_num_list,
        mnist_conv_subproblems_num_list,
        cifar10_subproblems_num_list,
        gtsrb_subproblems_num_list,
        bins, base_font_size)

    return fig


def plot_time_ratios(df, net_name, methods, base_font_size=18, highlight_fastest=False, same_marker_color=False):
    time_budget = get_time_budget(df, net_name)

    # --- plot settings ---
    plt.rcParams.update({
        'font.size': base_font_size + 4,
        'axes.titlesize': base_font_size + 4,
        'axes.labelsize': base_font_size + 4,
        'xtick.labelsize': base_font_size,
        'ytick.labelsize': base_font_size,
        'legend.fontsize': base_font_size,
    })

    plt.figure(figsize=(10, 6))

    # markers
    if same_marker_color:
        markers = ['o'] * len(methods)
    else:
        markers = ['X', 'o', '^', 'P', 'v']
    # colors
    if same_marker_color:
        colors = ['C0'] * len(methods)
    else:
        colors = ['C9', 'C1', 'C4', 'C3', 'C7']

    # prepare x-axis
    n_instances = len(df)
    x_positions = np.arange(n_instances)

    time_ratios_all = {}
    for method in methods:
        time_col = method_to_time_col[method]
        times = df[time_col].clip(upper=time_budget)
        time_ratios = times / time_budget
        time_ratios_all[method] = time_ratios

    time_ratios_df = pd.DataFrame(time_ratios_all)
    min_method = time_ratios_df.idxmin(axis=1)

    for i, method in enumerate(methods):
        y = time_ratios_df[method]
        if highlight_fastest:
            # highlight points where this method is the fastest
            highlight_mask = (min_method == method)
            plt.scatter(x_positions[highlight_mask], y[highlight_mask],
                        marker=markers[i % len(markers)],
                        color=colors[i],
                        s=120, alpha=0.9,
                        edgecolors='black', linewidths=1.2,
                        zorder=10)
        # plot other points
        plt.scatter(x_positions, y,
                    label=method_name_map[method],
                    marker=markers[i % len(markers)], color=colors[i],
                    s=80, alpha=0.7, zorder=1)

    plt.xlabel('Instance')
    plt.ylabel(f'Time Ratio (time / time budget)')
    plt.title('Time Ratios per Instance (Highlighted: Fastest Method)')
    plt.ylim(0, 1.1)
    plt.gca().set_axisbelow(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(False)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    # plt.grid(axis='y')
    # plt.legend(
    #     bbox_to_anchor=(1, 1),
    #     loc='upper left',
    #     borderaxespad=0.,
    #     frameon=False
    # )
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(methods),
        frameon=False
    )
    # plt.show()

    return plt


def plot_time_ratios_separate(dfs, net_names, methods, box_plot=True, base_font_size=18,
                              highlight_fastest=False, same_marker_color=True,
                              exclude_ratio1=False, title=True):
    # display name
    sabre_display_name = r'S\textsc{a}BR\textsc{e}' if plt.rcParams.get('text.usetex', False) else 'SᴀBRᴇ'
    method_name_temp = {
        'DS_dual_Z': sabre_display_name,
        'NS': 'DualIS',
        'NSInd': 'ClasIS',
    }
    net_name_to_display_name = {
        "acasxu": "ACAS Xu",
        "mnist4": "MNIST-F",
        "mnist-conv": "MNIST-C",
        "cifar10": "CIFAR",
        "gtsrb": "GTSRB",
    }
    # --- plot settings ---
    sns.set(style="whitegrid")
    # sns.set_context("talk", font_scale=1.3)

    plt.rcParams.update({
        'font.size': base_font_size + 4,
        'figure.titlesize': base_font_size + 6,
        'axes.titlesize': base_font_size + 4,
        'axes.labelsize': base_font_size + 4,
        'xtick.labelsize': base_font_size + 4,
        'ytick.labelsize': base_font_size + 4,
        'legend.fontsize': base_font_size,
    })

    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.8), sharey=True)

    if n == 1:
        axes = [axes]

    return_dfs = []

    for net_name, df, ax in zip(net_names, dfs, axes):
        time_budget = get_time_budget(df, net_name)

        # --- prepare data to plot ---
        records = []
        for m in methods:
            status_col = method_to_status_col[m]
            time_col = method_to_time_col[m]

            temp_df = df.copy()
            temp_df.loc[temp_df[status_col] != 'VERIFIED', time_col] = time_budget

            ratios = ((temp_df[time_col].clip(upper=time_budget)) / time_budget) * 100
            for i, r in enumerate(ratios):
                records.append({'Instance': i,
                                'Method': m,
                                'Time Ratio': r,
                                'Status': temp_df.iloc[i][status_col],
                                'Display Method': method_name_temp[m]})
        df_long = pd.DataFrame(records)

        return_dfs.append(df_long)

        if exclude_ratio1:
            df_long = df_long[df_long['Time Ratio'] < 100]

        # identify fastest method per instance
        if highlight_fastest:
            df_pivot = df_long.pivot(index='Instance', columns='Method', values='Time Ratio')
            min_per_instance = df_pivot.min(axis=1)
            df_long['is_min'] = (
                (df_long['Time Ratio'] == df_long['Instance'].map(min_per_instance)) &
                (df_long['Status'] == 'VERIFIED')
            )
        else:
            df_long['is_min'] = False

        # --- colors ---

        if same_marker_color:
            colors = ['C0'] * len(methods)
        else:
            colors = ['C9', 'C1', 'C4', 'C3', 'C7']
        unique_display_methods = df_long['Display Method'].unique()

        palette = {m: colors[i % len(colors)] for i, m in enumerate(unique_display_methods)}

        # boxplot
        if box_plot:
            sns.boxplot(
                data=df_long, x='Display Method', y='Time Ratio',
                ax=ax,
                hue='Display Method', legend=False,
                palette=palette,
                showcaps=True,
                boxprops={'alpha': 0.4},
                whiskerprops={'linewidth': 1, 'alpha': 0.8},
                showfliers=False,
                zorder=1,
                width=0.6,
                medianprops={'color': 'black', 'linewidth': 2, 'zorder': 10}
            )

        # all points
        sns.stripplot(
            data=df_long[~df_long['is_min']],
            x='Display Method', y='Time Ratio',
            ax=ax,
            hue='Display Method', palette=palette,
            size=5, alpha=0.7,
            jitter=0.2, zorder=2
        )

        # fastest points
        sns.stripplot(
            data=df_long[df_long['is_min']],
            x='Display Method', y='Time Ratio',
            ax=ax,
            hue='Display Method', palette=palette,
            size=7, alpha=0.9,
            edgecolor='black', linewidth=1.2,
            jitter=0.2, zorder=5,
            label='Fastest (VERIFIED)'
        )

        # decorations
        ax.axhline(100.0, ls="--", lw=1, color="gray", alpha=0.6)
        ymax = max(100, df_long['Time Ratio'].max())
        ax.set_ylim(0, ymax * 1.05)
        ax.set_xlabel("")
        ax.set_ylabel("Time Ratio (%)" if ax == axes[0] else "")
        if title:
            display_net_name = net_name_to_display_name[net_name]
            ax.set_title(display_net_name)

        # # line to separate
        # if ax != axes[-1]:
        #     ax.axvline(len(methods) - 0.5, ls="--", lw=2, color="gray", alpha=0.6)

    # overall
    # fig.suptitle('Time Ratio Comparison Across Datasets', y=0.98)
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    return fig, return_dfs


def plot_time_ratio_two_method(df, net_name, method_x, method_y, base_font_size=18):
    time_budget = get_time_budget(df, net_name)

    # --- plot settings ---
    plt.rcParams.update({
        'font.size': base_font_size + 4,
        'axes.titlesize': base_font_size + 4,
        'axes.labelsize': base_font_size + 4,
        'xtick.labelsize': base_font_size,
        'ytick.labelsize': base_font_size,
        'legend.fontsize': base_font_size,
    })

    # compute time ratios
    x = df[method_to_time_col[method_x]].clip(upper=time_budget) / time_budget
    y = df[method_to_time_col[method_y]].clip(upper=time_budget) / time_budget

    plt.figure(figsize=(7, 7))

    plt.scatter(x, y, s=100, alpha=0.8, edgecolor='black')

    # identity line (equal performance)
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)

    # display method names
    x_name = method_name_map[method_x]
    y_name = method_name_map[method_y]

    plt.xlabel(f'{x_name} Time Ratio')
    plt.ylabel(f'{y_name} Time Ratio')
    # plt.title(f'Time Ratio Comparison: {x_name} vs {y_name}')
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    # plt.show()

    return plt

def draw_boxplot():
    datasets = ['acasxu', 'mnist4', 'mnist-conv', 'cifar10', 'gtsrb']
    dfs = []
    for df in [acasxu, mnist4, mnist_conv, cifar10, gtsrb]:
        temp_df = df[df['base status'] != 'VERIFIED']
        dfs.append(temp_df)
    # for dataset_pair, df_pair in zip([datasets[:2], datasets[2:]], [dfs[:2], dfs[2:]]):
    #     metnods = ['NSInd', 'NS', 'DS_dual_Z']
    #     eithers = []
    #     for dataset, df in zip(dataset_pair, df_pair):
    #         eithers.append(sort_df_either_solved(df, dataset, metnods))
    #     fig = plot_time_ratios_separate(eithers, dataset_pair, metnods, base_font_size=26, exclude_ratio1=False)
    #     fig.savefig(f'../image/result_fig/box_plot_dsns_{"_".join(dataset_pair)}.png', bbox_inches='tight')
    #     print(dataset_pair)
    #     fig.show()

    result_df_list = []
    for dataset, df in zip(datasets, dfs):
        metnods = ['NSInd', 'NS', 'DS_dual_Z']
        eithers = []
        eithers.append(sort_df_either_solved(df, dataset, metnods))
        fig, return_dfs = plot_time_ratios_separate(eithers, [dataset], metnods, base_font_size=18, exclude_ratio1=False)
        # fig.savefig(f'../image/result_fig/box_plot_dsns_{dataset}.png', bbox_inches='tight')
        # print(dataset)
        fig.show()
        result_df_list.extend(return_dfs)