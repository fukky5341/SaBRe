from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

try:
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover - notebook convenience fallback
    display = print
    Markdown = None

from generate_csv_dp import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RESULT_ROOT,
    DEFAULT_SUMMARY,
    build_result_df,
    export_summary_csv_with_filter,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SOLVED_STATUSES = {"VERIFIED", "ADV_EXAMPLE"}


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value)


def ratio_to_str(ratio: Any) -> str:
    return str(float(ratio))


def load_summary_df(
    result_root: Path,
    keep_non_threshold_logs: bool,
    summary_csv: Path | None = None,
) -> pd.DataFrame:
    summary_df = export_summary_csv_with_filter(
        result_root=result_root,
        save_path=summary_csv,
        keep_non_threshold_logs=keep_non_threshold_logs,
    )

    if "has_threshold_suffix" in summary_df.columns:
        if keep_non_threshold_logs:
            return summary_df
        return summary_df[summary_df["has_threshold_suffix"] == True].copy()  # noqa: E712

    return summary_df


def resolve_log_path(row: dict[str, Any], result_root: Path) -> Path:
    ratio = ratio_to_str(row["perturbation_ratio"])
    suffix = "_threshold" if bool(row.get("has_threshold_suffix", False)) else ""
    return (
        result_root
        / f"{row['tool']}_dimperturb_{ratio}{suffix}"
        / f"d{int(row['d'])}_e{int(row['e'])}"
        / str(int(row["exe_id"]))
        / "log.md"
    )


def resolve_detail_path(row: dict[str, Any], output_root: Path) -> Path:
    ratio = ratio_to_str(row["perturbation_ratio"])
    return output_root / f"{row['tool']}_dp{ratio}" / f"d{int(row['d'])}_e{int(row['e'])}_{int(row['exe_id'])}.csv"


def load_detail_df(row: dict[str, Any], result_root: Path, output_root: Path) -> pd.DataFrame | None:
    csv_path = resolve_detail_path(row, output_root)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    log_path = resolve_log_path(row, result_root)
    if log_path.exists():
        return build_result_df(log_path)
    return None


def parse_unstable_counts(value: Any) -> list[dict[str, Any]]:
    if is_missing(value) or value == "":
        return []
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except Exception:
        return []


def parse_unstable_counts_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_counts = parse_unstable_counts(row.get("unstable_relu_counts"))
    if raw_counts:
        return raw_counts

    unstable_counts: list[dict[str, Any]] = []
    index = 0
    while True:
        prefix = f"unstable_{index}_"
        layer_idx_key = f"{prefix}layer_idx"
        if layer_idx_key not in row or is_missing(row.get(layer_idx_key)):
            break

        unstable_counts.append(
            {
                "layer_idx": int(row.get(f"{prefix}layer_idx")),
                "type": row.get(f"{prefix}type"),
                "total": int(row.get(f"{prefix}total")),
                "inp1_unstable": int(row.get(f"{prefix}inp1_unstable")),
                "inp2_unstable": int(row.get(f"{prefix}inp2_unstable")),
                "delta_unstable": int(row.get(f"{prefix}delta_unstable")),
            }
        )
        index += 1

    return unstable_counts


def parse_first_unstable_counts_from_log_text(log_text: str) -> list[dict[str, Any]]:
    block_match = re.search(
        r"### Unstable ReLU Count \(Linear/Conv2D Layers\)\n(?P<body>.*?)(?=^###\s|\Z)",
        log_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if block_match is None:
        return []

    body = block_match.group("body")
    unstable_counts: list[dict[str, Any]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        line_match = re.match(
            r"- layer_idx=(?P<layer_idx>\d+), type=(?P<type>[^,]+), total=(?P<total>\d+), "
            r"inp1_unstable=(?P<inp1_unstable>\d+), inp2_unstable=(?P<inp2_unstable>\d+), "
            r"delta_unstable=(?P<delta_unstable>\d+)",
            line,
        )
        if line_match is None:
            continue
        unstable_counts.append(
            {
                "layer_idx": int(line_match.group("layer_idx")),
                "type": line_match.group("type"),
                "total": int(line_match.group("total")),
                "inp1_unstable": int(line_match.group("inp1_unstable")),
                "inp2_unstable": int(line_match.group("inp2_unstable")),
                "delta_unstable": int(line_match.group("delta_unstable")),
            }
        )
    return unstable_counts


def load_first_unstable_counts_from_row(row: dict[str, Any], result_root: Path) -> list[dict[str, Any]]:
    log_path = resolve_log_path(row, result_root)
    if not log_path.exists():
        return []

    try:
        log_text = log_path.read_text(encoding="utf-8")
    except Exception:
        return []

    return parse_first_unstable_counts_from_log_text(log_text)


def build_analysis_records(
    summary_df: pd.DataFrame,
    result_root: Path,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    instance_records: list[dict[str, Any]] = []
    unstable_records: list[dict[str, Any]] = []

    for row in summary_df.to_dict(orient="records"):
        detail_df = load_detail_df(row, result_root=result_root, output_root=output_root)
        if detail_df is None:
            print(f"Skipping missing detail csv/log for {row['tool']} d{row['d']}_e{row['e']}_{row['exe_id']}")
            continue

        unstable_counts = parse_unstable_counts_from_row(row)
        resolved_status = row["final_status"] if not is_missing(row.get("final_status")) else row["base_status"]
        resolved_time = row["final_total_time"] if not is_missing(row.get("final_total_time")) else row["base_total_time"]
        time_budget = row.get("time_budget")
        solved = resolved_status in SOLVED_STATUSES
        time_ratio = (resolved_time / time_budget) if not is_missing(time_budget) and time_budget else None
        subproblems = int((detail_df["level"] > 0).sum()) if "level" in detail_df.columns else max(len(detail_df) - 1, 0)
        individual_unstable = sum(unstable["inp1_unstable"] + unstable["inp2_unstable"] for unstable in unstable_counts)
        relational_unstable = sum(unstable["delta_unstable"] for unstable in unstable_counts)
        total_neurons = sum(unstable["total"] for unstable in unstable_counts)
        individual_total_relus = 2 * total_neurons
        relational_total_relus = total_neurons
        total_relus = 3 * total_neurons
        individual_unstable_ratio = (individual_unstable / individual_total_relus) if individual_total_relus else None
        relational_unstable_ratio = (relational_unstable / relational_total_relus) if relational_total_relus else None
        total_unstable_ratio = ((individual_unstable + relational_unstable) / total_relus) if total_relus else None

        instance_records.append(
            {
                "tool": row["tool"],
                "perturbation_ratio": float(row["perturbation_ratio"]),
                "d": int(row["d"]),
                "e": int(row["e"]),
                "exe_id": int(row["exe_id"]),
                "has_threshold_suffix": bool(row.get("has_threshold_suffix", False)),
                "base_only": bool(row.get("base_only", False)),
                "resolved_status": resolved_status,
                "resolved_time": resolved_time,
                "time_budget": time_budget,
                "time_ratio": time_ratio,
                "solved": solved,
                "subproblems": subproblems,
                "individual_unstable": individual_unstable,
                "relational_unstable": relational_unstable,
                "total_delta_unstable": relational_unstable,
                "individual_total_relus": individual_total_relus,
                "relational_total_relus": relational_total_relus,
                "total_relus": total_relus,
                "individual_unstable_ratio": individual_unstable_ratio,
                "relational_unstable_ratio": relational_unstable_ratio,
                "total_unstable_ratio": total_unstable_ratio,
            }
        )

        for unstable in unstable_counts:
            unstable_records.append(
                {
                    "tool": row["tool"],
                    "perturbation_ratio": float(row["perturbation_ratio"]),
                    "d": int(row["d"]),
                    "e": int(row["e"]),
                    "exe_id": int(row["exe_id"]),
                    "layer_idx": int(unstable["layer_idx"]),
                    "type": unstable["type"],
                    "total": int(unstable["total"]),
                    "inp1_unstable": int(unstable["inp1_unstable"]),
                    "inp2_unstable": int(unstable["inp2_unstable"]),
                    "delta_unstable": int(unstable["delta_unstable"]),
                }
            )

    return pd.DataFrame(instance_records), pd.DataFrame(unstable_records)


def annotate_either_solved(instance_df: pd.DataFrame, comparison_tools: tuple[str, ...] | None) -> pd.DataFrame:
    if instance_df.empty or not comparison_tools:
        return instance_df

    key_cols = ["perturbation_ratio", "d", "e", "exe_id"]
    selected_df = instance_df[instance_df["tool"].isin(comparison_tools)].copy()
    if selected_df.empty:
        return instance_df

    either_solved_map = selected_df.groupby(key_cols, dropna=False)["solved"].any().rename("either_solved")
    return instance_df.merge(either_solved_map, on=key_cols, how="left")


def summarize_instances(instance_df: pd.DataFrame) -> pd.DataFrame:
    if instance_df.empty:
        return instance_df

    def summarize_group(group: pd.DataFrame) -> pd.Series:
        solved_group = group[group["solved"]]
        either_group = group[group["either_solved"]] if "either_solved" in group.columns else solved_group
        return pd.Series(
            {
                "instances": len(group),
                "solved_instances": int(group["solved"].sum()),
                "solved_rate": float(group["solved"].mean()),
                "base_only_instances": int(group["base_only"].sum()),
                "mean_time_ratio_all": group["time_ratio"].mean(),
                "mean_time_ratio_solved": solved_group["time_ratio"].mean(),
                "mean_time_either": either_group["time_ratio"].mean(),
                "mean_subproblems_either": either_group["subproblems"].mean(),
                "mean_subproblems_all": group["subproblems"].mean(),
                "mean_subproblems_solved": solved_group["subproblems"].mean(),
                "mean_individual_unstable": group["individual_unstable"].mean(),
                "mean_individual_unstable_ratio": group["individual_unstable_ratio"].mean(),
                "mean_relational_unstable_ratio": group["relational_unstable_ratio"].mean(),
                "mean_total_unstable_ratio": group["total_unstable_ratio"].mean(),
                "total_subproblems": int(group["subproblems"].sum()),
                "mean_total_delta_unstable": group["total_delta_unstable"].mean(),
            }
        )

    rows: list[pd.Series] = []
    for (tool, perturbation_ratio), group in instance_df.groupby(["tool", "perturbation_ratio"], dropna=False, sort=True):
        summary = summarize_group(group)
        summary["tool"] = tool
        summary["perturbation_ratio"] = perturbation_ratio
        rows.append(summary)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    return table.sort_values(["tool", "perturbation_ratio"]).reset_index(drop=True)


def summarize_unstable(unstable_df: pd.DataFrame) -> pd.DataFrame:
    if unstable_df.empty:
        return unstable_df

    unstable_summary = (
        unstable_df.groupby(["tool", "perturbation_ratio", "layer_idx"], dropna=False)
        .agg(
            mean_delta_unstable=("delta_unstable", "mean"),
            std_delta_unstable=("delta_unstable", "std"),
            mean_inp1_unstable=("inp1_unstable", "mean"),
            mean_inp2_unstable=("inp2_unstable", "mean"),
            count=("delta_unstable", "count"),
        )
        .reset_index()
        .sort_values(["tool", "perturbation_ratio", "layer_idx"])
    )
    return unstable_summary


def summarize_first_unstable_ratio(summary_df: pd.DataFrame, result_root: Path) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    instance_key_cols = ["perturbation_ratio", "d", "e", "exe_id"]
    if "has_threshold_suffix" in summary_df.columns:
        instance_key_cols.append("has_threshold_suffix")
    unique_summary_df = (
        summary_df.sort_values(instance_key_cols + ["tool"])
        .drop_duplicates(subset=instance_key_cols, keep="first")
        .reset_index(drop=True)
    )

    records: list[dict[str, Any]] = []
    for row in unique_summary_df.to_dict(orient="records"):
        unstable_counts = load_first_unstable_counts_from_row(row, result_root)
        if not unstable_counts:
            continue

        individual_num = sum(unstable["inp1_unstable"] + unstable["inp2_unstable"] for unstable in unstable_counts)
        relational_num = sum(unstable["delta_unstable"] for unstable in unstable_counts)
        individual_total = sum(2 * unstable["total"] for unstable in unstable_counts)
        relational_total = sum(unstable["total"] for unstable in unstable_counts)

        records.append(
            {
                "perturbation_ratio": float(row["perturbation_ratio"]),
                "individual_unstable_num": individual_num,
                "individual_unstable_total": individual_total,
                "relational_unstable_num": relational_num,
                "relational_unstable_total": relational_total,
            }
        )

    if not records:
        return pd.DataFrame()

    ratio_df = pd.DataFrame(records)
    ratio_df = (
        ratio_df.groupby("perturbation_ratio", dropna=False)
        .agg(
            individual_unstable_num=("individual_unstable_num", "mean"),
            individual_unstable_total=("individual_unstable_total", "mean"),
            relational_unstable_num=("relational_unstable_num", "mean"),
            relational_unstable_total=("relational_unstable_total", "mean"),
        )
        .reset_index()
        .sort_values("perturbation_ratio", ascending=False)
        .reset_index(drop=True)
    )
    return ratio_df


def format_unstable_ratio_table(ratio_df: pd.DataFrame) -> str:
    if ratio_df.empty:
        return ""

    ratios = ratio_df["perturbation_ratio"].tolist()
    ratio_lookup = {
        float(row["perturbation_ratio"]): row for row in ratio_df.to_dict(orient="records")
    }

    lines: list[str] = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(r"\begin{tabular}{l" + ("c" * len(ratios)) + r"}")
    lines.append(r"\toprule")
    lines.append(r"& " + " & ".join(f"$p^{{\\%}}={ratio:g}$" for ratio in ratios) + r" \\")
    lines.append(r"\midrule")

    individual_cells = []
    for ratio in ratios:
        row = ratio_lookup[float(ratio)]
        individual_cells.append(f"{float(row['individual_unstable_num']):.1f} / {float(row['individual_unstable_total']):.1f}")

    lines.append(r"Unstable ReLUs & " + " & ".join(individual_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def print_table(table_df: pd.DataFrame) -> None:
    if table_df.empty:
        print("No rows to summarize.")
        return

    display_df = table_df.copy()
    float_cols = [
        "perturbation_ratio",
        "solved_rate",
        "mean_time_ratio_all",
        "mean_time_ratio_solved",
        "mean_time_either",
        "mean_subproblems_either",
        "mean_subproblems_all",
        "mean_subproblems_solved",
        "mean_individual_unstable",
        "mean_individual_unstable_ratio",
        "mean_relational_unstable_ratio",
        "mean_total_unstable_ratio",
    ]
    for col in float_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    print(display_df.to_string(index=False))


LATEX_METHOD_LABELS = {
    "IS_dual_ind": "ClasIS",
    "IS_dual": "DualIS",
    "RS_dual_Z": "SaBRe",
    "RS_random_Z": "SaBRe",
}

LATEX_METHOD_ORDER = ("IS_dual_ind", "IS_dual", "RS_random_Z", "RS_dual_Z")


def format_latex_table(table_df: pd.DataFrame, comparison_tools: tuple[str, ...]) -> str:
    if table_df.empty:
        return ""

    available_tools = [tool for tool in LATEX_METHOD_ORDER if tool in comparison_tools and tool in table_df["tool"].unique()]
    if not available_tools:
        return ""

    ratios = sorted(table_df["perturbation_ratio"].dropna().unique().tolist(), reverse=True)
    if not ratios:
        return ""

    lines: list[str] = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(r"\begin{tabular}{l" + ("ccc" * len(ratios)) + r"}")
    lines.append(r"\toprule")

    dataset_headers = {
        "acasxu": "ACAS Xu",
        "mnist4": "MNIST-F",
        "mnist-conv": "MNIST-C",
        "cifar10": "CIFAR",
        "gtsrb": "GTSRB",
    }
    dataset_order = ("acasxu", "mnist4", "mnist-conv", "cifar10", "gtsrb")

    ratio_headers = " & ".join(f"\\multicolumn{{3}}{{c}}{{$p^{{\\%}}={ratio:g}$}}" for ratio in ratios)
    lines.append(rf"\multirow{{2}}{{*}}{{Method}} & {ratio_headers} \\")
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + idx * 3}-{4 + idx * 3}}}" for idx in range(len(ratios))
    )
    lines.append(cmidrules)
    lines.append(r"& " + " & ".join(r"$s^{\#}$ & $p^{\#}$ & $\Delta T$" for _ in ratios) + r" \\")
    lines.append(r"\midrule")

    for tool in available_tools:
        tool_df = table_df[table_df["tool"] == tool].set_index("perturbation_ratio")
        row_cells: list[str] = []
        for ratio in ratios:
            if ratio not in tool_df.index:
                row_cells.extend(["-", "-", "-"])
                continue

            row = tool_df.loc[ratio]
            solved_instances = row.get("solved_instances", pd.NA)
            mean_subproblems_either = row.get("mean_subproblems_either", pd.NA)
            mean_time_either = row.get("mean_time_either", pd.NA)

            solved_text = "-" if pd.isna(solved_instances) else str(int(round(float(solved_instances))))
            subproblems_text = "-" if pd.isna(mean_subproblems_either) else f"{float(mean_subproblems_either):.1f}"
            time_text = "-" if pd.isna(mean_time_either) else f"{float(mean_time_either) * 100:.2f}"
            row_cells.extend([solved_text, subproblems_text, time_text])

        lines.append(f"{LATEX_METHOD_LABELS.get(tool, tool)} & " + " & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_latex_table_display_df(table_df: pd.DataFrame, comparison_tools: tuple[str, ...]) -> pd.DataFrame:
    if table_df.empty:
        return pd.DataFrame()

    available_tools = [tool for tool in LATEX_METHOD_ORDER if tool in comparison_tools and tool in table_df["tool"].unique()]
    if not available_tools:
        return pd.DataFrame()

    ratios = sorted(table_df["perturbation_ratio"].dropna().unique().tolist(), reverse=True)
    if not ratios:
        return pd.DataFrame()

    columns = pd.MultiIndex.from_tuples(
        [(f"p^%={ratio:g}", metric) for ratio in ratios for metric in ("s^#", "p^#", "Delta T")]
    )

    rows: list[list[Any]] = []
    index: list[str] = []
    for tool in available_tools:
        tool_df = table_df[table_df["tool"] == tool].set_index("perturbation_ratio")
        row_cells: list[Any] = []
        for ratio in ratios:
            if ratio not in tool_df.index:
                row_cells.extend(["-", "-", "-"])
                continue

            row = tool_df.loc[ratio]
            solved_instances = row.get("solved_instances", pd.NA)
            mean_subproblems_either = row.get("mean_subproblems_either", pd.NA)
            mean_time_either = row.get("mean_time_either", pd.NA)

            solved_text = "-" if pd.isna(solved_instances) else int(round(float(solved_instances)))
            subproblems_text = "-" if pd.isna(mean_subproblems_either) else f"{float(mean_subproblems_either):.1f}"
            time_text = "-" if pd.isna(mean_time_either) else f"{float(mean_time_either) * 100:.2f}"
            row_cells.extend([solved_text, subproblems_text, time_text])

        rows.append(row_cells)
        index.append(LATEX_METHOD_LABELS.get(tool, tool))

    display_df = pd.DataFrame(rows, columns=columns, index=index)
    display_df.index.name = "Method"
    return display_df


def build_unstable_ratio_display_df(ratio_df: pd.DataFrame) -> pd.DataFrame:
    if ratio_df.empty:
        return pd.DataFrame()

    ratios = ratio_df["perturbation_ratio"].tolist()
    ratio_lookup = {float(row["perturbation_ratio"]): row for row in ratio_df.to_dict(orient="records")}
    columns = pd.MultiIndex.from_tuples([(f"p^%={ratio:g}", "") for ratio in ratios])

    row_cells: list[Any] = []
    for ratio in ratios:
        row = ratio_lookup[float(ratio)]
        row_cells.append(f"{float(row['individual_unstable_num']):.1f} / {float(row['individual_unstable_total']):.1f}")

    display_df = pd.DataFrame([row_cells], index=["Unstable ReLUs"], columns=columns)
    display_df.index.name = "Metric"
    return display_df


def plot_unstable_transition(unstable_df: pd.DataFrame, output_path: Path) -> None:
    if unstable_df.empty:
        return

    tools = sorted(unstable_df["tool"].dropna().unique().tolist())
    fig, axes = plt.subplots(len(tools), 1, figsize=(10, 4 * len(tools)), sharex=True)
    if len(tools) == 1:
        axes = [axes]

    for ax, tool in zip(axes, tools):
        tool_df = unstable_df[unstable_df["tool"] == tool].copy()
        tool_df = tool_df.sort_values(["layer_idx", "perturbation_ratio"])
        ratios = sorted(tool_df["perturbation_ratio"].dropna().unique().tolist())
        layer_indices = sorted(tool_df["layer_idx"].dropna().unique().tolist())

        for layer_idx in layer_indices:
            layer_df = tool_df[tool_df["layer_idx"] == layer_idx]
            layer_df = layer_df.sort_values("perturbation_ratio")
            ax.plot(
                layer_df["perturbation_ratio"],
                layer_df["mean_delta_unstable"],
                marker="o",
                linewidth=1.5,
                label=f"Layer {int(layer_idx)}",
            )

        ax.set_title(tool)
        ax.set_xlabel("Perturbation ratio")
        ax.set_ylabel("Mean delta unstable ReLU")
        ax.set_xticks(ratios)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_unstable_gap(
    table_df: pd.DataFrame,
    output_path: Path,
    baseline_tool: str | None = None,
) -> None:
    if table_df.empty or "mean_total_delta_unstable" not in table_df.columns:
        return

    tools = sorted(table_df["tool"].dropna().unique().tolist())
    if not tools:
        return

    if baseline_tool is None:
        baseline_tool = "RS_dual_Z" if "RS_dual_Z" in tools else tools[0]
    if baseline_tool not in tools:
        baseline_tool = tools[0]

    baseline_df = table_df[table_df["tool"] == baseline_tool][["perturbation_ratio", "mean_total_delta_unstable"]].rename(
        columns={"mean_total_delta_unstable": "baseline_mean_total_delta_unstable"}
    )

    plot_df = table_df.merge(baseline_df, on="perturbation_ratio", how="inner")
    plot_df = plot_df[plot_df["tool"] != baseline_tool].copy()
    if plot_df.empty:
        return

    plot_df["gap"] = plot_df["mean_total_delta_unstable"] - plot_df["baseline_mean_total_delta_unstable"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for tool in sorted(plot_df["tool"].dropna().unique().tolist()):
        tool_df = plot_df[plot_df["tool"] == tool].sort_values("perturbation_ratio")
        ax.plot(
            tool_df["perturbation_ratio"],
            tool_df["gap"],
            marker="o",
            linewidth=1.8,
            label=f"{tool} - {baseline_tool}",
        )

    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Perturbation ratio")
    ax.set_ylabel("Gap in mean total unstable ReLUs")
    ax.set_title("Unstable-ReLU gap by perturbation ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_analysis(
    result_root: Path,
    output_root: Path,
    summary_csv: Path | None,
    keep_non_threshold_logs: bool,
    comparison_tools: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    summary_df_all = load_summary_df(result_root, keep_non_threshold_logs, summary_csv=summary_csv)
    summary_df = summary_df_all.copy()
    if comparison_tools is not None:
        summary_df = summary_df[summary_df["tool"].isin(comparison_tools)].copy()
    instance_df, unstable_df = build_analysis_records(summary_df, result_root=result_root, output_root=output_root)
    instance_df = annotate_either_solved(instance_df, comparison_tools)
    table_df = summarize_instances(instance_df)
    unstable_ratio_df = summarize_first_unstable_ratio(summary_df_all, result_root=result_root)
    latex_table = format_latex_table(table_df, comparison_tools or ())
    unstable_latex_table = format_unstable_ratio_table(unstable_ratio_df)

    return (
        build_latex_table_display_df(table_df, comparison_tools or ()),
        build_unstable_ratio_display_df(unstable_ratio_df),
        latex_table,
        unstable_latex_table,
    )


comparison_approaches_map = {
    0: ("IS_dual_ind", "IS_dual", "RS_random_Z", "RS_dual_Z"),
    1: ("IS_dual_ind", "IS_dual", "RS_dual_Z"),
}

def dp_tables() -> None:
    parser = argparse.ArgumentParser(description="Analyze DP experiment CSVs and logs.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--keep-non-threshold-logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include logs from folders without the _threshold suffix.",
    )
    parser.add_argument(
        "--comparison-approach-key",
        type=int,
        default=1,
        help=f"Select a comparison set from {sorted(comparison_approaches_map)}.",
    )
    args, _ = parser.parse_known_args()

    if args.comparison_approach_key not in comparison_approaches_map:
        available_keys = ", ".join(str(key) for key in sorted(comparison_approaches_map))
        raise ValueError(f"Unknown comparison approach key: {args.comparison_approach_key}. Available keys: {available_keys}")

    comparison_tools = comparison_approaches_map[args.comparison_approach_key]
    table_df, unstable_ratio_df, latex_table, unstable_latex_table = run_analysis(
        result_root=args.result_root,
        output_root=args.output_root,
        summary_csv=args.summary_csv,
        keep_non_threshold_logs=args.keep_non_threshold_logs,
        comparison_tools=comparison_tools,
    )

    print(f"Comparison tools selected by key {args.comparison_approach_key}: {', '.join(comparison_tools)}")
    if Markdown is not None:
        display(Markdown("```latex\n" + latex_table + "\n```"))
        display(Markdown("```latex\n" + unstable_latex_table + "\n```"))
    else:
        print(latex_table)
        print(unstable_latex_table)
    display(table_df)
    display(unstable_ratio_df)
