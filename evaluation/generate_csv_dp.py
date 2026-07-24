from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = SCRIPT_DIR.parent / "nnRelationalVerify" / "result" / "mnist-256x4-dp"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "mnist-256x4-dp"
DEFAULT_SUMMARY = SCRIPT_DIR / "mnist-256x4-dp" / "mnist-256x4-dp_summary.csv"
NUMBER_PATTERN = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|NaN|inf|-inf)"


PATH_PATTERN = re.compile(
    r".*/result/mnist-256x4-dp/"
    r"(?P<tool>.+?)_dimperturb_(?P<ratio>[\d.]+)"
    r"(?:_threshold)?/d(?P<d>\d+)_e(?P<e>\d+)/(?P<exe_id>\d+)/log\.md$"
)

BASE_RESULT_PATTERN = re.compile(
    r"## BASE Result\s*\n"
    rf"execution time:\s*IAR \+\s*RelationalAnalysis\s*=\s*(?P<iar>{NUMBER_PATTERN})\s*\+\s*(?P<rel>{NUMBER_PATTERN})\s*=\s*(?P<total>{NUMBER_PATTERN})\s*seconds\s*\n"
    r"status:\s*Status\.(?P<status>[A-Z_]+)\s*\n"
    r"(?:relational distance\s*\n)?"
    rf"(?:Output dim:\s*(?P<dim>\d+),\s*lower bound:\s*(?P<lb>{NUMBER_PATTERN}),\s*upper bound:\s*(?P<ub>{NUMBER_PATTERN}))?",
    re.MULTILINE,
)

FINAL_RESULT_PATTERN = re.compile(
    r"##\s+(?P<method>IS|RS)\s+Result\s*\n"
    r"status:\s*Status\.(?P<status>[A-Z_]+)\s*\n"
    rf"execution time:\s*\(base\)\s+\+\s+\((?P<split_kind>is|rs)\)\s*=\s*(?P<base_time>{NUMBER_PATTERN})\s*\+\s*(?P<split_time>{NUMBER_PATTERN})\s*=\s*(?P<total_time>{NUMBER_PATTERN})\s*seconds",
    re.MULTILINE,
)

UNSTABLE_BLOCK_PATTERN = re.compile(
    r"### Unstable ReLU Count \(Linear/Conv2D Layers\)\n(?P<body>(?:- layer_idx=.*\n)+)",
    re.MULTILINE,
)

UNSTABLE_LINE_PATTERN = re.compile(
    r"- layer_idx=(?P<layer_idx>\d+),\s*type=(?P<type>[^,]+),\s*total=(?P<total>\d+),\s*"
    r"inp1_unstable=(?P<inp1_unstable>\d+),\s*inp2_unstable=(?P<inp2_unstable>\d+),\s*"
    r"delta_unstable=(?P<delta_unstable>\d+)"
)

SPLIT_ROW_PATTERN = re.compile(
    rf"^(?P<name>[A-Za-z0-9_]+),\s+status:\s+Status\.(?P<status>[A-Z_]+),\s+split count:\s+(?P<split>\d+),\s+time:\s+(?P<time>{NUMBER_PATTERN})\s*\n"
    rf"Output dim:\s*(?P<dim>\d+),\s+lower bound:\s*(?P<lb>{NUMBER_PATTERN}),\s+upper bound:\s*(?P<ub>{NUMBER_PATTERN})",
    re.MULTILINE,
)

EXTRA_INFO_PATTERN = re.compile(
    r"Dataset:\s+Dataset\.(?P<dataset>\S+)\s*\n"
    r".*?Epsilon:\s+(?P<epsilon>[\d.]+)\s*\n"
    r".*?Delta epsilon:\s+(?P<delta_epsilon>[\d.]+)\s*\n"
    r".*?execution index:\s+\((?P<d>\d+),\s+(?P<e>\d+),\s+(?P<exe_id>\d+)\)\s*\n"
    r".*?Time budget:\s+(?P<time_budget>[\d.]+)\s+seconds\s*\n"
    r"(?:.*?Threshold:\s+(?P<threshold>[\d.]+)\s*\n)?",
    re.DOTALL,
)


def read_folder(folder_path: str | Path) -> list[Path]:
    folder_path = Path(folder_path)
    return sorted(path for path in folder_path.rglob("log.md") if path.is_file())


def extract_exp_info_from_path(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    match = PATH_PATTERN.match(path.as_posix())
    if not match:
        raise ValueError(f"Unexpected DP log path: {file_path}")

    info = match.groupdict()
    info["ratio"] = float(info["ratio"])
    info["d"] = int(info["d"])
    info["e"] = int(info["e"])
    info["exe_id"] = int(info["exe_id"])
    info["has_threshold_suffix"] = "_threshold" in path.parent.parent.parent.name
    return info


def _read_text(file_path: str | Path) -> str:
    return Path(file_path).read_text().replace("\r\n", "\n")


def _to_float(value: str) -> float:
    return float(value)


def extract_extra_info_from_log(log_text: str) -> dict[str, Any]:
    match = EXTRA_INFO_PATTERN.search(log_text)
    if not match:
        return {
            "dataset": None,
            "epsilon": None,
            "delta_epsilon": None,
            "execution_d": None,
            "execution_e": None,
            "execution_idx": None,
            "time_budget": None,
            "threshold": None,
        }

    info = match.groupdict()
    return {
        "dataset": info["dataset"],
        "epsilon": _to_float(info["epsilon"]),
        "delta_epsilon": _to_float(info["delta_epsilon"]),
        "execution_d": int(info["d"]),
        "execution_e": int(info["e"]),
        "execution_idx": int(info["exe_id"]),
        "time_budget": int(float(info["time_budget"])),
        "threshold": _to_float(info["threshold"]) if info.get("threshold") else None,
    }


def extract_base_result(log_text: str) -> dict[str, Any]:
    match = BASE_RESULT_PATTERN.search(log_text)
    if not match:
        raise ValueError("BASE Result block not found")

    info = match.groupdict()
    return {
        "base_iar_time": _to_float(info["iar"]),
        "base_relational_time": _to_float(info["rel"]),
        "base_total_time": _to_float(info["total"]),
        "base_status": info["status"],
        "base_output_dim": int(info["dim"]) if info["dim"] is not None else None,
        "base_lower_bound": _to_float(info["lb"]) if info["lb"] is not None else None,
        "base_upper_bound": _to_float(info["ub"]) if info["ub"] is not None else None,
    }


def extract_final_result(log_text: str) -> dict[str, Any]:
    match = FINAL_RESULT_PATTERN.search(log_text)
    if not match:
        return {
            "final_method": None,
            "final_status": None,
            "final_base_time": None,
            "final_split_time": None,
            "final_total_time": None,
        }

    info = match.groupdict()
    return {
        "final_method": info["method"],
        "final_status": info["status"],
        "final_base_time": _to_float(info["base_time"]),
        "final_split_time": _to_float(info["split_time"]),
        "final_total_time": _to_float(info["total_time"]),
    }


def extract_unstable_relu_counts(log_text: str) -> list[dict[str, Any]]:
    match = UNSTABLE_BLOCK_PATTERN.search(log_text)
    if not match:
        return []

    return [
        {
            "layer_idx": int(item.group("layer_idx")),
            "type": item.group("type"),
            "total": int(item.group("total")),
            "inp1_unstable": int(item.group("inp1_unstable")),
            "inp2_unstable": int(item.group("inp2_unstable")),
            "delta_unstable": int(item.group("delta_unstable")),
        }
        for item in UNSTABLE_LINE_PATTERN.finditer(match.group("body"))
    ]


def extract_split_rows(log_text: str) -> list[dict[str, Any]]:
    rows = []
    for match in SPLIT_ROW_PATTERN.finditer(log_text):
        info = match.groupdict()
        rows.append(
            {
                "name": info["name"],
                "status": info["status"],
                "level": int(info["split"]),
                "split": int(info["split"]),
                "time": _to_float(info["time"]),
                "dim": int(info["dim"]),
                "lb": _to_float(info["lb"]),
                "ub": _to_float(info["ub"]),
            }
        )
    return rows


def build_result_df(file_path: str | Path) -> pd.DataFrame:
    log_text = _read_text(file_path)
    path_info = extract_exp_info_from_path(file_path)
    extra_info = extract_extra_info_from_log(log_text)
    base_result = extract_base_result(log_text)
    final_result = extract_final_result(log_text)
    unstable_counts = extract_unstable_relu_counts(log_text)
    split_rows = extract_split_rows(log_text)

    metadata = {
        "tool": path_info["tool"],
        "perturbation_ratio": path_info["ratio"],
        "has_threshold_suffix": path_info["has_threshold_suffix"],
        "d": path_info["d"],
        "e": path_info["e"],
        "exe_id": path_info["exe_id"],
        **extra_info,
        **base_result,
        **final_result,
        "unstable_relu_counts": json.dumps(unstable_counts, ensure_ascii=False),
    }

    rows = []
    for row in split_rows:
        rows.append({**row, **metadata})

    if not rows:
        rows.append(
            {
                "name": "BASE",
                "status": base_result["base_status"],
                "level": 0,
                "split": 0,
                "time": base_result["base_total_time"],
                "dim": base_result["base_output_dim"],
                "lb": base_result["base_lower_bound"],
                "ub": base_result["base_upper_bound"],
                **metadata,
            }
        )

    column_order = [
        "name",
        "status",
        "level",
        "split",
        "time",
        "dim",
        "lb",
        "ub",
        "tool",
        "perturbation_ratio",
        "has_threshold_suffix",
        "d",
        "e",
        "exe_id",
        "dataset",
        "epsilon",
        "delta_epsilon",
        "execution_d",
        "execution_e",
        "execution_idx",
        "time_budget",
        "threshold",
        "base_iar_time",
        "base_relational_time",
        "base_total_time",
        "base_status",
        "base_output_dim",
        "base_lower_bound",
        "base_upper_bound",
        "final_method",
        "final_status",
        "final_base_time",
        "final_split_time",
        "final_total_time",
        "unstable_relu_counts",
    ]

    return pd.DataFrame(rows)[column_order]


def build_summary_row(file_path: str | Path) -> dict[str, Any]:
    log_text = _read_text(file_path)
    path_info = extract_exp_info_from_path(file_path)
    extra_info = extract_extra_info_from_log(log_text)
    base_result = extract_base_result(log_text)
    final_result = extract_final_result(log_text)
    unstable_counts = extract_unstable_relu_counts(log_text)

    summary: dict[str, Any] = {
        "tool": path_info["tool"],
        "perturbation_ratio": path_info["ratio"],
        "has_threshold_suffix": path_info["has_threshold_suffix"],
        "d": path_info["d"],
        "e": path_info["e"],
        "exe_id": path_info["exe_id"],
        **extra_info,
        **base_result,
        **final_result,
        "unstable_relu_counts": json.dumps(unstable_counts, ensure_ascii=False),
        "unstable_layer_count": len(unstable_counts),
    }

    for index, unstable in enumerate(unstable_counts):
        prefix = f"unstable_{index}"
        summary[f"{prefix}_layer_idx"] = unstable["layer_idx"]
        summary[f"{prefix}_type"] = unstable["type"]
        summary[f"{prefix}_total"] = unstable["total"]
        summary[f"{prefix}_inp1_unstable"] = unstable["inp1_unstable"]
        summary[f"{prefix}_inp2_unstable"] = unstable["inp2_unstable"]
        summary[f"{prefix}_delta_unstable"] = unstable["delta_unstable"]

    if not extract_split_rows(log_text):
        summary["base_only"] = True
    else:
        summary["base_only"] = False

    return summary


def export_result_csvs(result_root: str | Path = DEFAULT_RESULT_ROOT, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> list[Path]:
    result_root = Path(result_root)
    output_root = Path(output_root)
    written_paths: list[Path] = []

    for log_path in read_folder(result_root):
        try:
            df = build_result_df(log_path)
        except Exception as error:
            print(f"Skipping {log_path}: {error}")
            continue

        path_info = extract_exp_info_from_path(log_path)
        save_dir = output_root / f"{path_info['tool']}_dp{path_info['ratio']}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"d{path_info['d']}_e{path_info['e']}_{path_info['exe_id']}.csv"
        df.to_csv(save_path, index=False)
        written_paths.append(save_path)

    return written_paths


def export_summary_csv(result_root: str | Path = DEFAULT_RESULT_ROOT, save_path: str | Path | None = None) -> pd.DataFrame:
    rows = []
    for log_path in read_folder(result_root):
        try:
            summary_row = build_summary_row(log_path)
            rows.append(summary_row)
        except Exception as error:
            print(f"Skipping {log_path}: {error}")

    df = pd.DataFrame(rows)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


def export_summary_csv_with_filter(
    result_root: str | Path = DEFAULT_RESULT_ROOT,
    save_path: str | Path | None = None,
    keep_non_threshold_logs: bool = False,
) -> pd.DataFrame:
    rows = []
    for log_path in read_folder(result_root):
        try:
            summary_row = build_summary_row(log_path)
        except Exception as error:
            print(f"Skipping {log_path}: {error}")
            continue

        if not keep_non_threshold_logs and not summary_row["has_threshold_suffix"]:
            continue

        rows.append(summary_row)

    df = pd.DataFrame(rows)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse DP experiment logs into CSV files.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--keep-non-threshold-logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include logs from folders without the _threshold suffix in the summary CSV.",
    )
    args = parser.parse_args()

    export_result_csvs(args.result_root, args.output_root)
    if args.summary_csv is not None:
        export_summary_csv_with_filter(
            args.result_root,
            args.summary_csv,
            keep_non_threshold_logs=args.keep_non_threshold_logs,
        )


if __name__ == "__main__":
    main()
