# -*- coding: utf-8 -*-
"""
Updated on Mon Jun 29 18:10 2026

@author: Alex

Import Prolific/social-continuity bandit experiment data.

Supports both:
  1. the older CSV export format with an experimentData JSON column, and
  2. the newer raw JSON/.bin output 

The newer script records one automatic reveal at trial/click 0 for each bandit,
then participant clicks numbered 1..50. It also records RT/clickTimestamp and
separate tutorial arrays.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


MAIN_LIST_FIELDS = {
    "trial", "block", "env", "demoType", "condOrder", "scaleValue",
    "choice", "choice_x", "choice_y", "score", "score_scaled", "RT",
    "clickTimestamp",
}

TUTORIAL_FIELDS = {
    "tutorialTrial", "tutorialPlanet", "tutorialChoice_x", "tutorialChoice_y",
    "tutorialScore", "tutorialRT", "tutorialClickTimestamp",
}

ALIASES = {
    # legacy script names -> newer column names
    "tColl": "trial",
    "blockColl": "block",
    "envColl": "env",
    "scaleColl": "scaleValue",
    "choices": "choice",
    "xcollect": "choice_x",
    "ycollect": "choice_y",
    "zcollect": "score",
    "zrescaled": "score_scaled",
    "demoTypeColl": "demoType",
    "rScore": "score",
}


def _loads_maybe_nested_json(value: Any) -> Dict[str, Any]:
    """Load JSON strings, including database rows whose experimentData is JSON."""
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        raise ValueError("Missing experimentData JSON")
    obj = json.loads(value) if isinstance(value, str) else value
    if isinstance(obj, dict) and "experimentData" in obj:
        nested = obj["experimentData"]
        obj = json.loads(nested) if isinstance(nested, str) else nested
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
    return obj


def _read_records(input_path: Path) -> pd.DataFrame:
    """Return a dataframe with at least experimentData; supports CSV, JSON, JSONL, .bin."""
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(input_path)
        if "experimentData" not in df.columns:
            # Legacy local export sometimes had no header and exactly four columns.
            if len(df.columns) == 4:
                df.columns = ["id", "workerID", "experimentData", "reward"]
            else:
                raise ValueError("CSV must contain an experimentData column")
        return df

    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{input_path} is empty")

    # Try a single JSON object/array first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return pd.DataFrame(parsed)
        if isinstance(parsed, dict):
            if "experimentData" in parsed or "trial" in parsed or "tutorialTrial" in parsed:
                return pd.DataFrame([parsed])
            # Common wrapper: {rows: [...]} or {data: [...]}.
            for key in ("rows", "data", "results"):
                if isinstance(parsed.get(key), list):
                    return pd.DataFrame(parsed[key])
            return pd.DataFrame([{"experimentData": json.dumps(parsed)}])
    except json.JSONDecodeError:
        pass

    # Fallback: JSON lines.
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return pd.DataFrame(rows)


def _rename_aliases(data: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(data)
    for old, new in ALIASES.items():
        if old in data and new not in data:
            data[new] = data[old]
    return data


def _list_len(data: Dict[str, Any], keys: Iterable[str]) -> int:
    lengths = [len(data[k]) for k in keys if isinstance(data.get(k), list)]
    return max(lengths) if lengths else 0


def _expand_scalar(value: Any, n: int) -> List[Any]:
    if isinstance(value, list):
        if len(value) == n:
            return value
        if len(value) == 1:
            return value * n
        if len(value) == 0:
            return [np.nan] * n
        # Keep malformed records importable while flagging mismatch via NaN padding/truncation.
        return (value + [np.nan] * n)[:n]
    return [value] * n


def _main_dataframe(part_data: Dict[str, Any], meta: Dict[str, Any], part_index: int) -> pd.DataFrame:
    part_data = _rename_aliases(part_data)
    n = _list_len(part_data, MAIN_LIST_FIELDS)
    if n == 0:
        return pd.DataFrame()

    # Include all scalar metadata and all main-length arrays; exclude tutorial arrays from main rows.
    out: Dict[str, List[Any]] = {}
    for key, value in part_data.items():
        if key in TUTORIAL_FIELDS:
            continue
        if isinstance(value, list):
            # Include list fields that correspond to the main task. Empty condOrder is tolerated.
            if len(value) in (0, 1, n) or key in MAIN_LIST_FIELDS:
                out[key] = _expand_scalar(value, n)
        else:
            out[key] = [value] * n

    for key, value in meta.items():
        out[key] = [value] * n

    out["participant_index"] = [part_index] * n
    if "reward" not in out and "reward" in meta:
        out["reward"] = [meta["reward"]] * n
    if "completionTime" not in out and {"startTime", "endTime"}.issubset(part_data):
        out["completionTime"] = [(part_data["endTime"] - part_data["startTime"]) / 60000] * n
    if "tutorialCompletionTime" not in out and {"tutorialStartTime", "startTime"}.issubset(part_data):
        out["tutorialCompletionTime"] = [(part_data["startTime"] - part_data["tutorialStartTime"]) / 60000] * n

    return pd.DataFrame(out)


def _tutorial_dataframe(part_data: Dict[str, Any], meta: Dict[str, Any], part_index: int) -> pd.DataFrame:
    n = _list_len(part_data, TUTORIAL_FIELDS)
    if n == 0:
        return pd.DataFrame()

    out: Dict[str, List[Any]] = {}
    for key in TUTORIAL_FIELDS:
        if key in part_data:
            out[key] = _expand_scalar(part_data[key], n)

    for key in ("startTime", "endTime", "tutorialStartTime", "age", "gender", "processDescription", "feedback", "compFail", "reward"):
        if key in part_data:
            out[key] = [part_data[key]] * n
    for key, value in meta.items():
        out[key] = [value] * n

    out["participant_index"] = [part_index] * n
    if {"tutorialStartTime", "startTime"}.issubset(part_data):
        out["tutorialCompletionTime"] = [(part_data["startTime"] - part_data["tutorialStartTime"]) / 60000] * n

    df = pd.DataFrame(out)
    df = df.rename(columns={
        "tutorialTrial": "trial",
        "tutorialPlanet": "block",
        "tutorialChoice_x": "choice_x",
        "tutorialChoice_y": "choice_y",
        "tutorialScore": "score",
        "tutorialRT": "RT",
        "tutorialClickTimestamp": "clickTimestamp",
    })
    df["phase"] = "tutorial"
    return df


def _add_search_dist(df: pd.DataFrame, group_cols: List[str], x_col: str = "choice_x", y_col: str = "choice_y") -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()

    sort_cols = [c for c in group_cols + ["trial"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).copy()

    if x_col in df.columns and y_col in df.columns:
        df["prev_choice_x"] = df.groupby(group_cols)[x_col].shift(1) if all(c in df.columns for c in group_cols) else df[x_col].shift(1)
        df["prev_choice_y"] = df.groupby(group_cols)[y_col].shift(1) if all(c in df.columns for c in group_cols) else df[y_col].shift(1)
        dx = pd.to_numeric(df[x_col], errors="coerce") - pd.to_numeric(df["prev_choice_x"], errors="coerce")
        dy = pd.to_numeric(df[y_col], errors="coerce") - pd.to_numeric(df["prev_choice_y"], errors="coerce")
        df["search_dist"] = np.sqrt(dx.pow(2) + dy.pow(2))
    elif "choice" in df.columns:
        # Legacy fallback when only a flattened cell index exists.
        choices = pd.to_numeric(df["choice"], errors="coerce")
        max_choice = choices.max(skipna=True)
        width = int(math.sqrt(max_choice + 1)) if pd.notna(max_choice) else 11
        width = width if width > 0 else 11
        prev = df.groupby(group_cols)["choice"].shift(1) if all(c in df.columns for c in group_cols) else df["choice"].shift(1)
        curr_idx = pd.to_numeric(df["choice"], errors="coerce")
        prev_idx = pd.to_numeric(prev, errors="coerce")
        curr_x, curr_y = curr_idx % width, curr_idx // width
        prev_x, prev_y = prev_idx % width, prev_idx // width
        df["prev_choice"] = prev
        df["search_dist"] = np.sqrt((curr_x - prev_x).pow(2) + (curr_y - prev_y).pow(2))
    else:
        df["search_dist"] = np.nan

    return df


def import_experiment(input_path: Path, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = _read_records(input_path)
    main_parts: List[pd.DataFrame] = []
    tutorial_parts: List[pd.DataFrame] = []
    payment_rows: List[Dict[str, Any]] = []

    for part_index, row in records.reset_index(drop=True).iterrows():
        if "experimentData" in row and pd.notna(row["experimentData"]):
            part_data = _loads_maybe_nested_json(row["experimentData"])
        else:
            part_data = _loads_maybe_nested_json(row.dropna().to_dict())
        part_data = _rename_aliases(part_data)

        meta = {}
        for key in ("id", "workerID", "PROLIFIC_PID", "STUDY_ID", "SESSION_ID"):
            if key in row and pd.notna(row[key]):
                meta[key] = row[key]
        # Prefer server/payment-column reward if present, otherwise JSON reward.
        if "reward" in row and pd.notna(row["reward"]):
            meta["reward"] = row["reward"]

        main_parts.append(_main_dataframe(part_data, meta, part_index))
        tutorial_parts.append(_tutorial_dataframe(part_data, meta, part_index))

        payment_rows.append({
            "workerID": meta.get("workerID", meta.get("PROLIFIC_PID", part_index)),
            "reward": meta.get("reward", part_data.get("reward", np.nan)),
        })

    main_df = pd.concat([p for p in main_parts if not p.empty], ignore_index=True) if any(not p.empty for p in main_parts) else pd.DataFrame()
    tutorial_df = pd.concat([p for p in tutorial_parts if not p.empty], ignore_index=True) if any(not p.empty for p in tutorial_parts) else pd.DataFrame()

    # New script records block 0..9 and trial 0..50. Grouping by block/participant makes
    # the trial-0 automatic reveal the previous sample for trial 1, but prevents distances
    # from leaking across bandits.
    main_group_cols = [c for c in ["block", "participant_index"] if c in main_df.columns]
    if main_group_cols:
        main_df = _add_search_dist(main_df, main_group_cols)
    elif not main_df.empty:
        main_df = _add_search_dist(main_df, ["participant_index"])

    tutorial_group_cols = [c for c in ["block", "participant_index"] if c in tutorial_df.columns]
    if tutorial_group_cols:
        tutorial_df = _add_search_dist(tutorial_df, tutorial_group_cols)

    payment_df = pd.DataFrame(payment_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    main_df.to_csv(output_dir / f"{input_path.stem}.csv", index=False)
    if not tutorial_df.empty:
        tutorial_df.to_csv(output_dir / f"{input_path.stem}_tutorial.csv", index=False)
    payment_df.to_csv(output_dir / f"{input_path.stem}_bonus.txt", sep=",", index=False)

    return main_df, tutorial_df, payment_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Import social-continuity bandit Prolific data")
    parser.add_argument("input", nargs="?", default="../data/pilot_asocial/raw/beta_test_2026.06.29.bin", help="CSV, JSON, JSONL, or .bin input file")
    parser.add_argument("--outdir", default="../data/pilot_asocial/csv", help="Directory for output CSV files")
    args = parser.parse_args()
    filename = Path(args.input).stem
    main_df, tutorial_df, payment_df = import_experiment(Path(args.input), Path(args.outdir))
    print(f"Wrote {len(main_df)} main rows to {Path(args.outdir) / filename}.csv")
    if not tutorial_df.empty:
        print(f"Wrote {len(tutorial_df)} tutorial rows to {Path(args.outdir) / filename}_tutorial.csv")
    print(f"Wrote {len(payment_df)} payment rows to {Path(args.outdir) / filename}_bonus.txt")


if __name__ == "__main__":
    main()
