"""
quality_checks.py
------------------
Data-quality diagnostics used to power the dashboard and to feed the
curation queue: missingness profiling, exact + near-duplicate detection,
and statistical anomaly detection on sensor readings.

Kept separate from the Streamlit UI so the logic is independently testable
and reusable (e.g. in a batch/Airflow context) rather than tied to widgets.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def normalize_equipment_id(raw_id: str) -> str:
    """
    Canonicalize an equipment id for duplicate/grouping purposes:
    trim whitespace, upper-case, and unify separators.
    This does NOT mutate the source column -- it's used to build a join key.
    """
    if not isinstance(raw_id, str):
        return raw_id
    cleaned = raw_id.strip().upper()
    cleaned = re.sub(r"[_\s]+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column missing-value count and percentage."""
    total = len(df)
    missing_count = df.isna().sum()
    missing_pct = (missing_count / total * 100).round(2)
    report = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_pct": missing_pct.values,
        }
    ).sort_values("missing_pct", ascending=False).reset_index(drop=True)
    return report


def find_exact_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """
    Flag rows that are exact duplicates on a given subset of columns
    (default: all columns except record_id, since record_id is always unique
    by construction and would mask true duplication).
    """
    if subset is None:
        subset = [c for c in df.columns if c != "record_id"]
    mask = df.duplicated(subset=subset, keep=False)
    return df.loc[mask].sort_values(subset)


def find_near_duplicates(
    df: pd.DataFrame,
    time_window_minutes: int = 10,
) -> pd.DataFrame:
    """
    Flag near-duplicate records: same normalized equipment id, same day,
    with timestamps within `time_window_minutes` of each other. This catches
    the common ingestion-bug pattern of the same physical event being logged
    twice with slightly different formatting or a few seconds/minutes apart.

    Returns the flagged rows plus a `dup_group_id` column so the UI can group
    them for side-by-side review.
    """
    working = df.copy()
    working["_norm_id"] = working["equipment_id"].apply(normalize_equipment_id)
    working = working.sort_values(["_norm_id", "timestamp"])

    group_id = 0
    group_ids = [np.nan] * len(working)
    idx_list = working.index.tolist()

    prev_norm_id = None
    prev_timestamp = None
    prev_positions: list[int] = []

    for pos, idx in enumerate(idx_list):
        norm_id = working.at[idx, "_norm_id"]
        ts = working.at[idx, "timestamp"]

        if (
            prev_norm_id == norm_id
            and prev_timestamp is not None
            and pd.notna(ts)
            and pd.notna(prev_timestamp)
            and abs((ts - prev_timestamp).total_seconds()) <= time_window_minutes * 60
        ):
            # Extends the current group.
            if not prev_positions or group_ids[prev_positions[-1]] is np.nan:
                group_id += 1
                for p in prev_positions[-1:]:
                    group_ids[p] = group_id
            group_ids[pos] = group_id
        prev_norm_id = norm_id
        prev_timestamp = ts
        prev_positions.append(pos)

    working["dup_group_id"] = group_ids
    flagged = working[working["dup_group_id"].notna()].copy()
    flagged = flagged.drop(columns=["_norm_id"])
    return flagged.sort_values(["dup_group_id", "timestamp"])


def detect_sensor_anomalies(
    df: pd.DataFrame,
    columns: tuple[str, ...] = ("temperature_c", "vibration_mm_s"),
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Flag statistical anomalies per equipment_type using a z-score on each
    numeric sensor column, computed within each equipment_type group (a
    Pump running hot is not the same as a Compressor running hot).

    Returns the original dataframe with an added boolean `is_anomaly` column
    and an `anomaly_reason` column listing which field(s) triggered it.
    """
    working = df.copy()
    working["is_anomaly"] = False
    working["anomaly_reason"] = ""

    for eq_type, group in working.groupby("equipment_type", dropna=True):
        for col in columns:
            values = group[col]
            mean = values.mean()
            std = values.std(ddof=0)
            if not std or np.isnan(std):
                continue
            z_scores = (values - mean) / std
            anomaly_idx = group.index[z_scores.abs() > z_threshold]
            working.loc[anomaly_idx, "is_anomaly"] = True
            working.loc[anomaly_idx, "anomaly_reason"] += f"{col} z>{z_threshold}; "

    working["anomaly_reason"] = working["anomaly_reason"].str.strip()
    return working


def data_health_summary(df: pd.DataFrame) -> dict:
    """
    Roll up the headline KPIs shown at the top of the dashboard:
    total records, duplication rate, average missingness, anomaly rate.
    """
    total = len(df)
    exact_dupes = find_exact_duplicates(df)
    near_dupes = find_near_duplicates(df)
    anomalies = detect_sensor_anomalies(df)
    missing_pct_avg = df.isna().mean().mean() * 100

    return {
        "total_records": total,
        "exact_duplicate_count": len(exact_dupes),
        "near_duplicate_count": len(near_dupes),
        "duplication_rate_pct": round(
            (len(exact_dupes) + len(near_dupes)) / total * 100, 2
        ) if total else 0,
        "avg_missing_pct": round(missing_pct_avg, 2),
        "anomaly_count": int(anomalies["is_anomaly"].sum()),
        "anomaly_rate_pct": round(anomalies["is_anomaly"].mean() * 100, 2) if total else 0,
    }
