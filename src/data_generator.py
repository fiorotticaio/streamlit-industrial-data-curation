"""
data_generator.py
------------------
Generates a synthetic dataset that mimics industrial IoT sensor readings and
maintenance-log text, in the style of the raw data a Data Foundry Engineer
would ingest before curation.

The generator deliberately injects the kinds of real-world messiness that
show up at scale (millions of samples/day): exact and near-duplicate
records, missing values, inconsistent casing/units, typos, and a handful of
out-of-range sensor anomalies. This lets the rest of the app demonstrate
detection + curation logic on data that behaves like production data,
rather than a toy clean dataset.
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Domain constants
# ----------------------------------------------------------------------

EQUIPMENT_TYPES = ["Pump", "Motor", "Compressor", "Conveyor", "Gearbox", "Fan"]
PLANTS = ["Plant-A", "Plant-B", "Plant-C", "Plant-D"]

# Raw, "as typed by a technician" issue descriptions. Intentionally noisy:
# mixed case, abbreviations, typos, extra whitespace, synonyms for the same
# underlying failure mode. This is the text the AI-suggestion module will
# later try to normalize/categorize.
RAW_ISSUE_TEMPLATES = [
    "temp high",
    "TEMP HIGH on bearing",
    "Temperature  high, needs check",
    "overheating issue reported by operator",
    "OVERHEAT - urgent",
    "vibration abnormal",
    "Vibration  ABNORMAL, checked twice",
    "excessive vibration on shaft",
    "vibr. high - possible imbalance",
    "leak detected near seal",
    "oil leak - minor",
    "LEAK, oil pooling on floor",
    "noise unusual during operation",
    "unusual noise, grinding sound",
    "strange noize from motor",  # typo intentional
    "electrical fault suspected",
    "elec fault - breaker tripped",
    "power fluctuation noted",
    "bearing wear detected",
    "bearing worn, replace soon",
    "no issue found",
    "routine check ok",
    "",  # will become NaN downstream
    "   ",  # whitespace-only, will become NaN downstream
]


def _random_timestamp(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between start and end (inclusive-ish)."""
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def _make_equipment_id(plant: str, eq_type: str, unit_number: int) -> str:
    """Build a canonical equipment id, e.g. 'Plant-A-PUMP-014'."""
    return f"{plant}-{eq_type.upper()}-{unit_number:03d}"


def _dirty_equipment_id(clean_id: str) -> str:
    """
    Randomly corrupt an equipment id the way upstream systems often do:
    inconsistent case, stray whitespace, or a trailing space.
    """
    variants = [
        clean_id,
        clean_id.lower(),
        f" {clean_id}",
        f"{clean_id} ",
        clean_id.replace("-", "_", 1),
    ]
    return random.choice(variants)


def generate_raw_dataset(
    n_records: int = 5000,
    duplicate_rate: float = 0.06,
    missing_rate: float = 0.08,
    anomaly_rate: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic, intentionally messy industrial dataset.

    Parameters
    ----------
    n_records: number of "clean" base records to generate before duplication.
    duplicate_rate: fraction of records that get an exact or near-duplicate
        counterpart appended (simulating retransmission / ingestion bugs).
    missing_rate: per-field probability of a value being dropped (NaN).
    anomaly_rate: fraction of records with sensor readings pushed outside
        physically plausible ranges (simulating faulty sensors).
    seed: random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns:
        record_id, equipment_id, equipment_type, plant, timestamp,
        temperature_c, vibration_mm_s, issue_description, status
    """
    random.seed(seed)
    np.random.seed(seed)

    start = datetime(2026, 1, 1)
    end = datetime(2026, 9, 1)

    rows = []
    for i in range(n_records):
        plant = random.choice(PLANTS)
        eq_type = random.choice(EQUIPMENT_TYPES)
        unit_number = random.randint(1, 40)
        clean_id = _make_equipment_id(plant, eq_type, unit_number)
        equipment_id = _dirty_equipment_id(clean_id)

        # Baseline "healthy" sensor distributions, per equipment type.
        base_temp = {
            "Pump": 55, "Motor": 65, "Compressor": 70,
            "Conveyor": 40, "Gearbox": 60, "Fan": 45,
        }[eq_type]
        base_vib = {
            "Pump": 2.5, "Motor": 3.0, "Compressor": 4.0,
            "Conveyor": 1.5, "Gearbox": 3.5, "Fan": 2.0,
        }[eq_type]

        temperature = float(np.random.normal(base_temp, 5))
        vibration = float(max(0.0, np.random.normal(base_vib, 0.8)))

        # Inject sensor anomalies (faulty readings).
        if random.random() < anomaly_rate:
            if random.random() < 0.5:
                temperature *= random.uniform(2.0, 3.5)   # runaway heat
            else:
                vibration *= random.uniform(3.0, 6.0)     # runaway vibration

        issue = random.choice(RAW_ISSUE_TEMPLATES)
        timestamp = _random_timestamp(start, end)

        rows.append(
            {
                "record_id": f"REC-{i:06d}",
                "equipment_id": equipment_id,
                "equipment_type": eq_type,
                "plant": plant,
                "timestamp": timestamp,
                "temperature_c": round(temperature, 2),
                "vibration_mm_s": round(vibration, 3),
                "issue_description": issue,
                "status": random.choice(["open", "closed", "in_review"]),
            }
        )

    df = pd.DataFrame(rows)

    # --- Inject duplicate / near-duplicate records -----------------------
    n_dupes = int(n_records * duplicate_rate)
    dupe_sample = df.sample(n=n_dupes, random_state=seed).copy()

    def _mutate_slightly(row: pd.Series) -> pd.Series:
        """Near-duplicate: same event, tiny formatting drift + a few
        seconds/minutes of timestamp jitter, as if re-ingested."""
        row = row.copy()
        row["record_id"] = row["record_id"] + "-DUP" + "".join(
            random.choices(string.digits, k=2)
        )
        row["equipment_id"] = _dirty_equipment_id(
            row["equipment_id"].strip().replace("_", "-")
        )
        row["timestamp"] = row["timestamp"] + timedelta(
            seconds=random.randint(1, 300)
        )
        if isinstance(row["issue_description"], str) and row["issue_description"]:
            row["issue_description"] = row["issue_description"].upper()
        return row

    dupes = dupe_sample.apply(_mutate_slightly, axis=1)
    df = pd.concat([df, dupes], ignore_index=True)

    # --- Inject missing values -------------------------------------------
    for col in ["temperature_c", "vibration_mm_s", "issue_description", "equipment_type"]:
        mask = np.random.rand(len(df)) < missing_rate
        df.loc[mask, col] = np.nan

    # Whitespace-only descriptions should also read as "missing" text.
    df["issue_description"] = df["issue_description"].apply(
        lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
    )

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df
