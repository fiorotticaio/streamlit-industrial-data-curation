"""
app.py
------
Internal Data Quality & Curation Tool (portfolio project).

A Streamlit prototype for a "Data Foundry Engineer"-style workflow: ingest
noisy industrial sensor/maintenance data, surface data-quality issues
(missingness, duplication, anomalies), and provide a human-in-the-loop
interface to review AI-generated normalization/categorization suggestions
before they're accepted into a "curated" dataset.

Run with:
    streamlit run app.py

Layout:
    1) Sidebar        - dataset generation controls
    2) Data Health     - KPI dashboard + charts
    3) Duplicate Review - group & resolve exact/near-duplicate records
    4) AI Curation Queue - accept/reject/override AI suggestions on
                            missing/noisy issue descriptions
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ai_suggestions import suggest_batch
from src.data_generator import generate_raw_dataset
from src.quality_checks import (
    data_health_summary,
    detect_sensor_anomalies,
    find_exact_duplicates,
    find_near_duplicates,
    missingness_report,
)

st.set_page_config(
    page_title="Industrial Data Curation & Quality Tool",
    page_icon="🛠️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Session state helpers
# --------------------------------------------------------------------------

def _init_state() -> None:
    """Ensure all session_state keys exist before first use."""
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = generate_raw_dataset()
    if "decisions" not in st.session_state:
        # record_id -> {"decision": "accepted"/"rejected"/"overridden",
        #               "final_category": str, "final_text": str}
        st.session_state.decisions = {}
    if "dup_resolutions" not in st.session_state:
        # dup_group_id -> record_id kept as the "primary" record
        st.session_state.dup_resolutions = {}


def _regenerate(n_records: int, dup_rate: float, missing_rate: float,
                 anomaly_rate: float, seed: int) -> None:
    st.session_state.raw_df = generate_raw_dataset(
        n_records=n_records,
        duplicate_rate=dup_rate,
        missing_rate=missing_rate,
        anomaly_rate=anomaly_rate,
        seed=seed,
    )
    st.session_state.decisions = {}
    st.session_state.dup_resolutions = {}


_init_state()

# --------------------------------------------------------------------------
# Sidebar - data generation controls
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Dataset Controls")
    st.caption(
        "Simulates a raw ingestion batch from industrial IoT sensors "
        "and technician maintenance logs."
    )
    n_records = st.slider("Base record count", 500, 20000, 5000, step=500)
    dup_rate = st.slider("Duplicate injection rate", 0.0, 0.3, 0.06, step=0.01)
    missing_rate = st.slider("Missing-value rate (per field)", 0.0, 0.3, 0.08, step=0.01)
    anomaly_rate = st.slider("Sensor anomaly rate", 0.0, 0.15, 0.03, step=0.01)
    seed = st.number_input("Random seed", value=42, step=1)

    if st.button("🔄 Regenerate dataset", use_container_width=True):
        _regenerate(n_records, dup_rate, missing_rate, anomaly_rate, seed)
        st.rerun()

    st.divider()
    st.caption(
        "**Tractian Data Foundry Engineer — portfolio demo**\n\n"
        "Showcases duplicate detection, missingness/anomaly profiling, "
        "and an AI-assisted human-in-the-loop curation queue."
    )

raw_df = st.session_state.raw_df

# --------------------------------------------------------------------------
# Page header
# --------------------------------------------------------------------------

st.title("🛠️ Industrial Data Curation & Quality Tool")
st.caption(
    "Prototype internal tool for reviewing and curating raw industrial "
    "sensor/maintenance data before it flows downstream to analytics or ML."
)

tab_dashboard, tab_dupes, tab_curation = st.tabs(
    ["📊 Data Health", "🧬 Duplicate Review", "🤖 AI Curation Queue"]
)

# ==========================================================================
# TAB 1 — DATA HEALTH DASHBOARD
# ==========================================================================
with tab_dashboard:
    summary = data_health_summary(raw_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total records", f"{summary['total_records']:,}")
    col2.metric("Duplication rate", f"{summary['duplication_rate_pct']}%")
    col3.metric("Avg. missingness", f"{summary['avg_missing_pct']}%")
    col4.metric("Anomaly rate", f"{summary['anomaly_rate_pct']}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Missing values by column")
        miss_report = missingness_report(raw_df)
        fig_missing = px.bar(
            miss_report,
            x="column",
            y="missing_pct",
            text="missing_pct",
            labels={"missing_pct": "% missing", "column": "Column"},
        )
        fig_missing.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_missing.update_layout(yaxis_range=[0, max(10, miss_report["missing_pct"].max() * 1.2)])
        st.plotly_chart(fig_missing, use_container_width=True)

    with right:
        st.subheader("Duplicate breakdown")
        dup_counts = pd.DataFrame(
            {
                "type": ["Exact duplicates", "Near-duplicates (fuzzy)"],
                "count": [summary["exact_duplicate_count"], summary["near_duplicate_count"]],
            }
        )
        fig_dup = px.bar(dup_counts, x="type", y="count", text="count", color="type")
        fig_dup.update_traces(textposition="outside")
        fig_dup.update_layout(showlegend=False)
        st.plotly_chart(fig_dup, use_container_width=True)

    st.divider()

    st.subheader("Sensor anomaly detection (z-score, per equipment type)")
    anomaly_df = detect_sensor_anomalies(raw_df)
    metric_choice = st.radio(
        "Sensor metric", ["temperature_c", "vibration_mm_s"], horizontal=True
    )
    fig_scatter = px.scatter(
        anomaly_df,
        x="timestamp",
        y=metric_choice,
        color="is_anomaly",
        facet_col="equipment_type",
        facet_col_wrap=3,
        color_discrete_map={True: "#e74c3c", False: "#3498db"},
        opacity=0.6,
        height=550,
        labels={"is_anomaly": "Anomaly"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        f"{summary['anomaly_count']:,} readings flagged as statistical "
        "outliers (|z-score| > 3) within their equipment-type baseline."
    )

    with st.expander("View raw dataset sample"):
        st.dataframe(raw_df.head(200), use_container_width=True, height=300)

# ==========================================================================
# TAB 2 — DUPLICATE REVIEW
# ==========================================================================
with tab_dupes:
    st.subheader("Near-duplicate groups")
    st.caption(
        "Groups records sharing a normalized equipment ID with timestamps "
        "within a configurable window — the classic 'same event logged "
        "twice' ingestion bug. Pick which record to keep as the system of "
        "record for each group."
    )

    window = st.slider("Time window for grouping (minutes)", 1, 60, 10)
    near_dupes = find_near_duplicates(raw_df, time_window_minutes=window)

    if near_dupes.empty:
        st.info("No near-duplicate groups found at this time window.")
    else:
        group_ids = sorted(near_dupes["dup_group_id"].unique())
        st.write(f"**{len(group_ids)} duplicate group(s) found** "
                 f"across {len(near_dupes)} records.")

        selected_group = st.selectbox(
            "Select a group to review",
            group_ids,
            format_func=lambda g: f"Group {int(g)}",
        )

        group_rows = near_dupes[near_dupes["dup_group_id"] == selected_group]
        display_cols = [
            "record_id", "equipment_id", "equipment_type", "plant",
            "timestamp", "temperature_c", "vibration_mm_s",
            "issue_description", "status",
        ]
        st.dataframe(group_rows[display_cols], use_container_width=True)

        keep_choice = st.radio(
            "Keep which record as the primary (others will be marked as merged)?",
            group_rows["record_id"].tolist(),
            key=f"keep_{selected_group}",
        )
        if st.button("✅ Resolve this group", key=f"resolve_{selected_group}"):
            st.session_state.dup_resolutions[int(selected_group)] = keep_choice
            st.success(f"Group {int(selected_group)} resolved — keeping {keep_choice}.")

        if st.session_state.dup_resolutions:
            st.divider()
            st.subheader("Resolved groups so far")
            resolved_df = pd.DataFrame(
                [
                    {"dup_group_id": g, "kept_record_id": rid}
                    for g, rid in st.session_state.dup_resolutions.items()
                ]
            )
            st.dataframe(resolved_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Exact duplicates")
    exact_dupes = find_exact_duplicates(raw_df)
    st.caption(
        f"{len(exact_dupes)} rows are byte-for-byte duplicates on every "
        "field except record_id — safe to auto-drop with a `drop_duplicates`."
    )
    if not exact_dupes.empty:
        st.dataframe(exact_dupes.head(100), use_container_width=True, height=250)

# ==========================================================================
# TAB 3 — AI CURATION QUEUE
# ==========================================================================
with tab_curation:
    st.subheader("Records needing review")
    st.caption(
        "Flags records with a missing or low-signal issue description, runs "
        "a simulated AI suggestion pass (category + normalized text + "
        "confidence), and lets a reviewer accept, reject, or manually "
        "override each suggestion before it's written to the curated table."
    )

    needs_review = raw_df[
        raw_df["issue_description"].isna()
        | raw_df["issue_description"].str.len().fillna(0).lt(4)
    ].copy()

    filter_status = st.multiselect(
        "Filter by status", options=sorted(raw_df["status"].dropna().unique()),
        default=None,
    )
    if filter_status:
        needs_review = needs_review[needs_review["status"].isin(filter_status)]

    st.write(f"**{len(needs_review)} record(s)** flagged for curation.")

    # Also surface a broader "noisy text" sample so reviewers can see the
    # AI engine work on records that DO have text, not just missing ones.
    show_noisy_sample = st.checkbox(
        "Also include a sample of noisy (non-missing) descriptions", value=True
    )
    if show_noisy_sample:
        noisy_sample = raw_df[raw_df["issue_description"].notna()].sample(
            n=min(30, raw_df["issue_description"].notna().sum()), random_state=1
        )
        review_pool = pd.concat([needs_review, noisy_sample]).drop_duplicates(
            subset="record_id"
        )
    else:
        review_pool = needs_review

    if review_pool.empty:
        st.info("Nothing to review with the current filters.")
    else:
        suggestions = suggest_batch(review_pool["issue_description"])
        review_pool = review_pool.reset_index(drop=True)
        review_pool["ai_category"] = [s.category for s in suggestions]
        review_pool["ai_normalized_text"] = [s.normalized_text for s in suggestions]
        review_pool["ai_confidence"] = [s.confidence for s in suggestions]

        # Pull in any decision already made in a prior render, so edits persist.
        review_pool["human_decision"] = review_pool["record_id"].map(
            lambda rid: st.session_state.decisions.get(rid, {}).get("decision", "pending")
        )
        review_pool["final_category"] = review_pool.apply(
            lambda r: st.session_state.decisions.get(r["record_id"], {}).get(
                "final_category", r["ai_category"]
            ),
            axis=1,
        )
        review_pool["final_text"] = review_pool.apply(
            lambda r: st.session_state.decisions.get(r["record_id"], {}).get(
                "final_text", r["ai_normalized_text"]
            ),
            axis=1,
        )

        editable_cols = [
            "record_id", "equipment_id", "issue_description",
            "ai_category", "ai_normalized_text", "ai_confidence",
            "human_decision", "final_category", "final_text",
        ]

        st.markdown("##### Review & edit suggestions")
        edited = st.data_editor(
            review_pool[editable_cols],
            key="curation_editor",
            use_container_width=True,
            height=420,
            hide_index=True,
            column_config={
                "record_id": st.column_config.TextColumn(disabled=True),
                "equipment_id": st.column_config.TextColumn(disabled=True),
                "issue_description": st.column_config.TextColumn(
                    "raw text", disabled=True
                ),
                "ai_category": st.column_config.TextColumn(disabled=True),
                "ai_normalized_text": st.column_config.TextColumn(disabled=True),
                "ai_confidence": st.column_config.ProgressColumn(
                    "AI confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "human_decision": st.column_config.SelectboxColumn(
                    options=["pending", "accepted", "rejected", "overridden"],
                    required=True,
                ),
                "final_category": st.column_config.SelectboxColumn(
                    options=sorted(
                        set(review_pool["ai_category"]) | {
                            "Overheating", "Vibration Anomaly", "Oil Leak",
                            "Unusual Noise", "Electrical Fault", "Bearing Wear",
                            "No Issue Found", "Unknown",
                        }
                    ),
                ),
                "final_text": st.column_config.TextColumn("final normalized text"),
            },
        )

        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            if st.button("💾 Save decisions", type="primary"):
                for _, row in edited.iterrows():
                    st.session_state.decisions[row["record_id"]] = {
                        "decision": row["human_decision"],
                        "final_category": row["final_category"],
                        "final_text": row["final_text"],
                    }
                st.success(f"Saved {len(edited)} decision(s) to the curation log.")
        with col_b:
            if st.button("✅ Bulk-accept high-confidence (≥0.85)"):
                for _, row in edited.iterrows():
                    if row["ai_confidence"] >= 0.85:
                        st.session_state.decisions[row["record_id"]] = {
                            "decision": "accepted",
                            "final_category": row["ai_category"],
                            "final_text": row["ai_normalized_text"],
                        }
                st.success("Bulk-accepted all high-confidence suggestions.")
                st.rerun()

        st.divider()
        st.subheader("Curation log")
        if st.session_state.decisions:
            log_df = pd.DataFrame(
                [
                    {"record_id": rid, **info}
                    for rid, info in st.session_state.decisions.items()
                ]
            )
            decision_counts = log_df["decision"].value_counts()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total reviewed", len(log_df))
            m2.metric("Accepted", int(decision_counts.get("accepted", 0)))
            m3.metric("Rejected", int(decision_counts.get("rejected", 0)))
            m4.metric("Overridden", int(decision_counts.get("overridden", 0)))
            st.dataframe(log_df, use_container_width=True, hide_index=True)

            csv = log_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export curation log as CSV", csv,
                file_name="curation_log.csv", mime="text/csv",
            )
        else:
            st.info("No decisions saved yet — review the table above and click 'Save decisions'.")
