import os
from typing import Sequence

import pandas as pd
import altair as alt
import streamlit as st
from sqlalchemy import create_engine, text, bindparam, String
from sqlalchemy.dialects.postgresql import ARRAY
from dotenv import load_dotenv, find_dotenv

# --- Config ---
load_dotenv(find_dotenv(), override=False)

DB_URL = os.environ.get("MONITOR_DB_URL") or os.environ.get("APP_DB_URL")
if not DB_URL:
    st.stop()

st.set_page_config(page_title="Toxic Moderation – Monitoring", layout="wide")

# --- Helpers ---
@st.cache_resource(show_spinner=False)
def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True, future=True)

def _window_to_interval(window: str) -> str:
    mapping = {
        "Last 1 hour": "1 hour",
        "Last 24 hours": "24 hours",
        "Last 7 days": "7 days",
        "Last 30 days": "30 days",
        "All time": None,
    }
    return mapping.get(window, "24 hours")


@st.cache_data(ttl=30, show_spinner=False)
def get_versions():
    with get_engine().connect() as conn:
        df = pd.read_sql_query(
            "SELECT DISTINCT model_version FROM prediction_logs ORDER BY 1", conn
        )
    return df["model_version"].astype(str).tolist()


def _version_filter_sql(selected: Sequence[str]) -> tuple[str, dict, dict]:
    """Return (sql_snippet, plain_params, typed_binds)."""
    if not selected:
        return "", {}, {}
    typed = {"vlist": bindparam("vlist", type_=ARRAY(String()))}
    return " AND model_version = ANY(:vlist) ", {"vlist": list(map(str, selected))}, typed


@st.cache_data(ttl=15, show_spinner=False)
def fetch_latency(window_label: str, versions: Sequence[str]) -> pd.DataFrame:
    interval = _window_to_interval(window_label)
    ver_sql, params, typed = _version_filter_sql(versions)

    base_sql = """
        SELECT created_at, latency_ms, model_version
        FROM prediction_logs
        WHERE 1=1 {ver}
        {interval}
        ORDER BY created_at
    """
    if interval:
        base_sql = base_sql.format(ver=ver_sql, interval="AND created_at >= now() - interval :ival")
        params = {**params, "ival": interval}
    else:
        base_sql = base_sql.format(ver=ver_sql, interval="")

    with get_engine().connect() as conn:
        stmt = text(base_sql).bindparams(**typed)
        df = pd.read_sql_query(stmt, conn, params=params)

    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert("UTC")

    # If very few points, bypass resample so the chart isn't empty
    if len(df) < 5:
        df = df.sort_values("created_at")
        df = df.assign(
            p50=df["latency_ms"],
            p90=df["latency_ms"],
            p95=df["latency_ms"],
            p99=df["latency_ms"],
        )
        return df

    # Otherwise, smooth with 5-min buckets
    df = (
        df.set_index("created_at")
          .groupby("model_version")
          .resample("5min")["latency_ms"]
          .agg(p50=lambda s: s.quantile(0.5),
               p90=lambda s: s.quantile(0.9),
               p95=lambda s: s.quantile(0.95),
               p99=lambda s: s.quantile(0.99))
          .reset_index()
    )
    return df


@st.cache_data(ttl=30, show_spinner=False)
def fetch_label_distribution(window_label: str, versions: Sequence[str]) -> pd.DataFrame:
    interval = _window_to_interval(window_label)
    ver_sql, params, typed = _version_filter_sql(versions)

    base_sql = """
        SELECT date_trunc('hour', created_at) AS ts_hour, lbl::text AS label, COUNT(*) AS n
        FROM (
            SELECT created_at, model_version, unnest(labels) AS lbl
            FROM prediction_logs
            WHERE 1=1 {ver}
            {interval}
        ) t
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    if interval:
        base_sql = base_sql.format(ver=ver_sql, interval="AND created_at >= now() - interval :ival")
        params = {**params, "ival": interval}
    else:
        base_sql = base_sql.format(ver=ver_sql, interval="")
    with get_engine().connect() as conn:
        stmt = text(base_sql).bindparams(**typed)
        df = pd.read_sql_query(stmt, conn, params=params)
    if df.empty:
        return df
    df["ts_hour"] = pd.to_datetime(df["ts_hour"], utc=True).dt.tz_convert("UTC")
    return df

@st.cache_data(ttl=30, show_spinner=False)
def fetch_feedback_accuracy(window_label: str, versions: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    interval = _window_to_interval(window_label)
    ver_sql, params, typed = _version_filter_sql(versions)

    # overall & per-day accuracy (only on rows where feedback exists)
    overall_sql = """
        SELECT
          AVG(CASE WHEN f.correct THEN 1 ELSE 0 END)::float AS accuracy,
          COUNT(*) AS n
        FROM feedback f
        JOIN prediction_logs p ON p.request_id = f.request_id
        WHERE 1=1 {ver} {interval}
    """
    daily_sql = """
        SELECT
          date_trunc('day', f.created_at) AS day,
          AVG(CASE WHEN f.correct THEN 1 ELSE 0 END)::float AS accuracy,
          COUNT(*) AS n
        FROM feedback f
        JOIN prediction_logs p ON p.request_id = f.request_id
        WHERE 1=1 {ver} {interval}
        GROUP BY 1
        ORDER BY 1
    """
    if interval:
        overall_sql = overall_sql.format(ver=ver_sql, interval="AND f.created_at >= now() - interval :ival")
        daily_sql   = daily_sql.format(ver=ver_sql, interval="AND f.created_at >= now() - interval :ival")
        params = {**params, "ival": interval}
    else:
        overall_sql = overall_sql.format(ver=ver_sql, interval="")
        daily_sql   = daily_sql.format(ver=ver_sql, interval="")

    with get_engine().connect() as conn:
        overall_stmt = text(overall_sql).bindparams(**typed)
        daily_stmt   = text(daily_sql).bindparams(**typed)
        overall = pd.read_sql_query(overall_stmt, conn, params=params)
        daily   = pd.read_sql_query(daily_stmt,   conn, params=params)

    if not daily.empty:
        daily["day"] = pd.to_datetime(daily["day"], utc=True).dt.tz_convert("UTC")
    return overall, daily

# --- UI ---
st.title("🖥️ Model Monitoring – Toxic Moderation")

# Controls
left, right = st.columns([3, 2])
with left:
    window = st.selectbox("Time window", ["Last 1 hour", "Last 24 hours", "Last 7 days", "Last 30 days", "All time"], index=1)
with right:
    versions = get_versions()
    selected_versions = st.multiselect("Filter by model version", versions, default=versions[-1:] if versions else [])

# --- Latency over time ---
st.subheader("Latency (p50 / p90 / p95 / p99)")
lat = fetch_latency(window, selected_versions)
if lat.empty:
    st.info("No data in the selected window.")
else:
    lat = lat.rename(columns={"created_at": "timestamp"})
    for q in ["p50", "p90", "p95", "p99"]:
        chart = (
            alt.Chart(lat)
               .mark_line()
               .encode(
                   x=alt.X("timestamp:T", title="Time (UTC)"),
                   y=alt.Y(f"{q}:Q", title=f"{q.upper()} latency (ms)"),
                   color=alt.Color("model_version:N", title="Model version"),
                   tooltip=[
                       alt.Tooltip("timestamp:T", title="Time (UTC)"),
                       alt.Tooltip("model_version:N", title="Version"),
                       alt.Tooltip(f"{q}:Q", title=f"{q.upper()} (ms)", format=".1f"),
                   ],
               )
               .properties(height=220, title=f"{q.upper()}")
        )
        st.altair_chart(chart, use_container_width=True)

# --- Target drift: predicted class distribution ---
st.subheader("Predicted label distribution (per hour)")
dist = fetch_label_distribution(window, selected_versions)
if dist.empty:
    st.info("No label data in the selected window.")
else:
    dist = dist.rename(columns={"ts_hour": "timestamp"})
    chart = (
        alt.Chart(dist)
           .mark_area()
           .encode(
               x=alt.X("timestamp:T", title="Time (UTC)"),
               y=alt.Y("n:Q", title="Predictions per hour", stack="zero"),
               color=alt.Color("label:N", title="Predicted label"),
               tooltip=[
                   alt.Tooltip("timestamp:T", title="Time (UTC)"),
                   alt.Tooltip("label:N", title="Label"),
                   alt.Tooltip("n:Q", title="Count"),
               ],
           )
           .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)

# --- Feedback-derived accuracy ---
st.subheader("Live accuracy (from /feedback)")
overall, daily = fetch_feedback_accuracy(window, selected_versions)
if overall.empty or pd.isna(overall.loc[0, "accuracy"]):
    st.info("No feedback yet in the selected window.")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Overall accuracy (feedback)", f"{overall.loc[0,'accuracy']*100:.1f}%", help=f"N={int(overall.loc[0,'n'])}")
    with c2:
        st.metric("Feedback count", f"{int(overall.loc[0,'n'])}")

    if not daily.empty:
        daily = daily.copy()
        daily["accuracy_pct"] = daily["accuracy"] * 100.0
        chart = (
            alt.Chart(daily.rename(columns={"day": "date"}))
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date (UTC)"),
                y=alt.Y("accuracy_pct:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    alt.Tooltip("date:T", title="Date (UTC)"),
                    alt.Tooltip("accuracy_pct:Q", title="Accuracy (%)", format=".1f"),
                    alt.Tooltip("n:Q", title="N feedback"),
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(chart, use_container_width=True)