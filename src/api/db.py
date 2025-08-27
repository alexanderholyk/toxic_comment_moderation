# src/api/db.py


from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text, bindparam, String
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

load_dotenv()

DEV_SKIP = os.getenv("DEV_DB_SKIP") == "1"
_DB_URL = os.getenv("APP_DB_URL")

engine: Optional[Engine]
if DEV_SKIP or not _DB_URL:
    engine = None
else:
    # Create the database engine
    engine = create_engine(
        _DB_URL,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        future=True,
    )

    # Create the tables if they don't exist
    if os.getenv("APP_DB_AUTOCREATE") == "1":
        with engine.begin() as conn:
            conn.execute(text("""
                create table if not exists prediction_logs (
                  id bigserial primary key,
                  request_id uuid not null,
                  comment_text text not null,
                  input_hash char(64) not null,
                  scores jsonb not null,
                  labels text[] not null,
                  model_name text not null,
                  model_version text not null,
                  latency_ms integer not null,
                  created_at timestamptz not null default now()
                );
                create unique index if not exists uq_prediction_logs_request
                  on prediction_logs(request_id);
                create index if not exists idx_prediction_logs_created_at on prediction_logs(created_at);
                create index if not exists idx_prediction_logs_input_hash on prediction_logs(input_hash);

                create table if not exists feedback (
                  id bigserial primary key,
                  request_id uuid not null references prediction_logs(request_id) on delete cascade,
                  correct boolean not null,
                  true_labels text[] null,
                  notes text null,
                  created_at timestamptz not null default now()
                );
            """))


@contextmanager
def get_conn():
    """
    Get a database connection.
    """
    if engine is None:
        raise RuntimeError("Database is not configured (DEV_DB_SKIP=1 or APP_DB_URL missing).")
    with engine.begin() as conn:
        yield conn


def log_prediction(
    comment_text: str,
    input_hash: str,
    scores: Dict[str, float] | Dict[str, Any],
    labels: list[str],
    model_name: str,
    model_version: str,
    latency_ms: int,
) -> str:
    
    if engine is None:
        return "dev-skip-" + uuid.uuid4().hex[:12]

    rid = str(uuid.uuid4())

    # Ensure pure-Python types (no numpy) for JSONB
    clean_scores = _ensure_json_serializable(scores)

    # Use typed bind params for JSONB and text[]
    stmt = text("""
        insert into prediction_logs
          (request_id, comment_text, input_hash, scores, labels, model_name, model_version, latency_ms)
        values
          (:rid, :ct, :ih, :sc, :lb, :mn, :mv, :lat)
    """).bindparams(
        bindparam("sc", type_=JSONB),
        bindparam("lb", type_=ARRAY(String())),
    )

    with get_conn() as conn:
        conn.execute(
            stmt,
            {
                "rid": rid,
                "ct": comment_text,
                "ih": input_hash,
                "sc": clean_scores,      # dict → JSONB
                "lb": labels,            # list[str] → text[]
                "mn": model_name,
                "mv": model_version,
                "lat": int(latency_ms),
            },
        )

    return rid


def fetch_cached(input_hash: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a cached prediction log by input_hash.
    """
    if engine is None:
        return None
    
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                select scores, labels, model_version
                from prediction_logs
                where input_hash = :ih
                order by created_at desc
                limit 1
            """),
            {"ih": input_hash},
        ).mappings().first()

        if not row:
            return None
        
        return {
            "scores": dict(row["scores"]) if row["scores"] is not None else {},
            "labels": list(row["labels"]) if row["labels"] is not None else [],
            "model_version": str(row["model_version"]),
        }


def _ensure_json_serializable(obj: Any) -> Any:
    """
    Ensure the object is JSON serializable.
    """
    try:
        import numpy as np
    except Exception:
        return obj
    
    if isinstance(obj, dict):
        return {k: _ensure_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ensure_json_serializable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj