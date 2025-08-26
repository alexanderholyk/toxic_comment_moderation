import os, uuid, time
from sqlalchemy import create_engine, text
from contextlib import contextmanager

DB_URL = os.environ["APP_DB_URL"]  # e.g. postgresql+psycopg2://user:pass@host:5432/moderation
engine = create_engine(DB_URL, pool_size=5, max_overflow=10, pool_pre_ping=True)

@contextmanager
def get_conn():
    with engine.begin() as conn:
        yield conn

def log_prediction(comment_text, input_hash, scores, labels, model_name, model_version, latency_ms):
    rid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(text("""
            insert into prediction_logs (request_id, comment_text, input_hash, scores, labels, model_name, model_version, latency_ms)
            values (:rid, :ct, :ih, :sc::jsonb, :lb, :mn, :mv, :lat)
        """), {"rid": rid, "ct": comment_text, "ih": input_hash,
               "sc": scores, "lb": labels, "mn": model_name, "mv": model_version, "lat": latency_ms})
    return rid

def fetch_cached(input_hash):
    with get_conn() as conn:
        row = conn.execute(text("""
            select scores, labels, model_version from prediction_logs
            where input_hash=:ih
            order by created_at desc limit 1
        """), {"ih": input_hash}).mappings().first()
        return dict(row) if row else None