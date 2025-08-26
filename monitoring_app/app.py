import os, streamlit as st, pandas as pd
from sqlalchemy import create_engine

DB_URL = os.environ["APP_DB_URL"]
engine = create_engine(DB_URL)

st.title("Model Monitoring")

@st.cache_data(ttl=60)
def load_data(n=50000):
    q = """
      select created_at, latency_ms, labels, scores
      from prediction_logs
      where created_at > now() - interval '30 days'
      order by created_at asc
    """
    return pd.read_sql(q, engine)

df = load_data()

st.subheader("Latency (ms) over time")
st.line_chart(df.set_index("created_at")["latency_ms"])

st.subheader("Predicted class distribution (last 30 days)")
# explode labels
lab_counts = (df.explode("labels")["labels"]
              .value_counts().rename_axis("label").reset_index(name="count"))
st.bar_chart(lab_counts.set_index("label"))

st.subheader("Live accuracy from feedback (if collected)")
acc = pd.read_sql("""
    select date_trunc('day', f.created_at) as d, avg(case when f.correct then 1 else 0 end) as accuracy
    from feedback f group by 1 order by 1 asc
""", engine)
if len(acc):
    st.line_chart(acc.set_index("d")["accuracy"])
else:
    st.info("No feedback yet. Wire a feedback widget in the user UI.")