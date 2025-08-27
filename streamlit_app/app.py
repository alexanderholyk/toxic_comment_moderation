# comment_moderation/streamlit_app/app.py
import os
import time
import requests
import streamlit as st
import pandas as pd

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
THRESHOLD_DEFAULT = float(os.environ.get("UI_THRESHOLD", "0.5"))
LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

st.set_page_config(page_title="Toxic Comment Moderation", layout="centered")
st.title("Toxic Comment Moderation")
st.caption(f"Backend: {API_URL}")

# UI controls
threshold = st.slider("Decision threshold", 0.0, 1.0, THRESHOLD_DEFAULT, 0.05)
txt = st.text_area("Enter a comment", height=180, max_chars=5000, placeholder="Type a comment to analyze…")

# Keep last result in session for feedback
if "last_result" not in st.session_state:
    st.session_state.last_result = None

def call_predict(comment_text: str, timeout=20):
    url = f"{API_URL}/predict"
    t0 = time.time()
    r = requests.post(url, json={"comment_text": comment_text}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    data["latency_ms_client"] = (time.time() - t0) * 1000.0
    return data

def call_feedback(request_id: str, correct: bool, true_labels=None, notes=None, timeout=10):
    url = f"{API_URL}/feedback"
    payload = {"request_id": request_id, "correct": correct}
    if true_labels is not None:
        payload["true_labels"] = true_labels
    if notes:
        payload["notes"] = notes
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

cols = st.columns([3,1])
with cols[0]:
    submit = st.button("Analyze")
with cols[1]:
    clear = st.button("Clear")

if clear:
    st.session_state.last_result = None
    st.experimental_rerun()

if submit and txt.strip():
    with st.spinner("Scoring…"):
        try:
            result = call_predict(txt.strip())
            st.session_state.last_result = result
        except requests.RequestException as e:
            st.error(f"Prediction failed: {e}")

res = st.session_state.last_result
if res:
    st.subheader("Prediction")
    model_version = res.get("model_version", "?")
    request_id = res.get("request_id", "")

    # predicted labels (>= threshold)
    predicted = [lab for lab, s in res["scores"].items() if float(s) >= threshold]
    if predicted:
        st.write("Predicted labels (≥ {:.2f}): ".format(threshold) + ", ".join(f"`{p}`" for p in predicted))
    else:
        st.write("No labels above threshold.")

    # scores chart
    df = pd.DataFrame(
        [{"label": k, "score": float(v)} for k, v in res["scores"].items()]
    ).sort_values("score", ascending=False)
    st.bar_chart(df.set_index("label"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Server model version", model_version)
    c2.metric("Client latency", f"{res.get('latency_ms_client', 0):.0f} ms")
    c3.metric("Threshold", f"{threshold:.2f}")

    st.divider()
    st.subheader("Feedback")
    fcols = st.columns([1,1,2])
    with fcols[0]:
        if st.button("Mark correct"):
            try:
                call_feedback(request_id=request_id, correct=True)
                st.success("Thanks. Marked as correct.")
            except requests.RequestException as e:
                st.error(f"Feedback failed: {e}")
    with fcols[1]:
        if st.button("Mark incorrect"):
            st.session_state["show_feedback_form"] = True

    if st.session_state.get("show_feedback_form"):
        true_labels = st.multiselect("Select the correct labels", LABELS, default=predicted)
        notes = st.text_input("Notes (optional)")
        if st.button("Submit correction"):
            try:
                call_feedback(request_id=request_id, correct=False, true_labels=true_labels, notes=notes)
                st.success("Correction recorded.")
                st.session_state["show_feedback_form"] = False
            except requests.RequestException as e:
                st.error(f"Feedback failed: {e}")