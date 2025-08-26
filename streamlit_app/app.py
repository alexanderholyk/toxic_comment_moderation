import os, requests, streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Toxic Comment Moderation")
txt = st.text_area("Enter a comment", height=180, max_chars=5000)
if st.button("Analyze"):
    r = requests.post(f"{API_URL}/predict", json={"comment_text": txt})
    r.raise_for_status()
    data = r.json()
    st.json(data)