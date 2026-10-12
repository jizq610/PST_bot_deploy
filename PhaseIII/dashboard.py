import json
import os
from pathlib import Path

import streamlit as st

MONITORING_PATH = os.getenv("MONITORING_PATH", os.path.join(os.getcwd(), "monitoring", "session_events.jsonl"))


def load_events():
    path = Path(MONITORING_PATH)
    if not path.exists():
        return []

    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            txt = line.strip()
            if not txt:
                continue
            try:
                events.append(json.loads(txt))
            except json.JSONDecodeError:
                continue
    return sorted(events, key=lambda item: item.get("timestamp", ""))


def sessions_by_id(events):
    sessions = {}
    for event in events:
        session_id = event.get("session_id") or "unknown"
        sessions.setdefault(session_id, []).append(event)

    summary = []
    for session_id, items in sessions.items():
        ordered = sorted(items, key=lambda item: item.get("timestamp", ""))
        summary.append(
            {
                "session_id": session_id,
                "start_time": ordered[0].get("timestamp", ""),
                "last_activity": ordered[-1].get("timestamp", ""),
                "events": ordered,
            }
        )

    return sorted(summary, key=lambda item: item["last_activity"], reverse=True)


st.set_page_config(page_title="STAR-C Research Dashboard", layout="wide")
st.title("STAR-C Research Dashboard")


@st.fragment(run_every=5)
def dashboard():
    events = load_events()
    sessions = sessions_by_id(events)

    if not sessions:
        st.info("No participant sessions yet.")
        return

    session_options = [
        f"{item['session_id']} | start {item['start_time']} | last {item['last_activity']}"
        for item in sessions
    ]
    selected_label = st.selectbox("Select participant session", session_options, index=0)
    selected_id = selected_label.split(" | ", 1)[0]

    selected_session = next((item for item in sessions if item["session_id"] == selected_id), sessions[0])

    st.caption(f"Session ID: {selected_session['session_id']}")
    st.caption(f"Session start: {selected_session['start_time']}")
    st.caption(f"Most recent activity: {selected_session['last_activity']}")

    for event in selected_session["events"]:
        role = event.get("role", "system")
        content = event.get("content", "")
        if not content:
            continue

        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        else:
            with st.chat_message("assistant"):
                st.write(content)

    st.caption("Auto-refreshing every 5 seconds")


dashboard()
