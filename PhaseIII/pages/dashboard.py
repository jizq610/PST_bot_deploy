import os
import sqlite3

import streamlit as st


DB_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "conversations.db"))


def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def get_sessions():
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT session_id, started_at, last_activity
            FROM sessions
            ORDER BY started_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_messages_for_session(session_id: str):
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT session_id, role, phase, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_errors_for_session(session_id: str):
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT error_text, created_at
            FROM monitoring_errors
            WHERE session_id = ?
            ORDER BY id DESC
            """,
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def validate_researcher():
    password = st.secrets.get("RESEARCHER_PASSWORD") or st.secrets.get("researcher_password")

    if "researcher_authenticated" not in st.session_state:
        st.session_state.researcher_authenticated = False

    if st.session_state.researcher_authenticated:
        return True

    st.title("Researcher dashboard")
    entered_password = st.text_input("Researcher password", type="password", key="researcher_pass")
    login_clicked = st.button("Log in")

    if login_clicked:
        if entered_password and password and entered_password == password:
            st.session_state.researcher_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
            st.stop()

    st.warning("Access restricted to authorized researchers.")
    st.stop()

    return False


st.set_page_config(page_title="STAR-C Research Dashboard", layout="wide")
validate_researcher()

header_col, right_col = st.columns([6, 1])
with right_col:
    st.markdown("[← Back to Chatbot](/chatbot)")


if st.sidebar.button("Log out"):
    st.session_state.researcher_authenticated = False
    st.rerun()


@st.fragment(run_every=5)
def dashboard_page():
    sessions = get_sessions()

    if not sessions:
        st.info("No participant sessions yet.")
        return

    session_labels = [
        f"{s['session_id']} | start {s['started_at']} | last {s['last_activity']}"
        for s in sessions
    ]

    selected_label = st.selectbox("Select participant session", session_labels, index=0, key="selected_session_label")
    selected_id = selected_label.split(" | ", 1)[0]

    selected_session = next((s for s in sessions if s["session_id"] == selected_id), sessions[0])
    st.caption(f"Session ID: {selected_session['session_id']}")
    st.caption(f"Session start: {selected_session['started_at']}")
    st.caption(f"Most recent activity: {selected_session['last_activity']}")

    errors = get_errors_for_session(selected_id)
    if errors:
        for err in errors[:5]:
            st.warning(f"Backup issue: {err['error_text']} at {err['created_at']}")

    messages = get_messages_for_session(selected_id)
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        phase = msg.get("phase")
        if role not in {"user", "assistant"}:
            continue
        with st.chat_message(role):
            if phase:
                st.caption(f"Phase: {phase}")
            st.write(content)

    st.caption("Auto-refreshing every 5 seconds")


dashboard_page()
