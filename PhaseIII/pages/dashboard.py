
import os
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import streamlit as st


# -------------------------------------------------
# Configuration
# -------------------------------------------------
DB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "conversations.db",
    )
)

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


# -------------------------------------------------
# Time formatting
# -------------------------------------------------
def format_pacific(timestamp):
    if not timestamp:
        return "Unknown"

    dt = datetime.fromisoformat(
        timestamp.replace("Z", "+00:00")
    )

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    pacific = dt.astimezone(PACIFIC_TZ)

    return pacific.strftime(
        "%b %d, %Y, %I:%M:%S %p %Z"
    )


# -------------------------------------------------
# Database connection
# -------------------------------------------------
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

        return [dict(row) for row in rows]

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

        return [dict(row) for row in rows]

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

        return [dict(row) for row in rows]

    finally:
        conn.close()


# -------------------------------------------------
# Researcher authentication
# -------------------------------------------------
def validate_researcher():
    password = (
        st.secrets.get("RESEARCHER_PASSWORD")
        or st.secrets.get("researcher_password")
    )

    if "researcher_authenticated" not in st.session_state:
        st.session_state.researcher_authenticated = False

    if st.session_state.researcher_authenticated:
        return True

    st.title("Researcher dashboard")

    entered_password = st.text_input(
        "Researcher password",
        type="password",
        key="researcher_pass",
    )

    if st.button("Log in"):
        if (
            entered_password
            and password
            and entered_password == password
        ):
            st.session_state.researcher_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
            st.stop()

    st.warning("Access restricted to authorized researchers.")
    st.stop()


# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(
    page_title="STAR-C Research Dashboard",
    layout="wide",
)

validate_researcher()

header_col, right_col = st.columns([6, 1])

# with right_col:
#     st.markdown("[← Back to Chatbot](/chatbot)")

if st.sidebar.button("Log out"):
    st.session_state.researcher_authenticated = False
    st.rerun()


# -------------------------------------------------
# Research dashboard
# -------------------------------------------------
@st.fragment(run_every=5)
def dashboard_page():

    sessions = get_sessions()

    if not sessions:
        st.info("No participant sessions yet.")
        return

    # Store sessions by their unique IDs.
    session_options = {
        session["session_id"]: session
        for session in sessions
    }

    # Display only the session start time in Pacific Time.
    selected_id = st.selectbox(
        "Select participant session",
        list(session_options.keys()),
        format_func=lambda sid: format_pacific(
            session_options[sid]["started_at"]
        ),
        key="selected_session_id",
    )

    selected_session = session_options[selected_id]

    # Session details
    st.caption(f"Session ID: {selected_id}")

    st.caption(
        "Session start: "
        + format_pacific(selected_session["started_at"])
    )

    st.caption(
        "Most recent activity: "
        + format_pacific(selected_session["last_activity"])
    )

    # Show backup errors for the selected session.
    errors = get_errors_for_session(selected_id)

    if errors:
        for error in errors[:5]:
            st.warning(
                "Backup issue: "
                + error["error_text"]
                + " at "
                + format_pacific(error["created_at"])
            )

    # Display conversation history.
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
