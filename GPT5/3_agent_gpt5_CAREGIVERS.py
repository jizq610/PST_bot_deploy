import os
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

load_dotenv()

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def load_prompt(filename: str) -> str:
    with open(os.path.join(PROMPTS_DIR, filename), "r", encoding="utf-8") as prompt_file:
        return prompt_file.read()

BEHAVIOR_PROMPT = load_prompt("behavior_prompt.txt")

AC_PROMPT = load_prompt("ac_prompt.txt")

STRATEGY_PROMPT = load_prompt("strategy_prompt.txt")

# -------------------------------------------------
# Constants
# -------------------------------------------------
MODEL_NAME = "gpt-5-nano"
REASONING_EFFORT = "low"

INITIAL_ASSISTANT_MESSAGE = (
    "Hello, I’m glad you’re here. To get started, could you briefly share your caregiving situation with me? "
    "For example, you might tell me who you’re caring for, your relationship to them, and what behavior or "
    "situation has been especially challenging recently."
)

EVAL_ITEMS = [
    {"key": "correct_behavior", "label": "The virtual assistant successfully guided the caregiver in developing an ABC plan. (1=disagree, 2=neutral, 3=agree)"},
    {"key": "expressed_warmth_compassion", "label": "The virtual assistant expressed emotions such as warmth, compassion, concern, or similar feelings towards the caregiver. (1=no expression, 2=weak expression, 3=strong expression)"},
    {"key": "communicated_understanding", "label": "The virtual assistant communicated an understanding of feelings and experiences inferred from the caregiver’s responses. (1=no expression, 2=weak expression, 3=strong expression)"},
    {"key": "improved_understanding", "label": "The virtual assistant improved their understanding of the caregiver by exploring feelings and experiences not stated in the caregiver’s response. (1=no expression, 2=weak expression, 3=strong expression)"},
    {"key": "overall_satisfaction", "label": "Overall, I am satisfied with the coaching session. (1=disagree, 2=neutral, 3=agree)"},
]

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_db_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "conversations.db"))

def get_db_connection():
    conn = sqlite3.connect(get_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

@st.cache_resource
def init_monitoring_db():
    conn = get_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                drive_file_id TEXT
            )
            """
        )
        columns = conn.execute("PRAGMA table_info(sessions)").fetchall()
        if not any(col[1] == "drive_file_id" for col in columns):
            conn.execute("ALTER TABLE sessions ADD COLUMN drive_file_id TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                phase TEXT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reasoning_summary TEXT,
                api_response_json TEXT,
                research_review_status TEXT
            )
            """
        )
        message_columns = {col[1] for col in conn.execute("PRAGMA table_info(messages)").fetchall()}
        for column in ("reasoning_summary", "api_response_json", "research_review_status"):
            if column not in message_columns:
                conn.execute(f"ALTER TABLE messages ADD COLUMN {column} TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS monitoring_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                error_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

def record_monitoring_error(session_id: str, error_text: str):
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO monitoring_errors (session_id, error_text, created_at) VALUES (?, ?, ?)",
            (session_id, str(error_text), datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
        conn.commit()
    finally:
        conn.close()

def record_session_start(session_id: str):
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, started_at, last_activity, status) VALUES (?, ?, ?, 'active')",
            (session_id, now, now),
        )
        conn.commit()
    finally:
        conn.close()

def record_message(session_id: str, role: str, content: str, phase: str = "", *, reasoning_summary: str = None):
    if not content or not content.strip():
        return
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = get_db_connection()
    try:
        message = conn.execute(
            """
            INSERT INTO messages (session_id, role, phase, content, created_at, reasoning_summary)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, role, phase, content.strip(), now, reasoning_summary),
        )
        conn.execute(
            "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
            (now, session_id),
        )
        conn.commit()
        return message.lastrowid
    finally:
        conn.close()

def get_drive_service(service_account_info):
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )
    return build("drive", "v3", credentials=credentials)

def find_drive_file(service, file_name: str, folder_id: str):
    response = service.files().list(
        q=f"name='{file_name}' and '{folder_id}' in parents and trashed=false",
        fields="files(id, name)",
        supportsAllDrives=True,
    ).execute()
    files = response.get("files", [])
    return files[0] if files else None

def set_session_drive_file_id(session_id: str, drive_file_id: str):
    conn = get_db_connection()
    try:
        conn.execute(
            "UPDATE sessions SET drive_file_id = ? WHERE session_id = ?",
            (drive_file_id, session_id),
        )
        conn.commit()
    finally:
        conn.close()

def get_session_file_name(session_id: str) -> str:
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT started_at FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise ValueError(f"No session found for {session_id}")

    started_at = datetime.strptime(row["started_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    pacific = started_at.astimezone(ZoneInfo("America/Los_Angeles"))
    tz_label = "PDT" if pacific.dst() else "PST"
    filename = pacific.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{filename}_{tz_label}_{session_id}.xlsx"

def save_session_transcript_to_drive(session_id: str, drive_settings):
    try:
        if drive_settings is None:
            raise RuntimeError("Google Drive credentials or folder ID are missing.")
        service_account_info, folder_id = drive_settings

        conn = get_db_connection()
        try:
            session_row = conn.execute(
                "SELECT drive_file_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            rows = conn.execute(
                """
                SELECT created_at AS timestamp, role, phase, content,
                       CASE WHEN research_review_status IS NULL THEN reasoning_summary ELSE NULL END AS reasoning_summary,
                       session_id
                FROM messages WHERE session_id = ? ORDER BY id
                """,
                (session_id,),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return

        drive_file_id = session_row["drive_file_id"] if session_row and session_row["drive_file_id"] else None
        chat_df = pd.DataFrame(
            [dict(row) for row in rows],
            columns=["timestamp", "role", "phase", "content", "reasoning_summary", "session_id"],
        )

        excel_bytes = dataframe_to_excel_bytes(chat_df)
        service = get_drive_service(service_account_info)

        if drive_file_id:
            service.files().update(
                fileId=drive_file_id,
                media_body=MediaIoBaseUpload(BytesIO(excel_bytes), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", resumable=False),
                supportsAllDrives=True,
            ).execute()
            return

        file_name = get_session_file_name(session_id)
        existing_file = find_drive_file(service, file_name, folder_id)

        if existing_file:
            drive_file_id = existing_file["id"]
            set_session_drive_file_id(session_id, drive_file_id)
            service.files().update(
                fileId=drive_file_id,
                media_body=MediaIoBaseUpload(BytesIO(excel_bytes), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", resumable=False),
                supportsAllDrives=True,
            ).execute()
            return

        file_metadata = {"name": file_name, "parents": [folder_id]}
        created_file = service.files().create(
            body=file_metadata,
            media_body=MediaIoBaseUpload(BytesIO(excel_bytes), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", resumable=False),
            fields="id, name",
            supportsAllDrives=True,
        ).execute()

        drive_file_id = created_file.get("id")
        if not drive_file_id:
            recovered_file = find_drive_file(service, file_name, folder_id)
            if recovered_file is None:
                raise RuntimeError(f"Create response missing file ID for session {session_id}; backup could not be recovered.")
            drive_file_id = recovered_file["id"]

        set_session_drive_file_id(session_id, drive_file_id)
    except Exception as exc:
        record_monitoring_error(session_id, f"Drive backup failed: {exc}")


def get_drive_settings():
    try:
        return dict(st.secrets["gcp_service_account"]), st.secrets["GOOGLE_DRIVE_FOLDER_ID"]
    except (KeyError, FileNotFoundError):
        return None


@st.cache_resource
def get_backup_executor():
    # Serial writes keep older transcript snapshots from overwriting newer ones.
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="drive-backup")

init_monitoring_db()

def phase_label(phase: str) -> str:
    if phase == "BEHAVIOR":
        return "Stage 1: Behavior Identification"
    if phase == "AC":
        return "Stage 2: Activators + Consequences"
    if phase == "STRATEGY":
        return "Stage 3: Strategies"
    return "Unknown Stage"

def get_system_prompt_for_phase(phase: str) -> str:
    return {
        "BEHAVIOR": BEHAVIOR_PROMPT,
        "AC": AC_PROMPT,
        "STRATEGY": STRATEGY_PROMPT,
    }[phase]

def get_kickoff_message_for_phase(phase: str) -> str:
    return {
        "AC": "Now let’s look at what happens before and after the behavior. In addition to what we've discussed so far, any thoughts that come to your mind?",
        "STRATEGY": "Now let's think about strategies. What can we change before or (what happens) after the behavior to see if it makes a difference going forward?",
    }[phase]

def make_ai_message(content: str, model_name: str = MODEL_NAME) -> AIMessage:
    return AIMessage(
        content=content,
        additional_kwargs={
            "timestamp": current_timestamp(),
            "model_name": model_name,
        },
    )

def make_human_message(content: str) -> HumanMessage:
    return HumanMessage(
        content=content,
        additional_kwargs={
            "timestamp": current_timestamp(),
            "model_name": "",
        },
    )

def messages_to_dataframe(messages):
    rows = []

    for m in messages:
        if isinstance(m, SystemMessage):
            continue

        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            role = "unknown"

        rows.append(
            {
                "timestamp": m.additional_kwargs.get("timestamp", ""),
                "role": role,
                "model_name": "model1",
                "phase": m.additional_kwargs.get("phase", ""),
                "content": m.content,
            }
        )

    return pd.DataFrame(rows)

def ratings_to_dataframe(ratings):
    rows = []

    for item in EVAL_ITEMS:
        key = item["key"]
        rows.append(
            {
                "criterion": item["label"],
                "rating": ratings.get(key, ""),
                "comments": ratings.get(f"{key}_comments", ""),
            }
        )

    return pd.DataFrame(rows)

def sidebar_status_to_dataframe():
    return pd.DataFrame(
        [
            {"field": "current_phase", "value": st.session_state.phase},
            {"field": "current_stage_label", "value": phase_label(st.session_state.phase)},
            {"field": "ac_kickoff_sent", "value": st.session_state.ac_kickoff_sent},
            {"field": "strategy_kickoff_sent", "value": st.session_state.strategy_kickoff_sent},
        ]
    )

def persona_description_to_dataframe():
    return pd.DataFrame(
        [
            {
                "persona_description": st.session_state.persona_description,
            }
        ]
    )

def study_id_to_dataframe():
    return pd.DataFrame(
        [
            {
                "study_id": st.session_state.study_id,
            }
        ]
    )

def dataframe_to_excel_bytes(chat_df, ratings_df=None):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        chat_df.to_excel(writer, index=False, sheet_name="chat_history")

    return output.getvalue()

def reset_conversation():
    st.session_state.phase = "BEHAVIOR"
    st.session_state.ac_kickoff_sent = False
    st.session_state.strategy_kickoff_sent = False
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.session_started_logged = False

    st.session_state.messages = [
        SystemMessage(content=get_system_prompt_for_phase("BEHAVIOR")),
        make_ai_message(INITIAL_ASSISTANT_MESSAGE),
    ]

    record_session_start(st.session_state.session_id)
    record_message(st.session_state.session_id, "assistant", INITIAL_ASSISTANT_MESSAGE, "BEHAVIOR")
    get_backup_executor().submit(save_session_transcript_to_drive, st.session_state.session_id, get_drive_settings())
    st.session_state.session_started_logged = True

def initialize_session_state():
    if "phase" not in st.session_state:
        st.session_state.phase = "BEHAVIOR"

    if "ac_kickoff_sent" not in st.session_state:
        st.session_state.ac_kickoff_sent = False

    if "strategy_kickoff_sent" not in st.session_state:
        st.session_state.strategy_kickoff_sent = False

    if "session_started_logged" not in st.session_state:
        st.session_state.session_started_logged = False

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=get_system_prompt_for_phase("BEHAVIOR")),
            make_ai_message(INITIAL_ASSISTANT_MESSAGE),
        ]

    if "ratings" not in st.session_state:
        st.session_state.ratings = {}
        for item in EVAL_ITEMS:
            st.session_state.ratings[item["key"]] = ""
            st.session_state.ratings[f"{item['key']}_comments"] = ""

    if "persona_description" not in st.session_state:
        st.session_state.persona_description = ""

    if "study_id" not in st.session_state:
        st.session_state.study_id = ""

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
        record_session_start(st.session_state.session_id)
        record_message(st.session_state.session_id, "assistant", INITIAL_ASSISTANT_MESSAGE, "BEHAVIOR")
        get_backup_executor().submit(save_session_transcript_to_drive, st.session_state.session_id, get_drive_settings())
        st.session_state.session_started_logged = True

    if not st.session_state.session_started_logged:
        st.session_state.session_started_logged = True

def set_message_phase_metadata(message):
    if "phase" not in message.additional_kwargs:
        message.additional_kwargs["phase"] = st.session_state.phase
    return message

def advance_phase_after_handoff(clean_text: str):
    current_phase = st.session_state.phase

    if current_phase == "BEHAVIOR":
        next_phase = "AC"
        kickoff_flag = "ac_kickoff_sent"
    elif current_phase == "AC":
        next_phase = "STRATEGY"
        kickoff_flag = "strategy_kickoff_sent"
    else:
        return "You have completed the ABC problem solving plan.\n\nHANDOFF_READY"

    st.session_state.phase = next_phase
    st.session_state.messages[0] = SystemMessage(content=get_system_prompt_for_phase(next_phase))
    if st.session_state[kickoff_flag]:
        return None

    st.session_state[kickoff_flag] = True
    kickoff_text = get_kickoff_message_for_phase(next_phase)
    return f"{clean_text}\n\n{kickoff_text}" if clean_text else kickoff_text


def run_llm_and_update_conversation(user_text: str):
    initialize_session_state()
    session_id = st.session_state.session_id
    phase = st.session_state.phase
    user_msg = make_human_message(user_text)
    user_msg.additional_kwargs["phase"] = phase
    st.session_state.messages.append(user_msg)
    record_message(session_id, "user", user_text, phase)

    api_messages = []
    for message in st.session_state.messages:
        if isinstance(message, SystemMessage):
            role = "developer"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        api_messages.append({"role": role, "content": str(message.content)})

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    with OpenAI(api_key=api_key) as client:
        response = client.responses.create(
            model=MODEL_NAME,
            input=api_messages,
            reasoning={"effort": REASONING_EFFORT, "summary": "auto"},
            store=False,
        )

    # API-provided summary from the same response that produced the caregiver reply.
    reasoning_summary = "\n".join(
        part.text
        for item in response.output
        if item.type == "reasoning"
        for part in (item.summary or [])
        if part.type == "summary_text"
    ).strip()

    assistant_text = (response.output_text or "").strip()
    if not assistant_text:
        raise RuntimeError(f"{MODEL_NAME} returned no caregiver-facing text.")

    if "HANDOFF_READY" in assistant_text:
        assistant_text = advance_phase_after_handoff(assistant_text.replace("HANDOFF_READY", "").strip())

    backup_executor = get_backup_executor()
    drive_settings = get_drive_settings()
    if assistant_text is None:
        backup_executor.submit(save_session_transcript_to_drive, session_id, drive_settings)
        return

    posted_phase = st.session_state.phase
    assistant_msg = make_ai_message(assistant_text)
    assistant_msg.additional_kwargs["phase"] = posted_phase
    st.session_state.messages.append(assistant_msg)

    posted_reply = assistant_text.replace("HANDOFF_READY", "").strip()
    record_message(
        session_id, "assistant", posted_reply, posted_phase,
        reasoning_summary=reasoning_summary or None,
    )
    backup_executor.submit(save_session_transcript_to_drive, session_id, drive_settings)


def upload_excel_to_drive(excel_data, file_name):
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )

    service = build("drive", "v3", credentials=credentials)

    file_metadata = {
        "name": file_name,
        "parents": [st.secrets["GOOGLE_DRIVE_FOLDER_ID"]],
    }

    media = MediaIoBaseUpload(
        BytesIO(excel_data),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        resumable=False,
    )

    uploaded_file = (
        service.files()
        .create(
            body=file_metadata,
            media_body=media,
            fields="id, name, webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )

    return uploaded_file

def chatbot_page():
    st.set_page_config(page_title="STAR-C Virtual Assistant", layout="wide", initial_sidebar_state="collapsed")
    st.markdown("## **STAR-C Virtual Assistant**")

    header_col, right_col = st.columns([6, 1])
    with right_col:
        st.page_link("pages/dashboard.py", label="Dashboard")

    initialize_session_state()

    chat_container = st.container(height=400, border=True)

    with chat_container:
        for m in st.session_state.messages:
            if isinstance(m, SystemMessage):
                continue

            role = "user" if isinstance(m, HumanMessage) else "assistant"

            with st.chat_message(role):
                st.markdown(m.content)

        live_chat_area = st.empty()
    user_text = st.chat_input("Type your message...")

    if user_text:
        with live_chat_area.container():
            with st.chat_message("user"):
                st.markdown(user_text)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    run_llm_and_update_conversation(user_text)

        st.rerun()

st.navigation(
    [
        st.Page(chatbot_page, title="Chatbot", url_path="chatbot", default=True),
        st.Page("pages/dashboard.py", title="Dashboard", url_path="dashboard"),
    ],
    position="hidden",
).run()

 
