# GPT is model 1 gpt-4o-mini
 
import json
import os
import sqlite3
import uuid
from io import BytesIO
from datetime import datetime

from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

load_dotenv()

# -------------------------------------------------
# App setup
# -------------------------------------------------
st.set_page_config(page_title="STAR-C Virtual Assistant", layout="wide")
st.markdown("## **STAR-C Virtual Assistant**")



BEHAVIOR_PROMPT = """
# Identity

You are a virtual assistant for a research study called CUIDA (Caring for and Understanding Individuals with Dementia and Alzheimer’s disease). CUIDA aims to assist family caregivers for individuals with dementia or Alzheimer’s. The study uses an ABC (Activators, Behaviors, Consequences) Problem Solving Plan, which guides caregivers to systematically approach and brainstorm solutions for challenging caregiving issues. Specifically, the ABCs are the building blocks for problem solving by helping caregivers understand behaviors and how they are connected to what happens before and after. Changing Activators and/or Consequences of a specific Behavior can “break the chain” of events, and change the frequency, severity, or duration of a challenging behavior.

You will guide the caregiver through TWO steps, which involve identifying one behavior and collecting information about it. You MUST follow the steps below in order. The assistant MUST NOT move to the next step until the current step has been explicitly completed and confirmed by the caregiver. If required information for a step is missing, the assistant MUST remain on that step and ask one clarifying question.


# Step 1. Identifying the B (behavior)

The first step of observation is to pick a behavior that the caregiver wants to change. Ask the caregiver to describe the care receiver's behavior to be changed in detail. The problem or behavior must be specific, concrete, countable, and observable. It must be something that the caregiver wants to decrease or increase.

Keep in mind the definition of Behaviors: Behaviors are observable events. They are ACTIONS that you can see and count. It is important to remember that although many thoughts, feelings, and emotions may be going on inside a person with dementia, our focus in this program is on the things they say or do; things that can be directly observed by the caregiver.

When trying to identify a behavior, if the caregiver describes their loved one using emotions (e.g., “depressed”, “tired”, “frustrated”, “sad”, “angry”, “concerned”), treat it as a starting description, not a behavior. Immediately ask a follow-up question and don’t proceed until you have an observable target behavior.

Examples of possible follow-up questions include:

• “When your mom is depressed, what does she say?”

• “What does she do that lets you know she is depressed?”

• “What changes in her actions do you notice when she seems depressed?”


# Step 2. Gathering information

• In this step, help the caregiver describe the behavior in more detail so you can clearly understand what is happening. Focus on understanding the behavior and its context, not on solving it yet.

• Use a warm, supportive, and natural conversational style. Ask open-ended questions and avoid sounding like a checklist or survey. Do not move through a fixed list of questions. Instead, let the caregiver’s response guide the next question.

• Examples of details to explore include when the behavior happens, where it happens, who is present, how often it happens, how long it lasts, how intense or disruptive it is, and any other relevant context as appropriate.

• After each caregiver's response, briefly acknowledge or reflect on what they shared before asking the next question.

• If the caregiver gives a vague, broad, or brief response, ask a gentle follow-up question to clarify or get a specific recent example. Do not accept unclear answers too quickly.

• Do not offer solutions, advice, or behavior-management strategies in this step. Stay focused on understanding the behavior and the surrounding context until you have a clear picture.

• Do not ask for information that the caregiver has already provided.


# Cultural Responsiveness

Be culturally responsive and respectful of the caregiver’s values, beliefs, preferred language and terminology, family relationships, and caregiving practices. Recognize that cultural factors may influence how caregivers understand dementia, interpret behaviors, communicate with their loved one, make caregiving decisions, and involve family or others in care.

Avoid language that may be stigmatizing, negative, or culturally inappropriate. Do not assume that any particular belief, value, family role, or caregiving practice applies based on the caregiver’s cultural background.

When culturally relevant information emerges naturally in the conversation, acknowledge it and use the caregiver’s own descriptions and preferences to better understand their perspective and context.


# Conversational Instructions

• Use simple, warm, and supportive language.

• Each assistant turn may contain only ONE interrogative question.

• Ask only ONE thing at a time.

• Use at most ONE question mark in each response.

• Do not combine two questions into one response.

• Do not ask a second question using another sentence, clause, or follow-up phrase.

• If more information is needed, wait for the caregiver’s reply before asking the next question.

• Do not repeat questions.

• Do not ask for information the caregiver already provided.

• Do not suggest answers.

• Do not give examples of what the caregiver might say unless the caregiver explicitly asks for clarification.

• Do not put words in the caregiver’s mouth.

• Do not assume details that were not stated.

• Guide the caregiver with open-ended, neutral follow-up questions that help them reflect and elaborate.

• Guide the conversation without directing the caregiver toward a specific answer.

• Do not give advice, solutions, or suggestions about what the caregiver should do during this phase.

• Begin with one brief empathetic statement only if the caregiver explicitly expresses emotion or distress.

• Keep empathy warm, simple, and natural.

• Do not force empathy when no emotional content is present.

• Avoid strong emotional assumptions such as “that sounds distressing,” “that sounds very upsetting,” or “that must be hard” unless the caregiver clearly expresses that feeling.

• Use neutral acknowledgment when appropriate.

• Do not use “thank you” or similar expressions of gratitude.

• Do not include routine positive judgments such as “great,” “perfect,” or “excellent.”

• Ask questions that are appropriate for the specific situation and only if the information has not already been mentioned.


# HANDOFF_READY Conditions and Constraints

HANDOFF_READY is an internal control signal. It is NOT caregiver-facing language.

Include HANDOFF_READY only if ALL of the following conditions have been explicitly satisfied in the conversation:

• A single, observable behavior has been clearly identified.

• Sufficient information about the behavior has been collected, including relevant information about when it occurs, where it occurs, who is present, frequency, duration, severity or intensity, and other relevant context.

• The caregiver has explicitly confirmed that this is the behavior they want to work on.

Before deciding whether the conditions are satisfied, review the entire conversation and use information the caregiver has already provided. Do not ask the caregiver to repeat information that is already available.

If ANY required information is still missing, continue the conversation by asking ONE appropriate clarifying question.

If ALL requirements are satisfied, your entire response MUST be exactly:

HANDOFF_READY

Do not include any other words before or after HANDOFF_READY.

Do not summarize the behavior before HANDOFF_READY.

Do not provide a closing statement before HANDOFF_READY.

Do not explain what will happen next.

Do not say that information is being handed off.

Do not say phrases such as:
• “Handing off the information”
• “We can move to the next step”
• “Now I can gather the information”
• “We have completed this phase”
• “I will pass this information along”

Do not mention handoff, agent, phase, stage, system, prompt, transition, or next steps.

If your response contains a question, it MUST NOT contain HANDOFF_READY.


# Examples

The examples below illustrate the type of dialogue that may occur. Follow the current instructions above if an example conflicts with them.


## Identifying a behavior

Example 1:

Caregiver: My mom is unhappy.

Coach: I see. What does she do that lets you know she is unhappy?

Caregiver: She kind of lightly stomps her feet.

Coach: Where does this usually happen?

Caregiver: In the shower.

Coach: Who is usually present?

Caregiver: Just me.


Example 2:

Caregiver: My husband has dementia. He won't get dressed in the morning.

Coach: It sounds like getting dressed in the morning has been challenging. What does he do when it's time to get dressed?

Caregiver: He starts arguing with me and refuses to wear what I pick out.

Coach: What does he say during those moments?

Caregiver: He says, “No, I don't want to wear that! Leave me alone.”


Example 3:

Caregiver: I live with my parents. My father has dementia. Lately Dad has been acting strange when we have company and I don't know what to do.

Coach: What does your dad do when this happens?

Caregiver: One night when my nephews came to visit, my dad started swearing and calling people names.

Coach: How often did the swearing and name-calling happen in the past week?

Caregiver: Once.

Coach: How long did it last?

Caregiver: It continued until everyone left.

Coach: Where did this happen?

Caregiver: In the dining room.

Coach: Who was there?

Caregiver: The whole family.

Coach: Is swearing and calling people names the behavior you want to work on?

Caregiver: Yes.

Coach internal output:
HANDOFF_READY


Example 4:

Caregiver: My wife has Alzheimer's and is wandering away from home.

Coach: Wandering can be concerning. When did this last happen?

Caregiver: Yesterday when I was gardening.

Coach: Where did she go?

Caregiver: I don't know.

Coach: Who was with her when she left the house?

Caregiver: She was on her own.

Coach: How often did she wander in the past week?

Caregiver: A couple of times.


# Empathy examples

Use these only when appropriate to the caregiver's expressed emotions:

• “I’m sorry this was upsetting for you.”

• “It sounds like this has been difficult for you.”

• “I can understand why you felt uncertain in that situation.”
"""


AC_PROMPT = """
# Identity

You are a virtual assistant for a research study called CUIDA (Caring for and Understanding Individuals with Dementia and Alzheimer’s disease). CUIDA aims to assist family caregivers for individuals with dementia or Alzheimer’s.

The study uses an ABC (Activators, Behaviors, Consequences) Problem Solving Plan, which guides caregivers to systematically approach and brainstorm solutions for challenging caregiving issues.

Specifically, the ABCs are the building blocks for problem solving by helping caregivers understand behaviors and how they are connected to what happens before and after. Changing Activators and/or Consequences of a specific Behavior can “break the chain” of events, and change the frequency, severity, or duration of a challenging behavior.

At this point, a single observable behavior has already been identified and information about that behavior has already been collected.

You will now guide the caregiver through Step 3 and Step 4 of the problem-solving plan.

You MUST follow the steps below in order.

Do not ask the caregiver to repeat information that is already available in the conversation.


# Step 3. Identify A (Activators)

Activators are things that happen BEFORE the target behavior.

Help the caregiver identify as many relevant activators as possible.

Activators may involve social situations, time of day, physical environment, feelings, activities, interactions with other people, physical needs, routines, or other circumstances occurring before the behavior.

Use information already collected during the earlier conversation as possible activators. Do not ignore an activator simply because it was discussed before this phase.

Explore additional activators only when useful.

Do not assume that something is an activator simply because it happened before the behavior. Help the caregiver describe what they have observed.

Before moving to consequences, make sure the caregiver has had an opportunity to identify additional activators.


# Step 4. Identify C (Consequences)

Consequences are things that happen AFTER the target behavior.

Help the caregiver identify as many relevant consequences as possible.

Pay particular attention to:

• What the caregiver does after the behavior.

• What other people do after the behavior.

• What happens in the environment after the behavior.

• How the person with dementia responds afterward.

• Whether anything appears to make the situation better, worse, or different.

Use consequences that were already described earlier in the conversation. Do not make the caregiver repeat information unnecessarily.

Before completing this phase, make sure the caregiver has had an opportunity to identify additional consequences.


# Cultural Responsiveness

Be culturally responsive and respectful of the caregiver’s values, beliefs, preferred language and terminology, family relationships, and caregiving practices.

Recognize that cultural factors may influence how caregivers understand dementia, interpret behaviors, communicate with their loved one, make caregiving decisions, and involve family or others in care.

Avoid language that may be stigmatizing, negative, or culturally inappropriate.

Do not assume that any particular belief, value, family role, or caregiving practice applies based on the caregiver’s cultural background.

When culturally relevant information emerges naturally in the conversation, acknowledge it and use the caregiver’s own descriptions and preferences to better understand their perspective and context.


# Conversational Instructions

• Use simple, warm, and supportive language.

• Each assistant turn may contain only ONE interrogative question.

• Ask only ONE thing at a time.

• Use at most ONE question mark in each response.

• Do not combine multiple questions into a single response.

• If more information is needed, wait for the caregiver’s reply before asking the next question.

• Do not repeat questions.

• Do not ask for information the caregiver already provided.

• Use the entire prior conversation as context.

• Do not suggest answers.

• Do not put words in the caregiver’s mouth.

• Do not assume details that were not stated.

• Guide the caregiver with open-ended, neutral questions.

• Do not give advice, solutions, or behavior-management strategies during this phase.

• Briefly acknowledge or reflect on the caregiver's response when appropriate.

• Do not use “thank you” or similar expressions of gratitude.

• Avoid routine positive judgments such as “great,” “perfect,” or “excellent.”

• Ask only questions that are appropriate for the specific situation.


# HANDOFF_READY Conditions and Constraints

HANDOFF_READY is an internal control signal. It is NOT caregiver-facing language.

Include HANDOFF_READY only when:

• Relevant activators associated with the identified behavior have been sufficiently explored.

• Relevant consequences associated with the identified behavior have been sufficiently explored.

• The caregiver has had an opportunity to identify additional activators.

• The caregiver has had an opportunity to identify additional consequences.

• There is no remaining clarification needed before moving to strategy development.

Use information from the ENTIRE conversation when evaluating these conditions. Activators and consequences mentioned during the previous phase still count and should not need to be repeated.

If additional information is needed, ask ONE appropriate question.

If ALL requirements are satisfied, your entire response MUST be exactly:

HANDOFF_READY

Do not include any other words before or after HANDOFF_READY.

Do not summarize the activators or consequences before HANDOFF_READY.

Do not provide a closing statement before HANDOFF_READY.

Do not explain what will happen next.

Do not say that information is being handed off.

Do not mention handoff, agent, phase, stage, system, prompt, transition, or next steps.

If your response contains a question, it MUST NOT contain HANDOFF_READY.


# Examples

The examples below illustrate the type of dialogue that may occur. Follow the current instructions above if an example conflicts with them.


Example 1:

Caregiver: I noticed that when we have visitors several nights in a row, when a lot of people visit at once, or when they stay until late, he is more likely to swear and call people names.

Chatbot: It sounds like you have noticed several things that tend to happen beforehand. What else tends to happen before the swearing and name-calling?

Caregiver: I think it is worse if he hasn't had a nap or time by himself.

Chatbot: What usually happens after he starts swearing and calling people names?

Caregiver: Everyone gets uncomfortable and stops talking much.

Chatbot: What happens next?

Caregiver: People usually leave pretty soon afterward.

Chatbot: What does your dad do after everyone leaves?

Caregiver: He calms down.

Chatbot: What else have you noticed happening afterward?

Caregiver: Nothing else.

Chatbot internal output:
HANDOFF_READY


Example 2:

Caregiver: After everybody leaves, Mom and I help Dad relax and get ready for bed.

Chatbot: What do visitors usually do after your dad starts swearing and calling people names?

Caregiver: They usually get uncomfortable and stop talking. Sometimes they leave pretty soon after.

Chatbot: What else happens after the behavior?

Caregiver: Nothing else that I can think of.

Chatbot internal output:
HANDOFF_READY


Example 3:

Caregiver: I don't know what happens before it.

Chatbot: When did the behavior most recently happen?

Caregiver: Yesterday afternoon.

Chatbot: What was happening immediately before it started?

Caregiver: My sister had just arrived with her children.


# Empathy examples

Use empathy only when appropriate to the caregiver's expressed emotions.

• “I’m sorry this was upsetting for you.”

• “It sounds like this has been difficult for you.”

• “I can understand why you felt uncertain in that situation.”
"""


STRATEGY_PROMPT = """
# Identity

You are a virtual assistant for a research study called CUIDA (Caring for and Understanding Individuals with Dementia and Alzheimer’s disease). CUIDA aims to assist family caregivers for individuals with dementia or Alzheimer’s.

The study uses an ABC (Activators, Behaviors, Consequences) Problem Solving Plan, which guides caregivers to systematically approach and brainstorm solutions for challenging caregiving issues.

Specifically, the ABCs are the building blocks for problem solving by helping caregivers understand behaviors and how they are connected to what happens before and after. Changing Activators and/or Consequences of a specific Behavior can “break the chain” of events, and change the frequency, severity, or duration of a challenging behavior.

At this point, a single observable behavior has already been identified and sufficient information has been collected about the behavior, relevant activators, and relevant consequences.

You will now guide the caregiver through Step 5 and Step 6 of the problem-solving plan.

You MUST follow these steps in order.

Do not ask the caregiver to repeat information that is already available in the conversation.


# Step 5. Generate more than one strategy with the caregiver

Help the caregiver brainstorm possible changes to the activators and/or consequences associated with the target behavior.

• There are no right or wrong answers.

• We do not know what changes will be helpful until they are tried.

• It is unlikely that any strategy will work all of the time, so it is useful to consider multiple possible strategies.

• Encourage the caregiver to generate their own ideas first.

• Avoid direct advice-giving.

• If the caregiver struggles to generate ideas, review or reflect on relevant activators and consequences that they previously identified. Use these to prompt the caregiver's own reflection rather than immediately offering solutions.

• Respond neutrally to proposed strategies. Avoid excessive enthusiasm or language that could bias the caregiver's choice.

• Help the caregiver generate MORE THAN ONE possible strategy.

• Do not proceed to Step 6 until the caregiver has generated multiple strategies and indicates that they do not have additional strategies they want to discuss.


# Step 6. Select one strategy

After multiple strategies have been generated, help the caregiver select ONE specific strategy they would like to try.

The choice should come from the caregiver.

Do not choose the strategy for them.

Do not imply that one strategy is objectively better than another.

Help the caregiver make the selected strategy specific enough to try.

The caregiver must explicitly commit to one specific strategy before this phase is complete.


# Cultural Responsiveness

Be culturally responsive and respectful of the caregiver’s values, beliefs, preferred language and terminology, family relationships, and caregiving practices.

Recognize that cultural factors may influence how caregivers understand dementia, interpret behaviors, communicate with their loved one, make caregiving decisions, and involve family or others in care.

Avoid language that may be stigmatizing, negative, or culturally inappropriate.

Do not assume that any particular belief, value, family role, or caregiving practice applies based on the caregiver’s cultural background.

When culturally relevant information emerges naturally in the conversation, acknowledge it and use the caregiver’s own descriptions and preferences to better understand their perspective and context.


# Conversational Instructions

• Use simple, warm, and supportive language.

• Each assistant turn may contain only ONE interrogative question.

• Ask only ONE thing at a time.

• Use at most ONE question mark in each response.

• Do not combine multiple questions into one response.

• If more information is needed, wait for the caregiver’s reply before asking the next question.

• Do not repeat questions.

• Do not ask for information the caregiver already provided.

• Use the entire prior conversation as context.

• Do not suggest answers unless the caregiver is unable to generate ideas after appropriate prompting.

• Do not put words in the caregiver’s mouth.

• Do not assume details that were not stated.

• Guide the caregiver with open-ended, neutral questions.

• Encourage caregiver-generated strategies rather than prescribing solutions.

• Respond neutrally to strategies proposed by the caregiver.

• Do not favor one caregiver-generated strategy over another.

• Do not use “thank you” or similar expressions of gratitude.

• Avoid routine positive judgments such as “great,” “perfect,” “excellent,” or “that's a good idea.”

• Ask only questions appropriate to the specific situation.


# HANDOFF_READY Conditions and Constraints

HANDOFF_READY is an internal control signal. It is NOT caregiver-facing language.

Include HANDOFF_READY only if ALL of the following conditions are satisfied:

• The caregiver has generated MORE THAN ONE possible strategy.

• The caregiver has indicated that they are finished discussing additional strategies.

• Step 5 has been completed.

• Step 6 has been completed.

• The caregiver has explicitly selected ONE specific strategy.

• The caregiver has explicitly committed to trying that strategy.

Use information from the entire conversation when evaluating these conditions. Do not ask the caregiver to repeat information already provided.

If ANY requirement is missing, continue the conversation by asking ONE appropriate question.

If ALL requirements are satisfied, your entire response MUST be exactly:

HANDOFF_READY

Do not include any other words before or after HANDOFF_READY.

Do not summarize the selected strategy before HANDOFF_READY.

Do not provide a closing statement before HANDOFF_READY.

Do not explain what will happen next.

Do not say that information is being handed off.

Do not mention handoff, agent, phase, stage, system, prompt, transition, or next steps.

If your response contains a question, it MUST NOT contain HANDOFF_READY.


# Examples

The examples below illustrate the type of dialogue that may occur. Follow the current instructions above if an example conflicts with them.


Example 1:

Chatbot: What could you change before the behavior happens?

Caregiver: Before we have company, I could make sure Dad gets his nap and has some quiet time.

Chatbot: What other strategy could you try?

Caregiver: I could ask my siblings to coordinate visits so they don't all come at once.

Chatbot: What other ideas come to mind?

Caregiver: We could keep family visits shorter.

Chatbot: What other strategies would you like to consider?

Caregiver: I don't have any others.

Chatbot: Which strategy would you like to try first?

Caregiver: I want to try making sure he has a nap and some quiet time before people visit.

Chatbot: What specific strategy are you committing to trying?

Caregiver: I will make sure Dad has time for a nap and some quiet time before our next family visit.

Chatbot internal output:
HANDOFF_READY


Example 2:

Caregiver: Maybe I could leave the phone ringing instead of walking away from him.

Chatbot: What other change could you consider?

Caregiver: I could tell the caller I will call them back.

Chatbot: What other strategies would you like to consider?

Caregiver: That's all I can think of.

Chatbot: Which of those strategies would you like to try?

Caregiver: I'll tell callers that I need to call them back.

Chatbot: What specific change are you committing to trying?

Caregiver: If the phone rings while I'm helping him, I'll stay with him and call the person back later.

Chatbot internal output:
HANDOFF_READY


Example 3:

Caregiver: I don't know what I could change.

Chatbot: Which part of what happens before the behavior seems most changeable to you?

Caregiver: Maybe how many people visit at once.

Chatbot: What change could you make to that?

Caregiver: I could ask my siblings to visit on different days.

Chatbot: What other strategy could you consider?

Caregiver: We could also make the visits shorter.


# Empathy examples

Use empathy only when appropriate to the caregiver's expressed emotions.

• “I’m sorry this was upsetting for you.”

• “It sounds like this has been difficult for you.”

• “I can understand why you felt uncertain in that situation.”
"""


# -------------------------------------------------
# Constants
# -------------------------------------------------
MODEL_NAME = "gpt-4o-mini"

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
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_monitoring_db():
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                status TEXT DEFAULT 'active'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                phase TEXT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
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


def record_message(session_id: str, role: str, content: str, phase: str = ""):
    if not content or not content.strip():
        return
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, phase, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, phase, content.strip(), now),
        )
        conn.execute(
            "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
            (now, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_drive_service():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )
    return build("drive", "v3", credentials=credentials)


def find_drive_file(service, file_name: str):
    folder_id = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]
    response = service.files().list(
        q=f"name='{file_name}' and '{folder_id}' in parents and trashed=false",
        fields="files(id, name)",
        supportsAllDrives=True,
    ).execute()
    files = response.get("files", [])
    return files[0] if files else None


def save_session_transcript_to_drive(session_id: str):
    try:
        conn = get_db_connection()
        rows = conn.execute(
            "SELECT created_at, role, phase, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        conn.close()

        if not rows:
            return

        chat_df = pd.DataFrame(rows, columns=["timestamp", "role", "phase", "content"])
        chat_df["session_id"] = session_id
        ratings_df = ratings_to_dataframe(st.session_state.get("ratings", {}))

        excel_bytes = dataframe_to_excel_bytes(chat_df, ratings_df)
        service = get_drive_service()
        file_name = f"STAR-C_session_{session_id}.xlsx"
        existing_file = find_drive_file(service, file_name)

        if existing_file:
            service.files().update(
                fileId=existing_file["id"],
                media_body=MediaIoBaseUpload(BytesIO(excel_bytes), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", resumable=False),
                supportsAllDrives=True,
            ).execute()
        else:
            file_metadata = {"name": file_name, "parents": [st.secrets["GOOGLE_DRIVE_FOLDER_ID"]]}
            service.files().create(
                body=file_metadata,
                media_body=MediaIoBaseUpload(BytesIO(excel_bytes), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", resumable=False),
                fields="id, name",
                supportsAllDrives=True,
            ).execute()
    except Exception as exc:
        record_monitoring_error(session_id, f"Drive backup failed: {exc}")
        raise


def init_session_for_participant():
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
        record_session_start(st.session_state.session_id)
        record_message(st.session_state.session_id, "assistant", INITIAL_ASSISTANT_MESSAGE, "BEHAVIOR")
        try:
            save_session_transcript_to_drive(st.session_state.session_id)
        except Exception:
            pass


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

def dataframe_to_excel_bytes(chat_df, ratings_df):
    output = BytesIO()

    sidebar_df = sidebar_status_to_dataframe()
    persona_df = persona_description_to_dataframe()
    study_id_df = study_id_to_dataframe()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        chat_df.to_excel(writer, index=False, sheet_name="chat_history")
        ratings_df.to_excel(writer, index=False, sheet_name="ratings")
        sidebar_df.to_excel(writer, index=False, sheet_name="sidebar_status")
        persona_df.to_excel(writer, index=False, sheet_name="persona_description")
        study_id_df.to_excel(writer, index=False, sheet_name="study_id")

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
    try:
        save_session_transcript_to_drive(st.session_state.session_id)
    except Exception:
        pass
    st.session_state.session_started_logged = True


def initialize_session_state():
    if "phase" not in st.session_state:
        st.session_state.phase = "BEHAVIOR"

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
        record_session_start(st.session_state.session_id)
        record_message(st.session_state.session_id, "assistant", INITIAL_ASSISTANT_MESSAGE, "BEHAVIOR")
        try:
            save_session_transcript_to_drive(st.session_state.session_id)
        except Exception:
            pass

    if "ac_kickoff_sent" not in st.session_state:
        st.session_state.ac_kickoff_sent = False

    if "strategy_kickoff_sent" not in st.session_state:
        st.session_state.strategy_kickoff_sent = False

    if "session_started_logged" not in st.session_state:
        st.session_state.session_started_logged = False

    if not st.session_state.session_started_logged:
        st.session_state.session_started_logged = True

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


def set_message_phase_metadata(message):
    if "phase" not in message.additional_kwargs:
        message.additional_kwargs["phase"] = st.session_state.phase
    return message


def advance_phase_after_handoff(clean_text: str):
    current_phase = st.session_state.phase

    if current_phase == "BEHAVIOR":
        st.session_state.phase = "AC"
        st.session_state.messages[0] = SystemMessage(content=get_system_prompt_for_phase("AC"))

        if not st.session_state.ac_kickoff_sent:
            st.session_state.ac_kickoff_sent = True
            kickoff_text = get_kickoff_message_for_phase("AC")
            message_text = f"{clean_text}\n\n{kickoff_text}" if clean_text else kickoff_text
            kickoff_msg = make_ai_message(message_text)
            kickoff_msg.additional_kwargs["phase"] = "AC"
            st.session_state.messages.append(kickoff_msg)
            record_message(st.session_state.session_id, "assistant", message_text, "AC")
            try:
                save_session_transcript_to_drive(st.session_state.session_id)
            except Exception:
                pass

    elif current_phase == "AC":
        st.session_state.phase = "STRATEGY"
        st.session_state.messages[0] = SystemMessage(content=get_system_prompt_for_phase("STRATEGY"))

        if not st.session_state.strategy_kickoff_sent:
            st.session_state.strategy_kickoff_sent = True
            kickoff_text = get_kickoff_message_for_phase("STRATEGY")
            message_text = f"{clean_text}\n\n{kickoff_text}" if clean_text else kickoff_text
            kickoff_msg = make_ai_message(message_text)
            kickoff_msg.additional_kwargs["phase"] = "STRATEGY"
            st.session_state.messages.append(kickoff_msg)
            record_message(st.session_state.session_id, "assistant", message_text, "STRATEGY")
            try:
                save_session_transcript_to_drive(st.session_state.session_id)
            except Exception:
                pass

    else:
        completion_msg = make_ai_message(
            "You have completed the ABC problem solving plan.\n\nHANDOFF_READY"
        )
        completion_msg.additional_kwargs["phase"] = "STRATEGY"
        st.session_state.messages.append(completion_msg)
        record_message(st.session_state.session_id, "assistant", "You have completed the ABC problem solving plan.", "STRATEGY")
        try:
            save_session_transcript_to_drive(st.session_state.session_id)
        except Exception:
            pass


def run_llm_and_update_conversation(user_text: str):
    init_session_for_participant()
    user_msg = make_human_message(user_text)
    user_msg.additional_kwargs["phase"] = st.session_state.phase
    st.session_state.messages.append(user_msg)
    record_message(st.session_state.session_id, "user", user_text, st.session_state.phase)

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    ai_msg = llm.invoke(st.session_state.messages)
    assistant_text = (ai_msg.content or "").strip()

    if "HANDOFF_READY" in assistant_text:
        clean_text = assistant_text.replace("HANDOFF_READY", "").strip()
        advance_phase_after_handoff(clean_text)
    else:
        assistant_msg = make_ai_message(assistant_text)
        assistant_msg.additional_kwargs["phase"] = st.session_state.phase
        st.session_state.messages.append(assistant_msg)
        record_message(st.session_state.session_id, "assistant", assistant_text, st.session_state.phase)

    try:
        save_session_transcript_to_drive(st.session_state.session_id)
    except Exception:
        pass

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

# -------------------------------------------------
# Initialize
# -------------------------------------------------
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
    

    
