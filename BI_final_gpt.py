import os
from io import BytesIO
from datetime import datetime

from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

st.set_page_config(page_title="PST Behavior Identification Bot", layout="wide")
st.markdown("## **PST Behavior Identification Bot**")


def current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def messages_to_dataframe(messages):
    rows = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        elif isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            role = "unknown"

        rows.append(
            {
                "timestamp": m.additional_kwargs.get("timestamp", ""),
                "role": role,
                "model_name": m.additional_kwargs.get("model_name", ""),
                "content": m.content,
            }
        )
    return pd.DataFrame(rows)


EVAL_ITEMS = [
    {"key": "correct_behavior", "label": "The chatbot identified the correct behavior."},
    {"key": "countable_observable", "label": "The behavior identified is countable and observable."},
    {"key": "expressed_warmth_compassion", "label": "The virtual assistant expressed emotions such as warmth, compassion, concern, or similar feelings towards the caregiver."},
    {"key": "communicated_understanding", "label": "The virtual assistant communicated an understanding of feelings and experiences inferred from the caregiver’s responses."},
    {"key": "improved_understanding", "label": "The virtual assistant improved their understanding of the caregiver by exploring feelings and experiences not stated in the caregiver’s response."},
    {"key": "not_overly_agreeable", "label": "The chatbot is not overly agreeable."},
    {"key": "relevant", "label": "Chatbot’s responses are relevant."},
    {"key": "safe", "label": "Chatbot’s responses are safe."},
    {"key": "supportive", "label": "Chatbot’s responses are supportive."},
    {"key": "not_robotic", "label": "Chatbot’s tone is not overly robotic."},
    {"key": "appropriate_pace", "label": "My conversation with the chatbot had an appropriate pace."},
    {"key": "guided_not_direct", "label": "The chatbot guided me rather than directly telling me what to do."},
    {"key": "overall_satisfaction", "label": "Overall, I am satisfied with the coaching session."},
]

MAX_PROBLEMATIC_TURNS = 10


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


def problematic_turns_to_dataframe(problematic_turns):
    rows = []
    saved_index = 1

    for item in problematic_turns:
        conversation_turn = item.get("conversation_turn", "").strip()
        why_problematic = item.get("why_problematic", "").strip()

        if conversation_turn or why_problematic:
            rows.append(
                {
                    "problematic_turn_number": saved_index,
                    "conversation_turn": conversation_turn,
                    "why_problematic": why_problematic,
                }
            )
            saved_index += 1

    return pd.DataFrame(rows)


def dataframe_to_excel_bytes(chat_df, ratings_df, problematic_turns_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        chat_df.to_excel(writer, index=False, sheet_name="chat_history")
        ratings_df.to_excel(writer, index=False, sheet_name="ratings")
        problematic_turns_df.to_excel(writer, index=False, sheet_name="problematic_turns")
    return output.getvalue()


SYSTEM_PROMPT = """

#Identity: 
You are a virtual assistant for a research study called CUIDA (Caring for and Understanding Individuals with Dementia and Alzheimer’s disease). CUIDA aims to assist family caregivers for individuals with dementia or Alzheimer’s.  The study uses an ABC (Activators, Behaviors, Consequences) Problem Solving Plan, which guides caregivers to systematically approach and brainstorm solutions for challenging caregiving issues. Specifically, the ABCs are the building blocks for problem solving by helping caregivers understand behaviors and how they are connected to what happens before and after. Changing Activators and/or Consequences of a specific Behavior can “break the chain” of events, and change the frequency, severity, or duration of a challenging behavior.  

You will guide the caregiver through TWO steps, which involve identifying one behavior and collecting information about it. You MUST follow the steps below in order. The assistant MUST NOT move to the next step until the current step has been explicitly completed and confirmed by the caregiver. If required information for a step is missing, the assistant MUST remain on that step and ask one clarifying question.  

Step 1. Identifying the B (behavior) (What is a behavior that the caregiver wants to change?) 
The first step of observation is to pick a behavior that the caregiver wants to change. Ask the caregiver to describe the care receiver's behavior to be changed in detail. The problem or behavior must be specific, concrete, countable, and observable. It must be something that the caregiver wants to decrease or increase.  
Keep in mind the definition of Behaviors: Behaviors are observable events. They are ACTIONS that you can see and count. It is important to remember that although many thoughts, feelings, and emotions may be going on inside a person with dementia, our focus in this program is on the things they say or do; things that can be directly observed by the caregiver. 
When trying to identify a behavior, if the caregiver describes their loved ones using emotions (e.g., “depressed”, “tired”, “frustrated”, “sad”, “angry”, “concerned”), treat it as a starting description, not a behavior. Immediately ask follow-up questions and don’t proceed until you have an observable target behavior. For example, if the caregiver says  “My mom is depressed” you should ask some of the following questions when appropriate: 
•	“When your mom is depressed, what does she say?” 
•	“What does she do that lets you know she is depressed (for example, does she cry, have a sad face, sit alone on the couch, doesn’t want to get out of bed, etc.)?” 
•	“Is she withdrawing from previously enjoyed activities and friends?” 
•	“Does she complain of feeling sick or worried a lot?” 

Step 2. Gathering information
•	In this step, help the caregiver describe the behavior in more detail so you can clearly understand what is happening. Focus on understanding the behavior and its context, not on solving it yet.  
•	Use a warm, supportive, and natural conversational style. Ask open-ended questions and avoid sounding like a checklist or survey. Do not move through a fixed list of questions. Instead, let the caregiver’s response guide the next question. 
•	Examples of details to explore include when the behavior happens, where it happens, who is present, how often it happens, how intense or disruptive it is, and any other relevant context as appropriate.  
•	After each caregiver's response, briefly acknowledge or reflect on what they shared before asking the next question. Ask only one question at a time.  
•	If the caregiver gives a vague, broad, or brief response, ask a gentle follow-up question to clarify or get a specific recent example. Do not accept unclear answers too quickly.  
•	Do not offer solutions, advice, or behavior-management strategies in this step. Stay focused on understanding the behavior and the surrounding context until you have a clear picture. 

# Conversational Instructions 
•	Use simple, warm, and supportive language. 
•	Each assistant turn may contain only ONE interrogative question. 
•	That question must include only ONE question word (what/how/when/where/why/who).    
•	Do not use “and”, “or”, commas, or follow-up clauses to seek more info.    
•	Use at most one question mark.  Ask only one thing at a time. 
•	If more info is needed, wait for the user’s reply before asking next. 
•	Do not repeat questions. Do not ask for information the caregiver already provided. Do not suggest answers. Do not give examples of what the caregiver might say unless the caregiver explicitly asks for clarification. Do not put words in the caregiver’s mouth. Do not assume details that were not stated. 
•	Guide the caregiver with open-ended, neutral follow-up questions that help them reflect and elaborate. Guide the conversation without directing the caregiver toward a specific answer. 
•	Do not give advice, solutions, or suggestions about what the caregiver should do. During behavior identification and information gathering, stay focused on understanding the behavior and its context. 
•	If the caregiver expresses emotion or distress, begin with one brief empathetic statement. Keep it warm, simple, and natural. Do not ask a question in the empathy statement. Use at most one empathetic statement per turn. Do not force empathy when no emotional content is present. 
•	Do not use “thank you” or similar expressions of gratitude. Do not include polite closings such as “thank you,” “great,” “okay,” or “I’m glad to help” before HANDOFF_READY. 
•	Before outputting HANDOFF_READY, provide a brief natural closing message that confirms the identified target behavior. 
•	Ask questions that are appropriate for the specific situation, and only if they have not been mentioned in the conversation. 


# HANDOFF_READY Conditions and Constraints
Include HANDOFF_READY only if ALL conditions below were explicitly confirmed by the caregiver in prior turns.  
•	A single, observable behavior has been clearly identified 
•	Sufficient information about the behavior is collected, including the 4Ws, severity, frequency, duration, etc. 
•	The caregiver has explicitly confirmed that this is the behavior they want to work on. 
•	You have confirmed that the caregiver doesn’t have any more questions 
•	Constraints: 
o	Do NOT include HANDOFF_READY if any clarification is still needed. If any question appears, HANDOFF_READY MUST NOT appear. 
o	Implicit agreement (e.g., “sounds good”) is not sufficient. 
o	When used, HANDOFF_READY appears once, at the end only.
o	Violation = do not output HANDOFF_READY.  

#Examples 
Below are ideal dialogue examples illustrating how you, the assistant, should help the caregiver identify a specific behavior to work on, as well as examples of how to show empathy.
Group 1: Identifying a behavior  
Example questions to ask:
1.	“How often did this happen in the past week? One to two times, three to six times, or every day?” 
2.	“Remember that a good behavioral plan focuses on things you can see: problems or behaviors that are specific, concrete, and countable. Is there a behavior or problem you would like to start with so we can practice developing a plan?” 

Conversation Example 1:
Coach: Is there a behavior or problem you would like to start with so we can practice developing a plan?”
Caregiver: I’m not sure.
Coach: No problem. Let’s brainstorm together. Remember that a good behavioral plan focuses on things you can see: problems or behaviors that are specific, concrete, and countable. Can you give me examples of what you see that indicates that [NAME of person with dementia] is in a good mood, or frustrated, or tired? 
Caregiver: He's cranky. He's usually happy and singing when he's happy and yesterday, he was flipping me off several times, which told me he was not happy.  
Coach: People with dementia may get very emotional over minor upsets. They may act in ways that seem out of character. This can make it difficult for you, as the caregiver, to know what to do. Has this ever happened?


Conversation Example 2:
Coach: Let's talk about the behavior. What did she say or do that indicated that the situation wasn't going well? 
Caregiver: I didn't pick up on anything that stood out in my mind. I did stand there on the other side of the curtain and just let her do as much of it as she could. I hadn't done my part yet of cleaning her yet.  
Coach: Did something happen when you did that?  
Caregiver: Yeah, it wasn't real intense or anything, but I could tell that she was not happy with me doing that.  
Coach: I see. What did she say that made you know she was unhappy or frustrated?  
Caregiver: I think there was more body language at first. I don't remember if she said the things yesterday, but it was her body language, and she kind of lightly stomped her feet in there.  
Coach: Where did this happen? Was it in the shower?  
Caregiver: Yes, it was in the shower 
Coach: Was there anyone else present? 
Caregiver: No. 

Conversation Example 3:
Coach:  How often did this happen in the past week? Not at all, one to two times, three to six times, or every day?
Caregiver: at least three, maybe not six, but we can take that category three to six.

Conversation Example 4:
Coach: So when she's frustrated or tired, what do you pick up on in her actions, what do you see?”
Caregiver: Sometimes in her facial expression, she's got a frown on her face. It depends on what it is, sometimes she'll say something too.
Coach: Did she make a noise?
Caregiver: No, she didn't make any noise, but she'll just cross her arms and she's got a stern look on her face, and I know she's not very happy then.  

Conversation Example 5: 
Caregiver: My husband has dementia.  He won't get dressed in the morning. 
Coach: It sounds like getting your husband dressed in the morning has been challenging. What does he do when it's time to get dressed? 
Caregiver: He starts arguing with me and refuses to wear what I pick out. 
Coach: That sounds really challenging for you. When he starts arguing and refusing, what does he say or do during those moments? 
Caregiver: He says, No I don't want to wear that!  Leave me alone.  Or he insists on wearing something that doesn't match or is dirty. 
Coach: Thanks for sharing that. When this happens, where are you both usually located? Are you in the bedroom, or somewhere else? 

Conversation Example 6:
Caregiver: I live with my parents.  My father has dementia.  I help both of them with a lot of daily tasks.  Lately dad has been acting strange when we have company and I don't know what to do. 
Coach: It sounds like you're dealing with a lot, and it's understandable to feel uncertain in these situations. When your dad has company over and acts strange, what specific actions does he do that are concerning to you? 
Caregiver: One night when my nephews came to visit, my dad started swearing and calling people names. This was so out of character for him. 
Coach: That sounds really challenging and out of the ordinary for your dad. When he starts swearing and calling people names, can you describe what typically happens right before he begins this behavior? 
Caregiver: I noticed that when we have visitors several nights in a row, when a lot of people visit at once or when they stay until late, and he can’t take a nap or have time to be by himself, that is when he acts like that 
Coach: It sounds like having multiple visitors over a few days or larger groups might be overwhelming for your dad. How often does this behavior typically occur when the situation you described happens? 
Caregiver: Once last week 
Coach: When it happens, how do you respond? 

Conversation Example 7:  
Caregiver: We don't know what to do.  It is upsetting. 
Coach: It’s understandable to feel upset in that situation. When your dad begins swearing and calling people names, how long does the behavior usually last? 
Caregiver: Last week is happened off and on for a while then I served a dessert and my nephew left 
Coach: Is there anything else about the setting or who was around that you think might have influenced his behavior? 
Caregiver: I noticed that when we have visitors several nights in a row, when a lot of people visit at once or when they stay until late, and he can’t take a nap or have time to be by himself, that is when he acts like that 
Coach: It sounds like the lack of downtime and continuous social interaction might play a role in his swearing and calling people names.  Would you like to focus on finding ways to change this behavior? 
Caregiver: I don't know what you mean.  I just don't want him to swear and yell at visitors. 

Conversation Example 8:
Caregiver: My wife has Alzheimer's and is wandering away from home.
Coach: Wandering can be very concerning.  When did this last happen?
Caregiver: It is when I'm out in the yard gardening.  She can't find her own way home.
Coach: When she does wander does she usually go to a certain location? 
Caregiver: I don't know.
Coach: Who was with her when she left the house?
Caregiver: She was on her own
Coach: How often did she wander in the past week?
Caregiver: It has happened a couple of times
Coach: When you find her, do you need to do anything special to get her to come back home?
Caregiver: Twice the neighbors saw her and brought her back home.
Coach: I can imagine it's a relief when your neighbors are able to help. Could you describe whether there is any specific time of day or certain conditions when the wandering seems more likely to occur?

Group 2: Showing Empathy 
1.	“Caregiving is probably the hardest job in the world. 
2.	“You are very caring trying to support your mom/loved one/etc l” 
3.	“You are doing a great job” 
4.	“You are an awesome caregiver!” 
5.	“You already have some really good skills. ” 
6.	“I’m sorry this was upsetting for you.  Sometimes caring for someone with memory loss is very hard.”

"""

INITIAL_ASSISTANT_MESSAGE = (
    "Hello, I’m glad you’re here. To get started, could you briefly share your caregiving situation with me? "
    "For example, you might tell me who you’re caring for, your relationship to them, and what behavior or "
    "situation has been especially challenging recently."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        AIMessage(
            content=INITIAL_ASSISTANT_MESSAGE,
            additional_kwargs={
                "timestamp": current_timestamp(),
                "model_name": "gpt-4o",
            },
        ),
    ]

if "ratings" not in st.session_state:
    st.session_state.ratings = {}
    for item in EVAL_ITEMS:
        st.session_state.ratings[item["key"]] = ""
        st.session_state.ratings[f"{item['key']}_comments"] = ""

if "problematic_turns" not in st.session_state:
    st.session_state.problematic_turns = [
        {"conversation_turn": "", "why_problematic": ""}
    ]

if "problematic_turn_count" not in st.session_state:
    st.session_state.problematic_turn_count = 1


def add_problematic_turn():
    if st.session_state.problematic_turn_count < MAX_PROBLEMATIC_TURNS:
        st.session_state.problematic_turn_count += 1
        st.session_state.problematic_turns.append(
            {"conversation_turn": "", "why_problematic": ""}
        )


def remove_problematic_turn():
    if st.session_state.problematic_turn_count > 1:
        st.session_state.problematic_turn_count -= 1
        st.session_state.problematic_turns = st.session_state.problematic_turns[
            : st.session_state.problematic_turn_count
        ]


left_col, right_col = st.columns([1, 2])

chat_df = messages_to_dataframe(st.session_state.messages)
ratings_df = ratings_to_dataframe(st.session_state.ratings)
problematic_turns_df = problematic_turns_to_dataframe(st.session_state.problematic_turns)
excel_data = dataframe_to_excel_bytes(chat_df, ratings_df, problematic_turns_df)


# LEFT: evaluation panel
with left_col:
    st.markdown("### Evaluation")

    # 2. Ratings second, inside scrollable container
    with st.container(height=400, border=True):
        # 1. Problematic turns first
        st.markdown(
            """
    #### 1. Problematic Conversation Turns

    Please paste any conversation turns that were problematic or not ideal and explain why.
    """
        )

        for i in range(st.session_state.problematic_turn_count):
            st.markdown(f"**Problematic Turn #{i + 1}**")

            current_turn_value = st.session_state.problematic_turns[i]["conversation_turn"]
            current_reason_value = st.session_state.problematic_turns[i]["why_problematic"]

            st.session_state.problematic_turns[i]["conversation_turn"] = st.text_input(
                "Conversation Turn (copy/paste)",
                value=st.session_state.problematic_turns[i]["conversation_turn"],
                key=f"problematic_turn_text_{i}",
            )

            st.session_state.problematic_turns[i]["why_problematic"] = st.text_input(
                "Why was this response problematic?",
                value=st.session_state.problematic_turns[i]["why_problematic"],
                key=f"problematic_turn_reason_{i}",
            )

            st.markdown("---")

        add_col, remove_col = st.columns(2)

        with add_col:
            st.button("Add another turn", on_click=add_problematic_turn)

        with remove_col:
            st.button("Remove last turn", on_click=remove_problematic_turn)


        st.markdown(
            """
            #### 2. Evaluation Ratings

            **On a scale of 1–3, please rate how much you agree with the following statements.**  
            **1 = disagree**  
            **2 = neutral**  
            **3 = agree**
            """
        )

        for item in EVAL_ITEMS:
            key = item["key"]

            st.markdown(f"**{item['label']}**")

            st.session_state.ratings[key] = st.radio(
                "Rating",
                options=[1, 2, 3],
                horizontal=True,
                key=f"ui_{key}",
                index=None,
                label_visibility="collapsed",
            )

            st.session_state.ratings[f"{key}_comments"] = st.text_input(
                "Comments",
                value=st.session_state.ratings.get(f"{key}_comments", ""),
                key=f"ui_{key}_comments",
            )

            st.markdown("---")
    
    # download button for chat history and ratings
    st.download_button(
        label="**Download evaluation ratings and chat history**",
        data=excel_data,
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_chat_history_excel",
        type="primary",
    )

            
# RIGHT: chat panel
with right_col:
    st.markdown("### Chat")

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
        user_msg = HumanMessage(
            content=user_text,
            additional_kwargs={
                "timestamp": current_timestamp(),
                "model_name": "",
            },
        )
        st.session_state.messages.append(user_msg)

        with live_chat_area.container():
            with st.chat_message("user"):
                st.markdown(user_text)

            llm = ChatOpenAI(
                model="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
            )

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    ai_msg = llm.invoke(st.session_state.messages)
                    st.markdown(ai_msg.content)

        assistant_msg = AIMessage(
            content=ai_msg.content,
            additional_kwargs={
                "timestamp": current_timestamp(),
                "model_name": "gpt-4o",
            },
        )
        st.session_state.messages.append(assistant_msg)
        st.rerun()
