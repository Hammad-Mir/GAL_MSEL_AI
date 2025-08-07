import os
import re
import uuid
import json
import redis
import requests
import datetime as dt
from typing import List
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_redis import RedisChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

load_dotenv()
# UNIT_API_BASE = "http://192.168.1.29:3001/api/ai-scenario-generation/hierarchy/unit"
UNIT_API_BASE =  os.getenv("UNIT_API_BASE")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

print(REDIS_URL)
print(UNIT_API_BASE)
CACHE_TTL_SECONDS = 60 * 60 * 24   # 24h TTL by default

app = FastAPI()

# --- PROMPTS ---
collection_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are ALERTSim AI, an expert exercise planner for emergency management scenarios.

*Conversationally* ask the user to provide all the following information from the user IN NATURAL LANGUAGE (not as a form, not as Field: Value pairs), one at a time, in a friendly and helpful manner:
- Asset Name, Asset Type, Asset Location, Ownership/Operator Name, Workforce Size and Shift Structure
- Primary Function, Key Processes/Operations Onsite, Presence of Hazardous Materials (yes/no and types)
- Onsite Response Equipment, Communication Systems Used
- Primary Risk Scenarios to Simulate, Local Environmental Conditions, Proximity to Sensitive or Populated Areas (yes/no)
- Emergency Response Framework, Preferred Complexity, Targeted Trainee Roles, Controllers and Inject Roles Needed

Make sure to request all the fields specified and ask follow-up questions based on the user's previous answers to clarify or gather more details.
If the user provides information out of order or piecemeal, deduce and track which details are complete.

When you believe you have all required information, OUTPUT A CLEAR summary in the below list format and ASK:
"Is all of this correct and complete? If yes, please type 'ok'. If not, let me know what needs to be changed or added."

summary format:
1. Core Asset Details
    • Asset Name:
    • Asset Type: (e.g., Refinery, Offshore Platform, Warehouse, Airport, Hospital, Chemical Plant)
    • Asset Location: (Region and Country)
    • Ownership/Operator Name:
    • Workforce Size and Shift Structure:
2. Operational Profile
    • Primary Function of the Asset:
    • Key Processes/Operations Onsite:
    • Presence of Hazardous Materials: (Yes/No + General Type)
3. Emergency Setup
    • Response Equipment Onsite: (e.g., fire extinguishers, spill kits, emergency comms)
    • Communication Systems Used: (e.g., VHF radio, satellite, mobile phones)
4. Environmental and Risk Context
    • Primary Risk Scenarios to Simulate: (Select: fire, oil spill, medical emergency, security breach, natural disaster)
    • Local Environmental Conditions: (e.g., coastal, desert, industrial zone)
    • Proximity to Sensitive or Populated Areas: (Yes/No)
5. Simulation Preferences
    • Emergency Response Framework Used: (e.g., ICS, MEMIR, Bronze-Silver-Gold Command, Local ERP)
    • Preferred Complexity Level: (Basic / Intermediate / Complex)
    • Targeted Trainee Roles: (e.g., Incident Commander, Planning Chief)
    • Controllers and Inject Roles Needed: Coast Guard, Fisherman Representative, Environmental Regulator, Company HQ Observer

Wait for explicit user confirmation before moving on.
"""),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

msel_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are ALERTSim AI, an expert scenario designer.
Using ONLY the user-confirmed asset and scenario information below, generate a detailed, stepwise Master Scenario Events List (MSEL) for emergency response training.
Write in English. STRICTLY AVOID tables or markdown; output a clear, plain-text, numbered list of events.
Each event must include:
- Event Number
- Scenario Time
- Event Type
- Inject Mode
- From
- To
- Message
- Expected Participant Response
- Objectives/Capabilities Tested
- Notes

Use realistic, escalation-aware, framework-aligned scenario progressions. Do NOT add fantastical or speculative details. If any details are missing, focus only on the provided scenario context.
User-verified details:

"""),
    ("human", "{collected_data}")
])

# --- MEMORY ---
# class SimpleInMemoryHistory(BaseChatMessageHistory):
#     def __init__(self):
#         self.messages = []
#     def add_message(self, message):
#         self.messages.append(message)
#     def add_messages(self, messages):
#         self.messages.extend(messages)
#     def clear(self):
#         self.messages = []

# _history_store = {}

# def get_session_history(session_id: str):
#     if session_id not in _history_store:
#         _history_store[session_id] = SimpleInMemoryHistory()
#     return _history_store[session_id]

# confirmed_summaries = {}

def store_summary(session_id: str, summary: str):
    """
    Store the confirmed summary text for a session in Redis.
    """
    redis_cli.setex(f"summary:{session_id}", CACHE_TTL_SECONDS, summary)

def get_summary(session_id: str) -> str | None:
    """
    Fetch the confirmed summary text from Redis if present.
    """
    return redis_cli.get(f"summary:{session_id}")

redis_cli = redis.from_url(REDIS_URL, decode_responses=True)

# ---- Helper serialization ----
def msg_to_dict(m: BaseMessage):
    return {"role": m.type, "content": m.content}

def dict_to_msg(d: dict) -> BaseMessage:
    if d["role"] == "ai":
        return AIMessage(content=d["content"])
    if d["role"] == "system":
        return SystemMessage(content=d["content"])
    return HumanMessage(content=d["content"])

# ---- Main History Store ----
class RedisChatMessageHistory(BaseChatMessageHistory):
    """
    Chat history stored in Redis List.
    Each message is a JSON blob: {"role": ..., "content": ...}
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._rkey = f"chat:{session_id}"

    @property
    def messages(self) -> List[BaseMessage]:
        raw = redis_cli.lrange(self._rkey, 0, -1)
        return [dict_to_msg(json.loads(r)) for r in raw]

    def add_message(self, message: BaseMessage):
        redis_cli.rpush(self._rkey, json.dumps(msg_to_dict(message)))
        redis_cli.expire(self._rkey, CACHE_TTL_SECONDS)

    def add_messages(self, messages: List[BaseMessage]):
        pipe = redis_cli.pipeline()
        for msg in messages:
            pipe.rpush(self._rkey, json.dumps(msg_to_dict(msg)))
        pipe.expire(self._rkey, CACHE_TTL_SECONDS)
        pipe.execute()

    def clear(self):
        redis_cli.delete(self._rkey)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(session_id)

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
collection_chain = collection_prompt | llm | StrOutputParser()
msel_chain = msel_prompt | llm | StrOutputParser()

history_chain = RunnableWithMessageHistory(
    collection_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

def is_confirmation_prompt(response: str):
    return bool(re.search(r"is all of this correct.*type ['\"]?ok['\"]?", response, re.IGNORECASE))

def extract_summary(response: str):
    pattern = r"(1\.\s+\*?\*?Core Asset Details.*?)(?:\n+)?Is all of this correct"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


class StartSession(BaseModel):
    unit_id: int

class ChatRequest(BaseModel):
    session_id: str
    input: str

class GenerateMSELRequest(BaseModel):
    session_id: str

# def format_unit_info(unit_data):
#     # Extract info from nested dict into friendly natural text for the bot
#     org = unit_data.get("asset", {}).get("organisation", {})
#     asset = unit_data.get("asset", {})
#     unit = unit_data
#     msg = (
#         f"Organisation: '{org.get('name', 'Unknown')}' ({org.get('description', '...')}). "
#         f"HQ Location: {org.get('hqLocation', 'Unknown')} in {org.get('country', 'Unknown')}. "
#         f"Asset: '{asset.get('name', 'Unknown')}', a {asset.get('type', 'Unknown')} asset located in {asset.get('country', 'Unknown')}. "
#         f"Asset Coordinates: {asset.get('coordinates', 'Unknown')}. "
#         f"Unit of Interest: '{unit.get('name', 'Unknown')}' (type: {unit.get('type', 'Unknown')})."
#         "\nLet's begin scenario setup."
#     )
#     return msg

def format_unit_info(unit_data: dict) -> str:
    """
    Turn the unit ingest structure into a compact, human-readable summary.

    Expected structure (truncated):
    {
        "unitInfo": {...},
        "teamsAssigned": [...]
    }
    """

    # ---------- 1) Core sections ----------
    unit   = unit_data.get("unitInfo", {})
    asset  = unit.get("asset", {})
    org    = asset.get("organisation", {})
    teams  = unit_data.get("teamsAssigned", [])

    # ---------- 2) Organisation ----------
    org_text = (
        f"Organisation/Owner: "
        f"'{org.get('name', 'Unknown')}' – "
        f"{org.get('description', 'No description')}. "
        f"HQ: {org.get('hqLocation', 'Unknown location')}, "
        f"{org.get('country', 'Unknown country')}."
    )

    # ---------- 3) Asset ----------
    asset_text = (
        f"Asset: "
        f"'{asset.get('name', 'Unknown')}', a "
        f"{asset.get('type', 'Unknown type')} asset in "
        f"{asset.get('country', 'Unknown country')} "
        f"(coords: {asset.get('coordinates', 'Unknown')})."
    )

    # ---------- 4) Unit ----------
    unit_text = (
        f"Unit: "
        f"'{unit.get('name', 'Unknown')}'. "
        f"Key Processes {unit.get('type', 'Unknown')}."
    )

    # ---------- 5) Teams / users ----------
    if teams:
        team_lines = []
        for t in teams:
            users = t.get("users", [])
            user_list = []
            for u in users:
                full_name = f"{u.get('firstName','')} {u.get('lastName','')}".strip()
                role      = u.get("roleName") or u.get("roleId") or ""
                user_list.append(f"{full_name}{f' [{role}]' if role else ''}")
            members = ", ".join(user_list) if user_list else "No users"
            team_lines.append(f"• Team: '{t.get('name','Unnamed')}' \n{members}")
        teams_text = "Teams Assigned for Training:\n" + "\n".join(team_lines)
    else:
        teams_text = "No teams have been assigned."

    # ---------- 6) Build final message ----------
    msg = "\n".join([org_text, asset_text, unit_text, teams_text, "Let's begin scenario setup."])

    return msg

@app.post("/start-session")
async def start_session(unit_id: StartSession):
    session_id = str(uuid.uuid4())
    # confirmed_summaries[session_id] = None
    store_summary(session_id, "")

    # --- Fetch unit hierarchy ---
    try:
        # unit_api_url = f"http://192.168.1.29:3001/api/ai-scenario-generation/hierarchy/unit/{unit_id}"
        # async with httpx.AsyncClient(timeout=10) as client:
        #     resp = await client.get(unit_api_url)
        #     resp.raise_for_status()
        #     unit_data = resp.json()
        unit_response = requests.get(f"{UNIT_API_BASE}/{unit_id.unit_id}")
        unit_response.raise_for_status()
        unit_data = unit_response.json()

        # user_msg = format_unit_info(unit_data)
    except Exception as e:
        return {
            "session_id": session_id,
            "initial_agent_message": "[Error]: Failed to fetch unit info.",
            "error": str(e),
        }
    
    user_msg = format_unit_info(unit_data)
    
    # # Provide initial prompt from backend – NOT from LLM (avoids Gemini blank error)
    # initial_agent_message = "Talk to me about your facility and simulation needs (in your own words)."
    # return {
    #     "session_id": session_id,
    #     "initial_agent_message": initial_agent_message,
    # }

    # --- Pass user_msg as the first message to chat chain ---
    # This triggers the LLM as if the user had written this as their turn-1 message.
    response = history_chain.invoke(
        {"input": user_msg},
        config={"configurable": {"session_id": session_id}}
    )

    # The LLM's reply is now your true "initial_agent_message"
    return {
        "session_id": session_id,
        # "user_message": user_msg,
        "response": response,
    }

@app.post("/chat")
def chat(req: ChatRequest):
    response = history_chain.invoke(
        {"input": req.input},
        config={"configurable": {"session_id": req.session_id}}
    )
    awaiting_confirmation = False
    summary = None
    if is_confirmation_prompt(response):
        awaiting_confirmation = True
        summary = extract_summary(response)
        if summary:
            # confirmed_summaries[req.session_id] = summary
            store_summary(req.session_id, summary)
    # if req.input.lower() == "ok" and confirmed_summaries.get(req.session_id) is not None:
    if req.input.lower() == "ok" and get_summary(req.session_id) is not "":
        return{
            "session_id": req.session_id,
            # "response": {
            #     "message": "Msel generated sucessfully.",
            #     "msel":generate_msel(req.session_id)}
            "response": generate_msel(req.session_id),
        }
    else:
        return {
        "session_id": req.session_id,
        "response": response,
        # "awaiting_confirmation": awaiting_confirmation,
        # "summary": summary,
        }

# @app.get("/generate-msel/{session_id}")
def generate_msel(session_id: str):
    # summary = confirmed_summaries.get(session_id)
    summary = get_summary(session_id)
    if not summary or not summary.strip():
        return {"error": "No confirmed summary found for this session. Complete and confirm the scenario first."}
    msel = msel_chain.invoke({"collected_data": summary})
    return msel

# To run: uvicorn msel:app --reload
