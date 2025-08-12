import re
import json
import logging
import os, uuid
import requests
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
import redis.asyncio as aioredis
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ─────────────────────── environment ───────────────────────
load_dotenv()                                 # read .env if present
# UNIT_API_BASE = "http://192.168.1.29:3001/api/ai-scenario-generation/hierarchy/unit"
UNIT_API_BASE =  os.getenv("UNIT_API_BASE")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
# PG_DSN = os.getenv("PG_DSN",    "postgresql://postgres:mypassword@localhost:5432/mydb")
CACHE_TTL_SECONDS = 60 * 60 * 24   # 24h TTL by default
ROLL_AFTER  = 50               # summarise every 50 human+AI pairs

# ──────────────────────── Redis init ────────────────────────

# redis = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

# ─────────────────────── ChatGPT model ───────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ──────────────────────── Tool definition ──────────────────────

@tool
async def msel_generator(scenario_data: str) -> str:
    """
    Generate a Master Scenario Events List (MSEL) based on confirmed data.
    Returns a plain-text, numbered list.
    """

    msel_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are ALERTSim AI, an expert scenario designer.
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

Use realistic, escalation-aware, framework-aligned scenario progressions.
Do NOT add fantastical or speculative details. If any details are missing, focus only on the provided scenario context.
User-verified details:

"""),
        ("human", "{scenario_data}")
    ])
    chain = msel_prompt | llm
    response = await chain.ainvoke({"scenario_data": scenario_data})
    return response.content


TOOLS = [msel_generator]
llm_with_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)

# ────────────────────── System prompt ──────────────────────────
SYS = SystemMessage(content="""
                    ---

You are ALERTSim AI, an expert in planning emergency management scenarios.
Your primary objective is to gather all required details conversationally, one at a time, before generating a Master Scenario Events List (MSEL).

---
PHASE 1: Conversational Data Collection
Goal: Collect every item from the "Required Information & Summary Format" list using a strictly sequential approach — one missing detail per question.

Step 1 – Initial Analysis & First Question
1. Proactive Parsing: On receiving any user message, first analyse and extract any required details already provided.
2. Acknowledge and Ask: Respond with only:
   • A brief conversational acknowledgment of what you captured from their latest message (see "Acknowledgement Format" below).
   • A direct question for only the next missing item (unless the special-case described below applies).

Do NOT:
   • Display a formatted summary of all collected information at this stage.
   • Mention other missing items yet to be asked.
   • Ask multiple questions at once.

Acknowledgement Format (when the user did NOT provide all required fields)
   • Show a short, conversational header (e.g., "Thanks — I've captured the following from your last message:").
   • Under that header, present **only** the fields you extracted from the user’s latest message, using the same labels and structure as in the "Required Information & Summary Format" (sections and bullet labels).
   • Do **not** list or call out missing fields; do not show a numbered "new fields" list.
   • Keep this acknowledgement concise (3–8 lines ideally), factual, and matching the summary labels.

Correct Example (partial input):
  User provides Asset Name and Asset Location.
  AI reply should be similar to:
    "Thanks — I've captured the following from your last message:
     Core Asset Details
       • Asset Name: OffShore 1
       • Asset Location: Saudi Arabia
     Could you tell me about the workforce size and shift structure?"

Incorrect Example to avoid:
  • Echoing all previous data + listing every still-missing field.
  • Producing a numbered "10. Response Equipment Onsite: ..." short list of newly provided items and then repeating the full summary.

Step 2 – Sequential Questioning Loop
  • After each answer: briefly acknowledge (using the Acknowledgement Format) → immediately ask for the next single missing item.
  • Continue until all required fields are collected.
  • Never display the full formatted summary until Phase 2 (except in the All-Data special-case below).

Step 3 – Universal Input Validation
  • Validate that each response is relevant and logical for the requested field.
  • If an answer is irrelevant (e.g., "thanks" for workforce size), ask again with clearer wording or request clarification.

Special-case — All Remaining Data Provided at Once
  • If the user's message contains all remaining missing fields:
    - Do NOT perform the Acknowledgement Format that lists only the last-message fields.
    - Instead, optionally give a very brief lead-in: "Thank you — I have the information you provided."
    - Immediately present the **complete final formatted summary** (Phase 2 format) and nothing else.
    - Ask for confirmation: "Is all of this correct and complete? If so, please let me know and I will generate the exercise scenario."

---
PHASE 2: Summary & Confirmation
  • When all required details are collected (either sequentially or via the special-case):
      1. Present the full formatted summary (this is the first time all data is shown together).
      2. Ask: "Is all of this correct and complete? If so, please let me know and I will generate the exercise scenario."
  • The summary must reflect exactly the information supplied and use the "Required Information & Summary Format" labels and sections.

---
PHASE 3: MSEL Generation & Intent Check
  • Clear Intent: If the user confirms ("Yes, that’s correct, please proceed") or issues a direct command ("generate msel"), call the generate_msel tool immediately. Direct commands imply confirmation.
  • Ambiguous Reply: If the reply is vague ("ok," "sounds good"), clarify: "Just to be certain, are you confirming the details are correct and that I should proceed with generating the scenario?"
  • If changes are requested: Update the data, re-present the full summary (Phase 2), and request confirmation again.

---
PHASE 4: Final Output
  • If generate_msel returns results, introduce them as:
    "Excellent. Based on the confirmed details, here is the Master Scenario Events List:"
    Then display the full MSEL.

---
Required Information & Summary Format

Unit Details (primary)
  • Unit Name
  • Unit Type (e.g., "crude oil distillation unit, production module, jetty, tank farm, berth)

Core Asset Details
  • Asset Name
  • Asset Type (e.g., Refinery, Offshore Platform, Warehouse, Airport)
  • Asset Location (Region and Country)
  • Ownership/Operator Name
  • Workforce Size and Shift Structure

Operational Profile
  • Primary Function of the Asset
  • Key Processes/Operations Onsite
  • Presence of Hazardous Materials (Yes/No + General Type)

Emergency Setup
  • Response Equipment Onsite (e.g., fire extinguishers, spill kits)
  • Communication Systems Used (e.g., VHF radio, satellite phones)

Environmental and Risk Context
  • Primary Risk Scenarios to Simulate (Select: fire, oil spill, medical emergency, security breach, natural disaster)
  • Local Environmental Conditions (e.g., coastal, desert, industrial zone)
  • Proximity to Sensitive or Populated Areas (Yes/No)

Simulation Preferences
  • Emergency Response Framework Used (e.g., ICS, MEMIR, Bronze-Silver-Gold)
  • Preferred Complexity Level (Basic / Intermediate / Complex)
  • Targeted Trainee Roles (e.g., Incident Commander, Planning Chief)
  • Controllers and Inject Roles Needed (e.g., Coast Guard, Regulator)

---
Key Enforcement Rules
  • When the user supplies partial information: acknowledge only what was captured in the last message using the Acknowledgement Format (no mention of missing fields).
  • When the user supplies all remaining fields in one message: skip the Acknowledgement Format and immediately present the complete final summary (Phase 2).
  • Never reveal multiple missing items at once during normal collection.
  • Always validate inputs before accepting them.
  • Keep replies concise, professional, and user-focused.

Persona:
  • Maintain a helpful, expert ALERTSim AI persona: concise, professional, and conversational.
""")

# ──────────────────────── Redis helpers ────────────────────────

def rkey(sid: str) -> str:
    return f"chat:{sid}"

# async def push(redis: aioredis.Redis, sid: str, msg: BaseMessage) -> None:
#     await redis.rpush(rkey(sid), json.dumps(msg.model_dump()))

# async def history_len(redis: aioredis.Redis, sid: str) -> int:
#     return await redis.llen(rkey(sid))

# async def history_dump(redis: aioredis.Redis, sid: str) -> List[Dict[str, Any]]:
#     raw = await redis.lrange(rkey(sid), 0, -1)
#     return [json.loads(j) for j in raw]

# async def trim_old(redis: aioredis.Redis, sid: str, keep_last: int) -> None:
#     await redis.ltrim(rkey(sid), keep_last, -1)

async def rpush(redis, sid, msg):
    # Push message to Redis list
    await redis.rpush(rkey(sid), json.dumps(msg.model_dump()))
    # Set/update TTL for the session key
    await redis.expire(rkey(sid), CACHE_TTL_SECONDS)

async def rlen(redis, sid):
    return await redis.llen(rkey(sid))

async def rget_all(redis, sid):
    raw = await redis.lrange(rkey(sid), 0, -1)
    return [json.loads(j) for j in raw]

async def rtrim(redis, sid, keep):
    await redis.ltrim(rkey(sid), keep, -1)

# Store the confirmed summary in Redis so any worker can fetch it
async def set_summary(redis, sid, txt):
    await redis.setex(f"summary:{sid}", CACHE_TTL_SECONDS, txt)

async def get_summary(redis, sid):
    return await redis.get(f"summary:{sid}")

# ──────────────────────── LangGraph ───────────────────────────

async def assistant(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    prompt: list[BaseMessage] = [SYS, *state["messages"]]
    reply = await llm_with_tools.ainvoke(prompt)
    return {"messages": [reply]}


async def build_graph() -> tuple[StateGraph, AsyncRedisSaver]:
    store_ctx = AsyncRedisSaver.from_conn_string(
        REDIS_URL, ttl={"default_ttl": CACHE_TTL_SECONDS})
    checkpointer = await store_ctx.__aenter__()

    sg = StateGraph(MessagesState)
    sg.add_node("assistant", assistant)
    sg.add_node("tools", ToolNode(TOOLS))
    sg.add_edge(START, "assistant")
    sg.add_conditional_edges("assistant", tools_condition)
    sg.add_edge("tools", "assistant")
    return sg.compile(checkpointer=checkpointer), store_ctx

# ───────────────────────── Chat loop ──────────────────────────

# async def chat(sid: str) -> None:
#     redis = aioredis.from_url(REDIS_URL, encoding="utf-8",
#                               decode_responses=True)

#     graph, ctx = await build_graph()
#     tid = sid  # thread_id for LangGraph

#     print(f"Session {sid} — type 'exit' to quit\n")

#     try:
#         # store system prompt once per session
#         await push(redis, sid, SYS)

#         while True:
#             text = input("🧑‍💻 ").strip()
#             if text.lower() in {"exit", "quit", "bye"}:
#                 break

#             human = HumanMessage(content=text)
#             await push(redis, sid, human)

#             state_in = {"messages": [human]}
#             cfg = {"configurable": {"thread_id": tid}}

#             async for chunk in graph.astream(state_in, cfg, stream_mode="values"):
#                 final = chunk  # keep last chunk

#             ai_msg: AIMessage = final["messages"][-1]
#             print("🤖", ai_msg.content)
#             await push(redis, sid, ai_msg)

#             # summarise & roll window
#             if await history_len(redis, sid) >= ROLL_AFTER * 2:
#                 hist = await history_dump(redis, sid)
#                 blob = "\n".join(m["content"] for m in hist)
#                 summary = await llm.ainvoke(blob)
#                 summ_msg = AIMessage(content="[SUMMARY]\n" + summary.content)
#                 await push(redis, sid, summ_msg)
#                 # keep newest entries (2×ROLL_AFTER + summary)
#                 await trim_old(redis, sid, ROLL_AFTER * 2)
#                 await push(redis, sid, SYS)

#     finally:
#         await ctx.__aexit__(None, None, None)
#         await redis.close()

# ────────────────────────── API ────────────────────────────────

# ----- request/response models -----
class StartSession(BaseModel):
    unit_id: int

class ChatRequest(BaseModel):
    session_id: str
    input: str

# ----- helper: format unit blob -----
def format_unit_info(unit: dict) -> str:
    """
    Format a unit blob for the agent. Extracts:
      - unit name, unit type
      - asset name, asset type, asset coordinates, asset country
      - org name, org description, org hq location, org country
      - teams assigned: team name and each user's role name
    """
    unit_info = unit.get("unitInfo", {}) or {}
    asset = unit_info.get("asset") or unit.get("asset", {}) or {}
    org = asset.get("organisation") or asset.get("organization") or {}
    teams = unit.get("teamsAssigned", []) or unit.get("teams", []) or []

    # Unit fields
    unit_name = unit_info.get("name") or unit_info.get("unitName") or "Unknown"
    unit_type = unit_info.get("type") or unit_info.get("unitType") or "Unknown type"

    # Asset fields
    asset_name = asset.get("name") or asset.get("assetName") or "Unknown"
    asset_type = asset.get("type") or asset.get("assetType") or "Unknown type"
    asset_coords = asset.get("coordinates") or asset.get("coords") or "?"
    asset_country = asset.get("country") or asset.get("countryName") or "Unknown"

    # Organisation fields
    org_name = org.get("name") or "Unknown"
    org_desc = org.get("description") or org.get("desc") or "No description"
    org_hq = org.get("hqLocation") or org.get("hq") or org.get("headquarters") or "Unknown"
    org_country = org.get("country") or asset_country or "Unknown"

    # Build string parts (unit-first)
    lines = []
    # Unit Details (primary)
    lines.append("Unit Details")
    lines.append(f"  • Unit Name: {unit_name}")
    lines.append(f"  • Unit Type: {unit_type}")

    # Asset context
    lines.append("")
    lines.append("Asset Details")
    lines.append(f"  • Asset Name: {asset_name}")
    lines.append(f"  • Asset Type: {asset_type}")
    lines.append(f"  • Asset Coordinates: {asset_coords}")
    lines.append(f"  • Asset Country: {asset_country}")

    # Organisation / Owner
    lines.append("")
    lines.append("Organisation / Owner")
    lines.append(f"  • Name: {org_name}")
    lines.append(f"  • Description: {org_desc}")
    lines.append(f"  • HQ Location: {org_hq}")
    lines.append(f"  • Country: {org_country}")

    # Teams
    lines.append("")
    if teams:
        lines.append("Teams Assigned")
        for t in teams:
            t_name = t.get("name") or t.get("teamName") or "Unnamed Team"
            users_raw = t.get("users", []) or t.get("members", []) or []
            if users_raw:
                user_lines = []
                for u in users_raw:
                    # build user full name
                    fname = u.get("firstName") or u.get("firstname") or ""
                    lname = u.get("lastName") or u.get("lastname") or ""
                    if not (fname or lname):
                        # fallback single-field name
                        fname = u.get("name") or u.get("fullName") or u.get("displayName") or "Unknown"
                        lname = ""
                    full_name = f"{fname} {lname}".strip()
                    role = u.get("roleName") or u.get("role") or u.get("position") or "Unknown role"
                    user_lines.append(f"{full_name} ({role})")
                users_str = ", ".join(user_lines)
            else:
                users_str = "No users"
            lines.append(f"  • Team: '{t_name}' - {users_str}")
    else:
        lines.append("Teams Assigned: No teams have been assigned.")

    # Closing prompt
    lines.append("")
    lines.append("Let's begin scenario setup.")

    return "\n".join(lines)

# ----- regex helpers for confirmation -----
CONFIRM_RE = re.compile(r"is all of this correct.*type ['\"]?ok['\"]?", re.I)
SUMMARY_RE = re.compile(r"(1\\.\\s+\\*?\\*?Core Asset Details.*?)(?:\\n+)?Is all of this correct", re.S | re.I)

def extract_summary(text: str) -> str | None:
    m = SUMMARY_RE.search(text)
    return m.group(1).strip() if m else None

# ────────────────────────── FastApi ENDPOINTS ──────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup and once at graceful shutdown.
    """
    # ---------- STARTUP ----------
    app.state.redis = aioredis.from_url(
        REDIS_URL, encoding="utf-8", decode_responses=True
    )
    app.state.graph, app.state.saver_ctx = await build_graph()

    logging.info("🚀 ALERTSim service started")
    yield                                       # ← application runs here
    # ---------- SHUTDOWN ----------
    await app.state.saver_ctx.__aexit__(None, None, None)
    await app.state.redis.close()
    logging.info("🛑 ALERTSim service stopped")

app = FastAPI(lifespan=lifespan)



@app.post("/start-session")
async def start_session(body: StartSession):
    sid = str(uuid.uuid4())
    await set_summary(app.state.redis, sid, "")            # empty placeholder

    # fetch unit info
    try:
        resp = requests.get(f"{UNIT_API_BASE}/{body.unit_id}")
        resp.raise_for_status()
        unit_json = resp.json()
    except Exception as e:
        return {"session_id": sid,
                "initial_agent_message": "[Error] Failed to fetch unit info.",
                "error": str(e)}

    # feed first turn
    first_user_msg = format_unit_info(unit_json)
    # print(f"Session {sid} started with unit {body.unit_id}: \n{first_user_msg}")
    human = HumanMessage(content=first_user_msg)
    await rpush(app.state.redis, sid, SYS)
    await rpush(app.state.redis, sid, human)

    state_in = {"messages": [human]}
    cfg = {"configurable": {"thread_id": sid}}
    async for chunk in app.state.graph.astream(state_in, cfg, stream_mode="values"):
        final = chunk

    ai_msg: AIMessage = final["messages"][-1]
    await rpush(app.state.redis, sid, ai_msg)

    return {"session_id": sid, "response": ai_msg.content}

@app.post("/chat")
async def chat(body: ChatRequest):
    human = HumanMessage(content=body.input)
    await rpush(app.state.redis, body.session_id, human)

    state_in = {"messages": [human]}
    cfg = {"configurable": {"thread_id": body.session_id}}
    async for chunk in app.state.graph.astream(state_in, cfg, stream_mode="values"):
        final = chunk
    ai_msg: AIMessage = final["messages"][-1]
    await rpush(app.state.redis, body.session_id, ai_msg)

    # store summary if present
    if CONFIRM_RE.search(ai_msg.content):
        summary = extract_summary(ai_msg.content)
        if summary: await set_summary(app.state.redis, body.session_id, summary)

    # intercept \"ok\" to auto-run MSEL tool
    if body.input.lower() == "ok":
        summary = await get_summary(app.state.redis, body.session_id)
        if summary:
            # simulate user confirmation by calling the tool explicitly
            msel_text = await msel_generator(summary)   # direct await
            ai_msg = AIMessage(
                content="Excellent. Based on the confirmed details, here is the Master Scenario Events List:\\n\\n" + msel_text
            )
            await rpush(app.state.redis, body.session_id, ai_msg)
            return {"session_id": body.session_id, "response": ai_msg.content}

    return {"session_id": body.session_id, "response": ai_msg.content}


# ────────────────────────── Run ────────────────────────────────
# if __name__ == "__main__":
#     asyncio.run(chat(str(uuid.uuid4())))