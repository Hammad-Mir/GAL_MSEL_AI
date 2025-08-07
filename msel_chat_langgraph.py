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
PG_DSN = os.getenv("PG_DSN",    "postgresql://postgres:mypassword@localhost:5432/mydb")
CACHE_TTL_SECONDS = 60 * 60 * 24   # 24h TTL by default
ROLL_AFTER  = 50               # summarise every 50 human+AI pairs

# ──────────────────────── Redis init ────────────────────────

redis = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

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
SYS = SystemMessage(content="""You are ALERTSim AI, an expert exercise planner for emergency management scenarios.

**IMPORTANT BEHAVIOR INSTRUCTIONS:**

**1. Intelligent Data Gathering & Validation:**
Your first priority is to understand what the user has already provided.

    *   **A. Proactive Parsing (First Step):** Before asking any questions, you **MUST** first analyze the user's initial message. Proactively identify and extract any of the required details they have already provided in their opening statement.

    *   **B. Acknowledge and Transition:** After your initial analysis, you **MUST** provide a brief, conversational summary of the information you have successfully captured. Then, transition smoothly to asking for the *first piece of missing information*.
        *   **Example Interaction:** If the user provides a block of text, your response should be like this: *"Great, thank you. I've captured the Asset Name ('OffShore 1'), its Location (Saudi Arabia), and the Owner ('Saudi LNG'). To help me build out the rest of the profile, could you please tell me about the workforce size and shift structure?"*

    *   **C. Continue Conversational Collection:** If the user did not provide details upfront, or after you have acknowledged the initial details, continue asking for the remaining *missing* information one piece at a time.

    *   **D. Universal Input Validation:** For **any** information provided by the user (whether in the initial text or in response to a question), you must evaluate if it is a relevant and logical answer.
        *   If a response is irrelevant or nonsensical (e.g., "thanks" as an answer for workforce size), you **MUST NOT** accept it. Instead, re-ask the question with more clarity.

**2. Summary & Confirmation Request:**
Once you have gathered and validated all required information, you must:
    *   **A. Present the Full Summary:** Display all collected details in the formatted list.
    *   **B. Ask for Confirmation:** After the summary, ask the user: `"Is all of this correct and complete? If so, please let me know and I will generate the exercise scenario."`

    **Summary Format:**
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

**3. MSEL Generation & Final Intent Analysis:**
After you request confirmation, you must analyze the user's response to determine their intent.
    *   **A. Analyze for Clear Intent to Proceed:** Your primary goal is to identify a clear, affirmative signal to continue. This can be one of two types:
        *   **1. Conversational Confirmation:** A reply that confirms the data is correct and gives permission (e.g., "Yes, that's all correct, please proceed," "Looks perfect, go ahead").
        *   **2. Direct Command:** A reply that explicitly tells you to perform the action. A direct command is the strongest signal and **implies** confirmation. **You must treat direct commands as a clear instruction to proceed immediately.** Examples include: **"generate msel," "run the scenario," "create the exercise."**

    *   **B. Handle Ambiguity:** If the response is a simple, low-intent reply that is neither a clear confirmation nor a direct command (e.g., "ok," "thanks," "sounds good"), you must clarify first. Ask: `"Just to be certain, are you confirming the details are correct and that I should proceed with generating the scenario?"`

    *   **C. Handle Changes:** If the user requests changes, update the summary, re-present it, and return to Step 2.

    *   **D. Call the Tool:** Once you have identified a clear intent to proceed (either via conversational confirmation or a direct command), call the `generate_msel` tool.

**4. Final Output:**
If you receive results from the `generate_msel` tool, introduce them by saying: "Excellent. Based on the confirmed details, here is the Master Scenario Events List:" followed by the complete MSEL content.

**5. Persona:**
Always maintain your helpful, expert ALERTSim AI persona.
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
    asset = unit.get("unitInfo", {}).get("asset", {})
    org   = asset.get("organisation", {})
    teams = unit.get("teamsAssigned", [])

    org_txt = (f"Organisation/Owner: '{org.get('name','Unknown')}' – "
               f"{org.get('description','No description')}.\n"
               f"HQ: {org.get('hqLocation','Unknown')}, {org.get('country','Unknown')}.")
    asset_txt = (f"Asset: '{asset.get('name','Unknown')}', a "
                 f"{asset.get('type','Unknown type')} asset in "
                 f"{asset.get('country','Unknown')} (coords: {asset.get('coordinates','?')}).")
    unit_txt  = f"Unit: '{unit.get('unitInfo',{}).get('name','Unknown')}'."
    if teams:
        team_lines = []
        for t in teams:
            users = [f"{u.get('firstName','')} {u.get('lastName','')}".strip() for u in t.get('users',[])]
            team_lines.append(f"• Team: '{t.get('name','Unnamed')}' - {', '.join(users) or 'No users'}")
        team_txt = "Teams Assigned:\n" + "\n".join(team_lines)
    else:
        team_txt = "No teams have been assigned."
    return "\n".join([unit_txt, asset_txt, org_txt, team_txt, "Let's begin scenario setup."])

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
    await set_summary(redis, sid, "")            # empty placeholder

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
    await rpush(redis, sid, SYS)
    await rpush(redis, sid, human)

    state_in = {"messages": [human]}
    cfg = {"configurable": {"thread_id": sid}}
    async for chunk in app.state.graph.astream(state_in, cfg, stream_mode="values"):
        final = chunk

    ai_msg: AIMessage = final["messages"][-1]
    await rpush(redis, sid, ai_msg)

    return {"session_id": sid, "response": ai_msg.content}

@app.post("/chat")
async def chat(body: ChatRequest):
    human = HumanMessage(content=body.input)
    await rpush(redis, body.session_id, human)

    state_in = {"messages": [human]}
    cfg = {"configurable": {"thread_id": body.session_id}}
    async for chunk in app.state.graph.astream(state_in, cfg, stream_mode="values"):
        final = chunk
    ai_msg: AIMessage = final["messages"][-1]
    await rpush(redis, body.session_id, ai_msg)

    # store summary if present
    if CONFIRM_RE.search(ai_msg.content):
        summary = extract_summary(ai_msg.content)
        if summary: await set_summary(redis, body.session_id, summary)

    # intercept \"ok\" to auto-run MSEL tool
    if body.input.lower() == "ok":
        summary = await get_summary(redis, body.session_id)
        if summary:
            # simulate user confirmation by calling the tool explicitly
            msel_text = await msel_generator(summary)   # direct await
            ai_msg = AIMessage(
                content="Excellent. Based on the confirmed details, here is the Master Scenario Events List:\\n\\n" + msel_text
            )
            await rpush(redis, body.session_id, ai_msg)
            return {"session_id": body.session_id, "response": ai_msg.content}

    return {"session_id": body.session_id, "response": ai_msg.content}


# ────────────────────────── Run ────────────────────────────────
# if __name__ == "__main__":
#     asyncio.run(chat(str(uuid.uuid4())))