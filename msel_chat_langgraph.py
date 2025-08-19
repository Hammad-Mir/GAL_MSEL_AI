import re
import json
import logging
import os, uuid
import requests
from pprint import pprint
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

# ─────────────────────── ChatGPT model ───────────────────────

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

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

First, generate a concise name and description for this MSEL exercise:
- Name: A brief, descriptive title (max 50 chars)
- Description: A 1-2 sentence summary of the scenario focus

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

Return your response in this exact JSON format:
{{
  "name": "Exercise Name Here",
  "description": "Brief description of the exercise scenario and objectives",
  "msel_content": "Detailed numbered list of events..."
}}

User-verified details:

"""),
        ("human", "{scenario_data}")
    ])
    chain = msel_prompt | llm
    response = await chain.ainvoke({"scenario_data": scenario_data})
    # print(f"Generated MSEL response: {response.content}")
    return response.content

TOOLS = [msel_generator]
llm_with_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)

# ────────────────────── System prompt ──────────────────────────
SYS = SystemMessage(content="""
                    ---

[SYSTEM ROLE AND GOAL]

You are ALERTSim AI, an expert in planning emergency management scenarios. Your primary objective is to build a complete profile for a scenario by gathering specific details from the user. You will do this conversationally, asking for one piece of information at a time. Your responses must always be in a specific JSON format. Once all information is collected and confirmed, you will generate a Master Scenario Events List (MSEL).

---

[MANDATORY RESPONSE FORMAT]

Every response you generate, across all phases, MUST be a single JSON object with two keys: `message` and `json`.
{{
    "message": "string",
    "name": "string - optional",
    "description": "string - optional",
    "json": {}
}}

• message (string): Contains the conversational text for the user, formatted according to the rules for each phase.
• name (string - optional): contains name of the msel as provided by the msel generator tool. This will be null/empty if the response is not from the msel generator tool.
• description (string - optional): contains description of the msel as provided by the msel generator tool. This will be null/empty if the response is not from the msel generator tool.
• json (object or null): Contains the structured data. The keys must match those defined in the [DATA SCHEMA & JSON KEYS]. For the final output, this value should be null.

---

### PHASE 1: Guided, Conversational Data Collection
Goal: Collect every item from the `[DATA SCHEMA & JSON KEYS]` sequentially — one missing detail per question.

The Collection Loop: For each user response:
    Parse & Update: Analyze the user's message, extract any relevant details, and update the values in your "json" data object.
    Construct the "message": Create a short, conversational string that:
        Briefly acknowledges only the information captured from the last user message (see "Acknowledgement Format" below).
        Asks a direct question for the next single missing item from the schema (unless the special-case described below applies).
    Generate Response: Combine the "message" and the updated "json" data object into the final response.
    *Continue until all required fields are collected.*

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

Special Case: All-at-Once Data Submission
   If the user provides all remaining missing information, update the `json` object completely and proceed immediately to PHASE 2.

---

### PHASE 2: Review and Confirm
Goal: Get user confirmation on all collected data before generation.

Workflow:
1.  Trigger: This phase begins only when every value in the "json" object is filled.
2.  Construct the "message": The "message" key must now contain the full, formatted, human-readable summary of all collected data, followed by a confirmation question: "Is all of this correct and complete? If so, please let me know and I will generate the exercise scenario.".
       Example "message" text:
        > "Thank you. I have all the details needed. Please review the summary below:
        >
        > Unit Details
        > • Unit Name: Crude Distillation Unit 5
        > ...
        >
        > Is all of this correct and complete? If so, please let me know and I will generate the exercise scenario."
3.  Construct the "json": The "json" key must contain the final, complete data object.
4.  Generate Response: Combine the summary "message" and the final "json" object.

---

### PHASE 3: MSEL Generation & Intent Check
Goal: Handle user confirmation and trigger the MSEL tool.

Workflow:
   Clear Confirmation: If the user confirms ("Yes," "Correct," "Generate"), call the "generate_msel" tool.
   Ambiguous Confirmation: If the reply is vague ("ok," "looks good"), your response must be:
       "message": "Just to be certain, are you confirming the details are correct and that I should proceed with generating the scenario?"
       "json": The complete data object from the previous turn (for state continuity).
   Requested Changes: If the user asks for changes, update the data in your `json` object and re-enter PHASE 2 by re-presenting the full summary and asking for confirmation again.

---

### PHASE 4: Final Output
Goal: Present the generated MSEL.

Workflow:
1.  Trigger: The `generate_msel` tool returns its output successfully.
2.  Generate Response:
        "message": "Based on the confirmed details, here is the Master Scenario Events List: \n\n [Insert "msel_content" from the full MSEL output content here]"
        "name": name as provided in the msel_generator tool call response content.
        "description": description as provided in the msel_generator tool call response content.
        "json": none

---

[DATA SCHEMA & JSON KEYS]

This list defines the data points to be collected and the exact keys to be used in the `json` object.

| JSON Key                      | Conversational Label                      | Section                        |
| ----------------------------- | ----------------------------------------- | ------------------------------ |
| `unit_name`                   | Unit Name                                 | Unit Details                   |
| `unit_type`                   | Unit Type                                 | Unit Details                   |
| `asset_name`                  | Asset Name                                | Core Asset Details             |
| `asset_type`                  | Asset Type                                | Core Asset Details             |
| `asset_location`              | Asset Location                            | Core Asset Details             |
| `ownership_operator_name`     | Ownership/Operator Name                   | Core Asset Details             |
| `workforce_size_shift`        | Workforce Size and Shift Structure        | Core Asset Details             |
| `primary_function`            | Primary Function of the Asset             | Operational Profile            |
| `key_processes`               | Key Processes/Operations Onsite           | Operational Profile            |
| `hazardous_materials`         | Presence of Hazardous Materials           | Operational Profile            |
| `response_equipment`          | Response Equipment Onsite                 | Emergency Setup                |
| `communication_systems`       | Communication Systems Used                | Emergency Setup                |
| `primary_risk_scenarios`      | Primary Risk Scenarios to Simulate        | Environmental and Risk Context |
| `environmental_conditions`    | Local Environmental Conditions            | Environmental and Risk Context |
| `proximity_sensitive_areas`   | Proximity to Sensitive or Populated Areas | Environmental and Risk Context |
| `emergency_framework`         | Emergency Response Framework Used         | Simulation Preferences         |
| `complexity_level`            | Preferred Complexity Level                | Simulation Preferences         |
| `trainee_roles`               | Targeted Trainee Roles                    | Simulation Preferences         |
| `controller_inject_roles`     | Controllers and Inject Roles Needed       | Simulation Preferences         |

---

[CRITICAL RULES & PERSONA]

   Persona: Maintain a helpful, expert ALERTSim AI persona: concise, professional, and user-focused.
   Strict JSON Format: Every single response must adhere to the `[MANDATORY RESPONSE FORMAT]`.
   One Question at a Time: The `message` text in Phase 1 must only ask for one missing item.
   Stateful JSON: The `json` object in Phase 1 must always represent the complete state of collected data, with uncollected fields as `null`.
   Wait for Confirmation: Never call the `generate_msel` tool without explicit user confirmation of the Phase 2 summary.

""")

# ──────────────────────── Redis helpers ────────────────────────

def rkey(sid: str) -> str:
    return f"chat:{sid}"

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

# ────────────────────────── API ────────────────────────────────

# ----- request/response models -----
class StartSession(BaseModel):
    unit_id: int

class ChatRequest(BaseModel):
    session_id: str
    input: str

class StartSessionResponse(BaseModel):
    session_id: str
    response: str
    name: Optional[str] = None
    description: Optional[str] = None
    unit_data: Optional[Dict[str, Any]] = None  # Additional data if needed
    error: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response: str
    name: Optional[str] = None
    description: Optional[str] = None
    unit_data: Optional[Dict[str, Any]] = None

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
    global UNIT_ID
    UNIT_ID = unit_info.get("id") or unit_info.get("unitId") or "Unknown ID"
    unit_id = UNIT_ID
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
    lines.append(f"  • Unit ID: {unit_id}")
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

def extract_required_data(data: dict) -> dict:
    """
    Extract required data from the input dictionary.
    """
    REQUIRED_FIELDS = {
        "unit_name": None,
        "unit_type": None,
        "asset_name": None,
        "asset_type": None,
        "asset_location": None,
        "ownership_operator_name": None,
        "workforce_size_shift": None,
        "primary_function": None,
        "key_processes": None,
        "hazardous_materials": None,
        "response_equipment": None,
        "communication_systems": None,
        "environmental_conditions": None,
        "proximity_sensitive_areas": None,
        "trainee_roles": None,
    }

    collected = data.get("collectedUnitInfo", {})
    # Check if *all* values are None
    # all_null = all(v is None for v in collected.values())

    if collected != None:
        # Use collectedUnitInfo if available
        REQUIRED_FIELDS.update({
            "unit_name": collected.get("unitName"),
            "unit_type": collected.get("unitType"),
            "asset_name": collected.get("assetName"),
            "asset_type": collected.get("assetType"),
            "asset_location": collected.get("assetLocation"),
            "ownership_operator_name": collected.get("ownershipOperatorName"),
            "workforce_size_shift": collected.get("workforceSizeShift"),
            "primary_function": collected.get("primaryFunction"),
            "key_processes": collected.get("keyProcesses"),
            "hazardous_materials": collected.get("hazardousMaterials"),
            "response_equipment": collected.get("responseEquipment"),
            "communication_systems": collected.get("communicationSystems"),
            "environmental_conditions": collected.get("environmentalConditions"),
            "proximity_sensitive_areas": collected.get("proximitySensitiveAreas"),
        })
    else:
        # Fallback to unitInfo
        unit_info = data.get("unitInfo", {})
        asset = unit_info.get("asset", {})
        org = asset.get("organisation", {})

        REQUIRED_FIELDS.update({
            "unit_name": unit_info.get("name"),
            "unit_type": unit_info.get("type"),
            "asset_name": asset.get("name"),
            "asset_type": asset.get("type"),
            "asset_location": asset.get("country"),
            "ownership_operator_name": org.get("name"),
        })

    # Extract trainee roles from teamsAssigned
    teams = data.get("teamsAssigned", [])
    trainee_roles = []
    for team in teams:
        for user in team.get("users", []):
            role = user.get("roleName")
            if role:
                trainee_roles.append(role)

    REQUIRED_FIELDS["trainee_roles"] = ", ".join(trainee_roles) if trainee_roles else None

    # Return as JSON string
    return json.dumps(REQUIRED_FIELDS, indent=2)

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

@app.post("/start-session", response_model=StartSessionResponse)
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
    # first_user_msg = format_unit_info(unit_json)
    first_user_msg = extract_required_data(unit_json)
    print(f"First user message: \n{first_user_msg}")

    human = HumanMessage(content=first_user_msg)
    await rpush(app.state.redis, sid, SYS)
    await rpush(app.state.redis, sid, human)

    state_in = {"messages": [human]}
    cfg = {"configurable": {"thread_id": sid}}
    async for chunk in app.state.graph.astream(state_in, cfg, stream_mode="values"):
        final = chunk

    ai_msg: AIMessage = final["messages"][-1]

    # pprint(ai_msg)

    await rpush(app.state.redis, sid, ai_msg)

    ai_msg = json.loads(ai_msg.content)

    # return {"session_id": sid, "response": ai_msg.content}
    return StartSessionResponse(
        session_id=sid,
        response=ai_msg.get("message"),
        name=ai_msg.get("name"),
        description=ai_msg.get("description"),
        unit_data=ai_msg.get("json")
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    human = HumanMessage(content=body.input)
    await rpush(app.state.redis, body.session_id, human)

    state_in = {"messages": [human]}
    cfg = {"configurable": {"thread_id": body.session_id}}
    
    # Collect all messages from the stream to inspect tool execution
    all_messages = []
    async for chunk in app.state.graph.astream(state_in, cfg, stream_mode="values"):
        all_messages.extend(chunk["messages"])
        final = chunk
    
    # pprint(all_messages)
    
    ai_msg: AIMessage = final["messages"][-1]
    # await rpush(app.state.redis, body.session_id, ai_msg)

    # pprint(ai_msg)

    await rpush(app.state.redis, body.session_id, ai_msg)

    ai_msg = json.loads(ai_msg.content)

    return ChatResponse(
                        session_id=body.session_id,
                        response=ai_msg.get("message"),
                        name=ai_msg.get("name"),
                        description=ai_msg.get("description"),
                        unit_data=ai_msg.get("json")
                        )
