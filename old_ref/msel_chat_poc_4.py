import asyncio
import os, re, uuid
from typing import Optional
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.redis import RedisSaver
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory

from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

## ---------------------------- Environment Variables ---------------------------- ##
# Load environment variables from .env file
load_dotenv()
UNIT_API_BASE = os.getenv("UNIT_API_BASE")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

print(REDIS_URL)
print(UNIT_API_BASE)
CACHE_TTL_SECONDS = 60 * 60 * 24   # 24h TTL by default

## ---------------------------- Model ---------------------------- ##

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

## ---------------------------- Prompts ---------------------------- ##

collection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are ALERTSim AI, an expert exercise planner for emergency management scenarios.

**IMPORTANT BEHAVIOR INSTRUCTIONS:**
1. If you receive tool results from generate_msel, you must present the MSEL content to the user in your own voice with an appropriate introduction like: "Excellent. Based on the confirmed details, here is the Master Scenario Events List:" followed by the complete MSEL content.

2. During data collection phase: *Conversationally* ask the user to provide all the following information from the user IN NATURAL LANGUAGE (not as a form, not as Field: Value pairs), one at a time, in a friendly and helpful manner:
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

3. When user confirms with 'ok' or similar explicit confirmation, call the generate_msel tool with the complete summary data.

4. You are the ONLY entity that communicates with the user. Always maintain your ALERTSim AI persona in all responses.

Wait for explicit user confirmation before moving on. If the user makes any changes, update the summary accordingly, request confirmation only for the changed details and forward the updated summary to the MSEL generator tool.
"""),
    MessagesPlaceholder("messages"),
    ("human", "{input}")
])

## ---------------------------- Tools ---------------------------- ##

@tool
def generate_msel(scenario_data: str):
    """Generate a Master Scenario Events List based on collected emergency response data.
    
    Takes user-confirmed asset and scenario information and generates a detailed,
    stepwise Master Scenario Events List for emergency response training.
    
    Args:
        scenario_data (str): The user-confirmed summary of asset and scenario information
    
    Returns:
        str: A detailed MSEL as a plain-text numbered list of events
    """
    
    msel_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are ALERTSim AI, an expert scenario designer.
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
    msel_chain = msel_prompt | llm
    response = msel_chain.ainvoke({"scenario_data": scenario_data})

    return response.content

@tool
def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y

## ---------------------------- State and Graph ---------------------------- ##

class AgentState(MessagesState):
    """State for the conversational agent."""
    session_id: str
    user_id: str
    summary: Optional[str]
    pass

# Create tools and tool node
tools = [generate_msel, multiply]

def should_continue(state: AgentState):
    """Determine whether to continue to tools or end."""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"
    
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "__end__"

## ---------------------------- Checkpointer ---------------------------- ##
checkpointer = RedisSaver(
    redis_url=REDIS_URL,
    ttl=CACHE_TTL_SECONDS,
)

## ---------------------------- Agent Class ---------------------------- ##


## ---------------------------- Interactive Interface ---------------------------- ##

