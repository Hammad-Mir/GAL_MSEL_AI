"""
ALERTSim AI – master script
Author: <you>
Updated: 2025-08-06
"""

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg
import redis.asyncio as aioredis
from dataclasses_json import config
from dotenv import load_dotenv
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.redis import RedisSaver

# ─────────────────────────── Logging ───────────────────────────
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ───────────────────────── Environment ─────────────────────────
load_dotenv()  # reads .env if present
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL_SECONDS: int = 86400  # 24 h
ROLL_AFTER: int = 50             # summarise every 50 human+AI pairs

# ─────────────────────────── Models ────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ──────────────────────── Tool definition ─────────────────────────
@tool
def msel_generator(scenario_data: str) -> str:
    """
    Generate a Master Scenario Events List (MSEL) based on confirmed data.
    Returns a plain-text, numbered list.
    """
    from langchain_core.prompts import ChatPromptTemplate
    msel_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are ALERTSim AI, an expert scenario designer.\n"
             "Using ONLY the confirmed asset/scenario info below, create a detailed, "
             "stepwise MSEL for emergency-response training.\n"
             "Write in English. STRICTLY OUTPUT plain text (no markdown, no tables).\n"
             "Every event must include:\n"
             "- Event Number\n- Scenario Time\n- Event Type\n- Inject Mode\n"
             "- From\n- To\n- Message\n- Expected Participant Response\n"
             "- Objectives/Capabilities Tested\n- Notes\n\n"
             "Confirmed details:\n"),
            ("human", "{scenario_data}")
        ]
    )
    chain     = msel_prompt | llm
    response  = chain.invoke({"scenario_data": scenario_data})
    return response.content

TOOLS           = [msel_generator]
llm_with_tools  = llm.bind_tools(TOOLS, parallel_tool_calls=False)

# ───────────────────── System prompt ──────────────────────────────
SYS = SystemMessage(
    content="""
You are ALERTSim AI, an expert exercise planner for emergency-management scenarios.

IMPORTANT BEHAVIOR RULES
1. If a tool call returns an MSEL, introduce it with:
   "Excellent. Based on the confirmed details, here is the Master Scenario Events List:"
   and then output the MSEL verbatim.
2. During data-collection, conversationally gather all of the following,
   one piece at a time (not as a form):
   • Asset Name, Asset Type, Asset Location, Ownership/Operator, Workforce Size & Shifts
   • Primary Function, Key Operations, Hazardous Materials (y/n & type)
   • Onsite Response Equipment, Communication Systems
   • Primary Risk Scenarios, Local Environmental Conditions, Proximity to Populated Areas
   • Emergency-response Framework, Preferred Complexity, Target Trainee Roles,
     Controllers/Inject Roles
   Ask follow-ups to clarify anything missing.
3. Once all info is complete, print a clear summary in *exactly* this list structure,
   then ask the user to confirm with "ok":
   1. Core Asset Details
      • Asset Name:
      • Asset Type:
      • Asset Location:
      • Ownership/Operator Name:
      • Workforce Size and Shift Structure:
   2. Operational Profile
      • Primary Function of the Asset:
      • Key Processes/Operations Onsite:
      • Presence of Hazardous Materials:
   3. Emergency Setup
      • Response Equipment Onsite:
      • Communication Systems Used:
   4. Environmental and Risk Context
      • Primary Risk Scenarios to Simulate:
      • Local Environmental Conditions:
      • Proximity to Sensitive or Populated Areas:
   5. Simulation Preferences
      • Emergency Response Framework Used:
      • Preferred Complexity Level:
      • Targeted Trainee Roles:
      • Controllers and Inject Roles Needed:
4. When the user types "ok", call the msel_generator tool with the complete summary.
5. Stay in the ALERTSim AI persona at all times.
"""
)

# ────────────────────────── Graph nodes ──────────────────────────
def assistant(state: MessagesState):
    """Synchronous assistant node."""
    reply = llm_with_tools.invoke([SYS] + state["messages"])
    return {"messages": [reply]}

# ────────────────────────── Build graph ──────────────────────────
checkpointer = RedisSaver.from_conn_string(
    REDIS_URL, ttl={"default_ttl": CACHE_TTL_SECONDS}
)

sg = StateGraph(MessagesState)
sg.add_node("assistant", assistant)
sg.add_node("tools", ToolNode(TOOLS))
sg.add_edge(START, "assistant")
sg.add_conditional_edges("assistant", tools_condition)
sg.add_edge("tools", "assistant")

graph = sg.compile(checkpointer=checkpointer)

