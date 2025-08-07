import re
import uuid
from dotenv import load_dotenv
from colorama import Fore, Back, Style
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

load_dotenv()

# --- 1. PROMPTS
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

# --- 2. MEMORY (RunnableWithMessageHistory compatible)

class SimpleInMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []
    def add_message(self, message):
        self.messages.append(message)
    def add_messages(self, messages):
        self.messages.extend(messages)
    def clear(self):
        self.messages = []

_history_store = {}
def get_session_history(session_id: str):
    if session_id not in _history_store:
        _history_store[session_id] = SimpleInMemoryHistory()
    return _history_store[session_id]

# --- 3. LLM Chains
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
collection_chain = collection_prompt | llm | StrOutputParser()
msel_chain = msel_prompt | llm | StrOutputParser()

# --- 4. Conversation Loop Using RunnableWithMessageHistory
history_chain = RunnableWithMessageHistory(
    collection_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# --- 5. CONFIRMATION REGEX
def is_confirmation_prompt(response: str):
    return bool(re.search(r"is all of this correct.*type ['\"]?ok['\"]?", response, re.IGNORECASE))

def extract_summary(response: str):
    # Match from any form of "1. Core Asset Details" until just before the confirmation prompt
    pattern = r"(1\.\s+\*?\*?Core Asset Details.*?)(?:\n+)?Is all of this correct"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

# --- 6. MAIN LOOP
def main():
    print("Welcome to ALERTSim AI - Emergency Scenario Setup!")
    session_id = str(uuid.uuid4())[:8]
    print(f"(Session ID: {session_id})")
    print("Talk to me about your facility and simulation needs (in your own words). Type 'exit' to quit.\n")

    collected_summary = None

    while True:
        user = input("You: ")
        if user.lower() in {"exit", "quit"}:
            print("ALERTSim AI: Goodbye!")
            break

        response = history_chain.invoke(
            {"input": user},
            config={"configurable": {"session_id": session_id}}
        )
        print("ALERTSim AI:", response)

        if is_confirmation_prompt(response):
            summary = extract_summary(response)
            if summary:
                collected_summary = summary
            else:
                print(Fore.YELLOW + "⚠️ Couldn't extract summary, please re-confirm or try again." + Style.RESET_ALL)
                continue

            while True:
                user_confirm = input("You (confirm): ")
                if user_confirm.strip().lower() in {"ok", "yes", "confirm"}:
                    if not collected_summary:
                        print(Fore.RED + "❌ Error: No valid summary data available." + Style.RESET_ALL)
                        break
                    print("\nGenerating your Master Scenario Event List (MSEL), please wait...\n")
                    print(Fore.CYAN + "✅ Using summary:\n" + collected_summary + Style.RESET_ALL)
                    msel = msel_chain.invoke({"collected_data": collected_summary})
                    print("\nALERTSim AI:\n" + msel + "\n")
                    return  # Exit after successful MSEL generation
                else:
                    response = history_chain.invoke(
                        {"input": user_confirm},
                        config={"configurable": {"session_id": session_id}}
                    )
                    print("ALERTSim AI:", response)
                    if is_confirmation_prompt(response):
                        summary = extract_summary(response)
                        if summary:
                            collected_summary = summary

# --- RUN APP
if __name__ == "__main__":
    main()