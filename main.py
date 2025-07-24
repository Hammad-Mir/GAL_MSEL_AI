import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="ALERTSim AI Chatbot", page_icon="🚨")
st.title("🚨 ALERTSim AI: Emergency Simulation Chatbot")

# --- Sidebar: Unit ID Input and Start Button (now also acts as Start New Session) ---
st.sidebar.header("Start Scenario Setup")
unit_id_input = st.sidebar.text_input("Enter Unit ID", key="unit_id_input")
if "last_unit_id" not in st.session_state:
    st.session_state.last_unit_id = None
start_clicked = st.sidebar.button("Start")

# --- Session State Initialization ---
for key, default in [
    ("session_id", None),
    ("messages", []),
    ("awaiting_confirmation", False),
    ("collected_summary", None),
    ("chat_enabled", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- On Start click with a Unit ID, reset and begin new session ---
if start_clicked and unit_id_input:
    try:
        resp = requests.post(
            f"{API_BASE}/start-session",
            json={"unit_id": int(unit_id_input)}
        )
        data = resp.json()
        st.session_state.session_id = data["session_id"]
        st.session_state.messages = [
            {"role": "user", "content": data["user_message"]},
            {"role": "assistant", "content": data["initial_agent_message"]}
        ]
        st.session_state.awaiting_confirmation = False
        st.session_state.collected_summary = None
        st.session_state.chat_enabled = True
        st.session_state.last_unit_id = unit_id_input
    except Exception as e:
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.awaiting_confirmation = False
        st.session_state.collected_summary = None
        st.session_state.chat_enabled = False
        st.error("Failed to start scenario: " + str(e))

st.caption(f"Session ID: `{st.session_state.session_id}`" if st.session_state.session_id else "")

# --- Display chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def handle_user_input(user_input: str):
    user_input_lower = user_input.strip().lower()
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # After confirmation, get MSEL
    if st.session_state.awaiting_confirmation and user_input_lower in ("ok", "yes", "confirm"):
        msel_resp = requests.get(
            f"{API_BASE}/generate-msel/{st.session_state.session_id}"
        ).json()
        msel = msel_resp.get("msel") or msel_resp.get("error")
        ai_msg = f"📝 **Master Scenario Event List (MSEL):**\n\n{msel}"
        with st.chat_message("assistant"):
            st.markdown(ai_msg)
        st.session_state.messages.append({"role": "assistant", "content": ai_msg})
        st.session_state.awaiting_confirmation = False
        return

    resp = requests.post(
        f"{API_BASE}/chat",
        json={"session_id": st.session_state.session_id, "input": user_input}
    ).json()
    ai_msg = resp["response"]
    st.session_state.awaiting_confirmation = resp.get("awaiting_confirmation", False)
    st.session_state.collected_summary = resp.get("summary")
    with st.chat_message("assistant"):
        st.markdown(ai_msg)
    st.session_state.messages.append({"role": "assistant", "content": ai_msg})

# --- Only enable chat after Start was clicked with valid Unit ID ---
if st.session_state.chat_enabled and st.session_state.session_id:
    if prompt := st.chat_input("Type your message and press Enter..."):
        handle_user_input(prompt)
else:
    st.info("Please enter the Unit ID at left and press 'Start' to begin scenario setup.")

st.sidebar.markdown("🔗 [Open Backend API Docs](http://localhost:8000/docs)")
st.markdown("---")
st.caption("ALERTSim AI © 2025. Powered by Streamlit & FastAPI.")
