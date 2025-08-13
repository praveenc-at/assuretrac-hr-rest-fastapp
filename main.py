import os
import json
from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
# from pyngrok import ngrok
import uvicorn
import re
from typing import Tuple, List
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import os
import io
import ssl
import smtplib
import threading
from email.message import EmailMessage
import pandas as pd
import re

# Load env variables
load_dotenv()

# --- LLM Config ---
llm = ChatOpenAI(model="gpt-4o-mini", api_key="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")

# --- Data Models ---
class UserDetails(BaseModel):
    full_name: str = Field(..., description="User's full name")
    email: str = Field(..., description="User's email address")
    phone_number: str = Field(..., description="User's phone number")

class TechStackDetails(TypedDict):
    tech_stack: str
    rating: str
    explanation: str

class NodeNavigation(BaseModel):
    node_name: str = Field(description="The next node to navigate to. Must be one of: ask_user_if_interested, confirm_user_interest, extract_user_details, confirm_user_details, fetch_user_tech_stack_node, greet_bye")

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_interested: str
    node_name: str
    user_details: Optional[dict] = None
    tech_stack_details: Optional[List[dict]] = None


class TechStackItem(BaseModel):
    technology: str = Field(..., description="Name of the technology")
    experience_years: float = Field(..., description="Years of experience with this technology")
    rating_out_of_10: int = Field(..., description="Self-assessed rating out of 10")
    explanation: str = Field(..., description="Brief explanation of the experience with this technology")

class TechStackResponse(BaseModel):
    tech_stacks: List[TechStackItem] = Field(..., description="List of technologies the user has worked with")


def build_conversation_text(state: State, max_messages: Optional[int] = None) -> Tuple[str, str, List[str]]:
    """
    Build a labeled conversation string from state["messages"].
    - Labels each message as USER or ASSISTANT.
    - Returns (conversation_text, last_message_text, list_of_lines).
    - If max_messages is set, it will keep only the last max_messages messages.
    """
    msgs = list(state.get("messages", []))
    if max_messages is not None and len(msgs) > max_messages:
        msgs = msgs[-max_messages:]

    convo_lines = []
    for m in msgs:
        # Prefer class-based checks so we preserve roles
        if isinstance(m, HumanMessage):
            role = "USER"
        elif isinstance(m, AIMessage):
            role = "ASSISTANT"
        else:
            # fallback: try to read role attribute or assume user
            role = getattr(m, "role", "user").upper()
            if role not in ("USER", "ASSISTANT"):
                role = "USER"

        text = getattr(m, "content", str(m))
        # normalize whitespace
        text = text.strip()
        convo_lines.append(f"{role}: {text}")

    all_messages_text = "\n".join(convo_lines)
    last_msg_text = convo_lines[-1].split(":", 1)[1].strip() if convo_lines else ""
    return all_messages_text, last_msg_text, convo_lines

def create_excel_bytes(personal: dict = None, tech_stack_list: list = None) -> bytes:
    """
    Create an in-memory Excel (.xlsx) file.
    If personal is provided, writes a 'personal' sheet.
    If tech_stack_list is provided, writes a 'tech_stack' sheet.
    Returns bytes.
    """
    personal = personal or {}
    tech_stack_list = tech_stack_list or []

    with io.BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            # Personal sheet only if personal has any keys
            if personal:
                pd.DataFrame([personal]).to_excel(writer, index=False, sheet_name="personal")
            # Tech sheet only if tech_stack_list has items
            if tech_stack_list:
                pd.DataFrame(tech_stack_list).to_excel(writer, index=False, sheet_name="tech_stack")

        buffer.seek(0)
        return buffer.read()


# def _smtp_send_message(msg: EmailMessage):
#     """Blocking SMTP send; run in background thread."""
#     host = os.getenv("SMTP_HOST")
#     port = int(os.getenv("SMTP_PORT", "587"))
#     user = os.getenv("SMTP_USER")
#     password = os.getenv("SMTP_PASSWORD")
#     use_ssl = os.getenv("SMTP_USE_SSL", "false").lower() in ("true", "1", "yes")

#     try:
#         if use_ssl or port == 465:
#             context = ssl.create_default_context()
#             with smtplib.SMTP_SSL(host, port, context=context) as smtp:
#                 if user and password:
#                     smtp.login(user, password)
#                 smtp.send_message(msg)
#         else:
#             with smtplib.SMTP(host, port) as smtp:
#                 smtp.starttls(context=ssl.create_default_context())
#                 if user and password:
#                     smtp.login(user, password)
#                 smtp.send_message(msg)
#         print("Email sent successfully.")
#     except Exception as e:
#         # Log or print; in production, integrate with proper logs
#         print("Failed to send email:", e)


# def send_email_with_attachment_background(subject: str, body: str, attachment_bytes: bytes, attachment_filename: str):
#     """
#     Build EmailMessage and send in background daemon thread.
#     """
#     from_addr = os.getenv("EMAIL_FROM")
#     to_addrs = os.getenv("EMAIL_TO", "")
#     if not to_addrs:
#         print("EMAIL_TO not configured, not sending email.")
#         return

#     recipients = [addr.strip() for addr in to_addrs.split(",") if addr.strip()]

#     msg = EmailMessage()
#     msg["Subject"] = subject
#     msg["From"] = from_addr
#     msg["To"] = ", ".join(recipients)
#     msg.set_content(body)

#     # Attach excel
#     maintype = "application"
#     subtype = "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#     msg.add_attachment(attachment_bytes, maintype=maintype, subtype=subtype, filename=attachment_filename)

#     thread = threading.Thread(target=_smtp_send_message, args=(msg,), daemon=True)
#     thread.start()

def naviagate_to_respective_node(state: State) -> dict:
    """ Determine the next node to navigate to based on the conversation history."""

    print("state[messages] in navigate to respective node -->", state["messages"])
    all_messages_text, last_msg_text, convo_lines = build_conversation_text(state, max_messages=None)
    last = last_msg_text.strip().lower()

    # Helpers
    def is_plain_greeting(s: str) -> bool:
        tokens = re.findall(r"\w+", s)
        greetings = {"hi", "hello", "hey", "hiya", "yo"}
        return 0 < len(tokens) <= 4 and all(t in greetings or t in {"good","morning","evening","afternoon"} for t in tokens)

    def contains_negative_intent(s: str) -> bool:
        return bool(re.search(r"\b(no|not interested|nope|don't want|do not want|not now|maybe later|not for me)\b", s))

    # Priority deterministic routing
    if is_plain_greeting(last):
        print("Detected greeting -> ask_user_if_interested")
        return {"node_name": "ask_user_if_interested"}

    if contains_negative_intent(last):
        print("Detected negative intent -> greet_bye")
        return {"node_name": "greet_bye"}

    message = ""

    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            message = message + "USER: " + msg.content + "\n"
        elif isinstance(msg, AIMessage):
            message = message + "ASSISTANT: " + msg.content + "\n"
        else:
            message = message + msg.content + "\n"

    print("message in naviagate_to_respective_node -->\n", message)
    print("\n\n")
    
    prompt = (
        "You are an orchestration assistant for an HR voice chatbot.\n"
        "Your task is to select the NEXT node to execute based on the FULL conversation. You have the FULL Conversation below. Please understand what is conversation and return the name of the node as per the Rules or Instructions\n"
        "You must follow the rules below in strict priority order (higher rules override lower ones).\n\n"

        "RULES (Highest priority first):\n"
        "1) If the last user message is ONLY a greeting (e.g., 'Hi', 'Hello', 'Hey') -> choose 'ask_user_if_interested'.\n"
        "2) If the user clearly states they are NOT interested (e.g., 'No', 'Not interested', 'No thanks') -> choose 'greet_bye'.\n"
        "3) If the user clearly says they are interested in the job opportunity and we have NOT yet asked for or collected their personal details -> choose 'confirm_user_interest'.\n"
        "4) If the user has shared personal details but NOT confirmed they are correct -> choose 'extract_user_details'.\n"
        "5) If the user has CONFIRMED his personal details are correct -> choose 'confirm_user_details' where are asking user for his technical work details & experience.\n"
        "6) If the user has shared the Technical details / Technical work experiences he worked on OR gave rating to his skillset -> choose 'fetch_user_tech_stack_node'.\n"
        "7) When the user is not interested in the job opening or when we have collected all the details and we need to end the conversation -> choose 'greet_bye' \n\n"        
        "8) If none of the above conditions are met, pick the node that logically continues the conversation.\n\n"

        "IMPORTANT:\n"
        "- Always review the ENTIRE conversation history, not just the last message.\n"
        "- Output ONLY one of the following node names, exactly as written:\n"
        "- Return 'confirm_user_interest' - If the user has shown interest in the job opportunity. choose this node\n"
        "- Return 'confirm_user_details' - If the user has confirmed his personal details are correct. Choose this node.\n"
        "- Understand the difference between 'confirm_user_interest' and 'confirm_user_details' by the explanation give to you.\n"
        "- Return 'fetch_user_tech_stack_node' - when you see technical skill names/ rating for user technical skills out of 5 OR out of 10 \n"
        " Make sure before returning 'greet_bye' node, We have extracted the technical details by returning 'fetch_user_tech_stack_node' \n"
        " Make sure to return 'greet_bye' node only  if below 2 are satisfied\n"
        " 1) We have collected the technical details from the user -- To know this make sure in the conversation we have this particular string - 'We have saved your tecnical details' \n"
        " 2) User has rejected saying, he is not intrested in this oppurtunity  \n"
        " Never ever return 'greet_bye' if user says, he is interested in the job opening. NEVER EVER \n"
        "[ask_user_if_interested, confirm_user_interest, extract_user_details, confirm_user_details, fetch_user_tech_stack_node, greet_bye]\n\n"

        "Conversation history:\n"
        f"{message}\n"
        "Your answer (node name only):"
        " (choose from: ask_user_if_interested, confirm_user_interest, extract_user_details, confirm_user_details, fetch_user_tech_stack_node, greet_bye)\n"
        "Make sure to choose the node that best fits the conversation context and follows the rules above.\n"
        "Do not output anything other than the node name.\n"
        "If you are unsure, default to 'ask_user_if_interested' to re-engage the user.\n"
        "Make sure you understand the conversation and decide node name \n"
        "Do not hallucinate or make up node names, only return node names based on the context provided.\n"
    )


    print("LLM fallback prompt -> sending to LLM")
    res = llm.with_structured_output(NodeNavigation).invoke([HumanMessage(content=prompt)])
    chosen = getattr(res, "node_name", "").strip()
    valid = {"ask_user_if_interested", "confirm_user_interest", "extract_user_details", "confirm_user_details",  "fetch_user_tech_stack_node", "greet_bye"}
    if chosen not in valid:
        print("LLM gave invalid node -> defaulting to ask_user_if_interested")
        return {"node_name": "ask_user_if_interested"}
    print("LLM chose:", chosen)
    return {"node_name": chosen}


def navigate_as_per_node_navigation(state: State) -> str:
    valid_nodes = [
        "ask_user_if_interested",
        "confirm_user_interest",
        "extract_user_details",
        "confirm_user_details",
        "fetch_user_tech_stack_node",
        "greet_bye"
    ]
    print("state[node_name] in navigate_as_per_node_navigation -->", state["node_name"])
    return state["node_name"] if state["node_name"] in valid_nodes else "greet_bye"

def confirm_user_interest(state: State) -> dict:
    """ Confirm if the user is interested in the job opening based on their last message."""
    print("state[messages] in check user interest -->", state["messages"])
    prompt = (
        "You are a HR Agent. You asked the user if he is interested in the Job Openinings at Assuretrac Consulting for the Data Engineering role. "
        "You should analyse the users response and decide wheather the user is interested in the job or not. "
        "If the user is intrested, Please respond with 'yes' and if not, respond with 'no'. Please do not say any other words."
        "Here is the user response: " + state["messages"][-1].content
    )
    res = llm.invoke([HumanMessage(content=prompt)])
    print("confirm_user_interest res -->", res.content)
    if res.content.lower().strip() == 'yes':
        return {"messages": [AIMessage(content='Thank you for your interest! Can you please provide these personal details ? Name, Email and Phone Number?')]}
    else:
        return {"messages": [AIMessage(content='No problem, we will come back to you later. Have a great day!')]}

def extract_user_details(state: State) -> dict:
    """ Extract user details (full name, email, phone) from the last user message and inform to the user back."""
    text = state["messages"][-1].content
    prompt = f"Extract Full Name, Email and Phone Number from: User Response: {text}"
    res = llm.with_structured_output(UserDetails).invoke([HumanMessage(content=prompt)])
    user_details = res.model_dump()
    # STORE into the state for later checks
    state["user_details"] = user_details
    # with open("user_details.json", "w") as f:
    #     json.dump(user_details, f, indent=4)

    # Create an Excel with only personal sheet
    # excel_bytes = create_excel_bytes(personal=user_details, tech_stack_list=None)

    # subject = "User Personal Details"
    # body = "Hello,\n Attached are the personal details provided by the user."

    # Send email in background
    # send_email_with_attachment_background(subject, body, excel_bytes, "personal_details.xlsx")

    return {"messages": [AIMessage(content=f"Got your details: {user_details}. Please do confirm if the details are correct.")]}

def confirm_user_details(state: State) -> dict:
    """ Validates with the user if the personal details (name, email, phone) are correct and asks the technical details of user."""
    print("state[messages] in confirm user details -->", state["messages"])
    
    prompt = (
        "You are an HR Agent. You have just repeated the user's personal details (name, email, phone) back to them "
        "and asked them to confirm if the details are correct. "
        "Analyze ONLY the user's latest response and decide whether they confirmed that the details are correct or not. "
        "If the user confirms (e.g., 'yes', 'correct', 'that's right', 'looks good', 'confirmed'), respond with 'yes'. "
        "If the user indicates the details are wrong or needs to be changed, respond with 'no'. "
        "Do NOT output anything other than 'yes' or 'no'. Nothing else, no extra words or explanations.\n\n"
        "Here is the user's latest response: " + state["messages"][-1].content
    )
    
    res = llm.invoke([HumanMessage(content=prompt)])
    print("confirm_user_details res -->", res.content)
    
    if res.content.lower().strip() == 'yes':
        prompt_internal = """Below is what i would like to ask the user and i want you to understand the context and rephrase the question.\n
                            I would like to ask the user this :: --> "Great! Could you please share your technical skills or tech stack with the rating you can give out of 10 for each tech stack and any explanation about your work experience in that that tech stack?. 
                            Please rephrase the question in a way that it is more clear and easy to understand for the user. Just return the rephrased question without any additional text or explanation. Only the rephrased question, nothing else.\n"""
        res_internal = llm.invoke([HumanMessage(content=prompt_internal)])
        print("confirm_user_details rephrased question -->", res_internal.content)
        return {
            "messages": [
                AIMessage(content=res_internal.content.strip())
            ]
        }
    else:
        return {
            "messages": [
                AIMessage(content="Alright, could you please provide the correct Name, Email, and Phone Number?")
            ]
        }


def fetch_user_tech_stack_node(state: State) -> dict:
    """ Extracts user's tech stack details based on the technical details we got from user."""
    print("state[messages] in fetch_user_tech_stack_node -->", state["messages"])

    user_input = state["messages"][-1].content

    model_with_structured_output = llm.with_structured_output(TechStackResponse)

    prompt = f"""
        You are an assistant that extracts a user's technical skills, years of experience, ratings, 
        and brief explanations from their message. Always return valid JSON in the schema provided.

        User message:
        {user_input}
        """

    result: TechStackResponse = model_with_structured_output.invoke(prompt)

    print("Extracted tech stack:", result.model_dump_json())

    tech_stack_py = result.model_dump()
    tech_list = tech_stack_py.get("tech_stacks", [])

    # with open("tech_stack_details.json", "w") as f:
    #     json.dump(result.model_dump(), f, indent=4)

    state["tech_stack_details"] = tech_list

    # Create Excel with only tech sheet
    # excel_bytes = create_excel_bytes(personal=None, tech_stack_list=tech_list)

    # subject = "User Technical Details"
    # body = "Attached are the technical details provided by the user."

    # # Send email in background
    # send_email_with_attachment_background(subject, body, excel_bytes, "technical_details.xlsx")


    return {
        "messages": [
            AIMessage(content=f"We have saved your tecnical details. We will come back to you with the next steps. Thank you for sharing your technical details. Have a great day!"),
        ],
        "tech_stack": result.model_dump_json()
    }

def ask_user_if_interested(state: State) -> dict:
    user_greeting = state["messages"][-1].content
    prompt = "You will receive a greeting message like Hi or Hello from the user. You should in return greet the user politely and then You should ask the user if he is interested in the Job Openings at Assuretrac Consulting for the Data Engineering role. Here is the user greeting : " + user_greeting
    print("ask_user_if_interested user_greeting -->", user_greeting)
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=res.content.lower().strip())]}  

def greet_bye(state: State) -> dict:
    return {"messages": [AIMessage(content="Thank you for your time. Goodbye!")]}

# --- Build Graph ---
def build_hr_agent_graph():
    graph = StateGraph(State)
    graph.add_node("ask_user_if_interested", ask_user_if_interested)
    graph.add_node("naviagate_to_respective_node", naviagate_to_respective_node)
    graph.add_node("confirm_user_interest", confirm_user_interest)
    graph.add_node("extract_user_details", extract_user_details)
    graph.add_node("confirm_user_details", confirm_user_details)
    graph.add_node("fetch_user_tech_stack_node", fetch_user_tech_stack_node)
    graph.add_node("greet_bye", greet_bye)

    graph.add_conditional_edges(
        "naviagate_to_respective_node",
        navigate_as_per_node_navigation,
        {
            "ask_user_if_interested": "ask_user_if_interested",
            "confirm_user_interest": "confirm_user_interest",
            "extract_user_details": "extract_user_details",
            "confirm_user_details": "confirm_user_details",
            "fetch_user_tech_stack_node": "fetch_user_tech_stack_node",
            "greet_bye": "greet_bye"
        }
    )

    graph.add_edge("confirm_user_interest", END)
    graph.add_edge("extract_user_details", END)
    graph.add_edge("ask_user_if_interested", END)
    graph.add_edge("confirm_user_details", END)
    graph.add_edge("fetch_user_tech_stack_node", END)
    graph.add_edge("greet_bye", END)
    graph.set_entry_point("naviagate_to_respective_node")
    return graph

compile_graph = build_hr_agent_graph().compile()

# --- FastAPI + Auto-ngrok ---
app = FastAPI()
# public_url = ngrok.connect(8000).public_url
# print(f"Public ngrok URL: {public_url}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://assuretrac-hr-assistant.onrender.com", "http://localhost:3000"],  # change for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageInput(BaseModel):
    message: str

conversation_states = {}

@app.post("/chat")
async def chat_with_agent(msg: MessageInput):
    user_id = "default_user"
    if user_id not in conversation_states:
        conversation_states[user_id] = {"messages": []}

    conversation_states[user_id]["messages"].append(HumanMessage(content=msg.message))

    final_state = compile_graph.invoke(conversation_states[user_id])

    conversation_states[user_id] = final_state

    ai_msgs = [m for m in final_state["messages"] if isinstance(m, AIMessage) ]
    reply = ai_msgs[-1].content if ai_msgs else "No reply."

    return {"reply": reply}


@app.get("/")
async def root():
    return {"status": "FastAPI backend is live"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


