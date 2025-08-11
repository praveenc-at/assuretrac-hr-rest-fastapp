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

# Load env variables
load_dotenv()

# --- LLM Config ---
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

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

import re

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

    print("message in naviagate_to_respective_node -->", message)
    print("\n\n")
    # prompt = (
    #     "You are an orchestration assistant for a HR voice chatbot. FOLLOW THESE PRIORITY RULES (HIGHEST first):\n"
    #     "1) If the last user message is ONLY a greeting like Hi, Hello -> return 'ask_user_if_interested'.\n"
    #     "2) If the user clearly says they are NOT interested -> return 'greet_bye'.\n"
    #     "3) If the user clearly indicates interest (yes/I'm interested) in the Job oppurtunity -> return 'confirm_user_interest'.\n"
    #     "4) If the user's last message confirms he is interested in the job opening then we should extract the user info OR collect the personal details (email/phone/full name) -> return 'extract_user_details'.\n"
    #     "5) If the user's have shared personal details (email, phone, full name) in their last message, we should share back the details user shared with us and get the confirmation if the personal details are correct -> return 'confirm_user_details' . \n"
    #     "6) If the user confirms the personal details are correct, then we need to fetch the technical details so return -> 'fetch_user_tech_stack_node'.\n"
    #     "7) Otherwise inspect the full conversation and choose the node that best continues the flow.\n\n"

    #     "Here are the conversation messages which you can use to understand the context and return what is expected:\n"
    #     f" Conversation :\n {message}\n"
    #     "You must respond with EXACTLY one node name from this set by understanding the whole conversation. Do not just see the last message OR input. Please understand whole conversation and provide the node name:"
    #     "ask_user_if_interested --> If the input is a greeting alone.\n "
    #     "confirm_user_interest --> After we receive the user response to the question like 'Are you interested in the job opening?'\n"
    #     "extract_user_details --> After the user confirms he is interested in the job opening and we need to collect the personal details like Name, Email and Phone Number.\n"
    #     "confirm_user_details --> When we have collected the personal details and we need to confirm if the details are correct.\n"
    #     "fetch_user_tech_stack_node --> When we have confirmed the personal details are correct and we need to collect the technical details from the user.\n"
    #     "greet_bye --> When the user is not interested in the job opening or when we have collected all the details and we need to end the conversation.\n\n"

    #     "OUTPUT: Reply with EXACTLY one node name from: [ask_user_if_interested, confirm_user_interest, extract_user_details, confirm_user_details, fetch_user_tech_stack_node, greet_bye]\n"
    # )
    prompt = (
        "You are an orchestration assistant for an HR voice chatbot.\n"
        "Your task is to select the NEXT node to execute based on the FULL conversation.\n"
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
        "Make sure to choose the node that best fits the conversation context and follows the rules above.\n"
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

# def extract_user_details(state: State) -> dict:
#     text = state["messages"][-1].content
#     prompt = f"Extract Full Name, Email and Phone Number from: User Response: {text}"
#     res = llm.with_structured_output(UserDetails).invoke([HumanMessage(content=prompt)])
#     with open("user_details.json", "w") as f:
#         json.dump(res.dict(), f, indent=4)
#     return {"messages": [AIMessage(content=f"Got your details: {res.dict()}" + "Please do confirm if the details are correct." )]}

def extract_user_details(state: State) -> dict:
    """ Extract user details (full name, email, phone) from the last user message and inform to the user back."""
    text = state["messages"][-1].content
    prompt = f"Extract Full Name, Email and Phone Number from: User Response: {text}"
    res = llm.with_structured_output(UserDetails).invoke([HumanMessage(content=prompt)])
    user_details = res.model_dump()
    # STORE into the state for later checks
    state["user_details"] = user_details
    with open("user_details.json", "w") as f:
        json.dump(user_details, f, indent=4)
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

# def fetch_user_tech_stack_node(state: State) -> dict:
#     return {"messages": [AIMessage(content="Please share your experience, rating, and explanation for each required tech stack.")]}  

from pydantic import BaseModel, Field
from typing import List


class TechStackItem(BaseModel):
    technology: str = Field(..., description="Name of the technology")
    experience_years: float = Field(..., description="Years of experience with this technology")
    rating_out_of_10: int = Field(..., description="Self-assessed rating out of 10")
    explanation: str = Field(..., description="Brief explanation of the experience with this technology")

class TechStackResponse(BaseModel):
    tech_stacks: List[TechStackItem] = Field(..., description="List of technologies the user has worked with")


def fetch_user_tech_stack_node(state: State) -> dict:
    """ Extracts user's tech stack details based on the technical details we got from user."""
    print("state[messages] in fetch_user_tech_stack_node -->", state["messages"])

    # Last message from the user containing tech stack info
    user_input = state["messages"][-1].content

    # Create a model that outputs in structured JSON
    model_with_structured_output = llm.with_structured_output(TechStackResponse)

    # Prompt to guide extraction
    prompt = f"""
        You are an assistant that extracts a user's technical skills, years of experience, ratings, 
        and brief explanations from their message. Always return valid JSON in the schema provided.

        User message:
        {user_input}
        """

    # Invoke the structured extraction
    result: TechStackResponse = model_with_structured_output.invoke(prompt)

    print("Extracted tech stack:", result.model_dump_json())

    with open("tech_stack_details.json", "w") as f:
        json.dump(result.model_dump(), f, indent=4)

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

    # Append user message to state
    conversation_states[user_id]["messages"].append(HumanMessage(content=msg.message))

    # Run graph
    final_state = compile_graph.invoke(conversation_states[user_id])

    # Store updated state
    conversation_states[user_id] = final_state

    # Extract last assistant reply
    ai_msgs = [m for m in final_state["messages"] if isinstance(m, AIMessage) ]
    reply = ai_msgs[-1].content if ai_msgs else "No reply."

    return {"reply": reply}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


