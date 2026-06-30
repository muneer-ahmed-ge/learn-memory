import os
import uuid
from typing import Annotated
from typing_extensions import TypedDict
import logging

# 1. LOAD ENVIRONMENT VARIABLES FIRST
from dotenv import load_dotenv

logging.getLogger("openai").propagate = False
logging.getLogger("httpx").propagate = False
logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)
load_dotenv()  # This reads the local .env file

# LangChain Azure OpenAI Client
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangGraph Orchestration & AWS Checkpointer Backend
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph_checkpoint_aws import AgentCoreMemorySaver

# ==========================================
# 2. INITIALIZATION & CONFIGURATION
# ==========================================
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AWS_MEMORY_ID = os.getenv("AWS_MEMORY_ID")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    temperature=0.5
)

memory_saver = AgentCoreMemorySaver(
    memory_id=AWS_MEMORY_ID,
    region_name=AWS_REGION
)


# ==========================================
# 3. DEFINE AGENT STATE & PROMPT TEMPLATE
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert enterprise assistant powered by GPT-5.2. Maintain strict context awareness."
    ),
    MessagesPlaceholder(variable_name="messages")
])


# ==========================================
# 4. DEFINE GRAPH NODES AND WORKFLOW
# ==========================================
def call_agent_model(state: AgentState):
    fetched_messages = state["messages"]
    chain = prompt_template | llm
    response = chain.invoke({"messages": fetched_messages})
    return {"messages": [response]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_agent_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

agent_app = workflow.compile(checkpointer=memory_saver)

# ==========================================
# 5. CONTINUOUS REPL EXECUTION (WHILE LOOP)
# ==========================================
if __name__ == "__main__":
    unique_session_id = f"session-{uuid.uuid4().hex[:8]}"
    config = {
        "configurable": {
            "thread_id": unique_session_id,
            "actor_id": "user-muneer-ahmed"
        }
    }

    print("====================================================")
    print(f"Chat Session Started (ID: {unique_session_id})")
    print("Type your message and press Enter. Enter '0' to quit.")
    print("====================================================\n")

    while True:
        user_input = input("You: ").strip()

        # Exit criteria condition
        if user_input == "0":
            print("\nExiting chat session. Goodbye!")
            break

        if not user_input:
            continue

        # Structure the payload for LangGraph
        current_turn_input = {
            "messages": [HumanMessage(content=user_input)]
        }

        # Stream Azure response while AWS updates memory records
        for event in agent_app.stream(current_turn_input, config=config):
            for node, value in event.items():
                print(f"[{node}]: {value['messages'][-1].content}\n")
