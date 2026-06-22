import os
from typing import Annotated
from typing_extensions import TypedDict

# 1. LOAD ENVIRONMENT VARIABLES FIRST
from dotenv import load_dotenv

import logging
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

# Fetch configurations directly from the environment variables loaded from .env
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AWS_MEMORY_ID = os.getenv("AWS_MEMORY_ID")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

# Instantiate Azure OpenAI Model Client
# The SDK automatically detects AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_VERSION from your environment
llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    temperature=0.5
)

# Instantiate the AWS AgentCore Checkpointer
# The underlying boto3 client automatically uses AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from your environment
memory_saver = AgentCoreMemorySaver(
    memory_id=AWS_MEMORY_ID,
    region_name=AWS_REGION
)


# ==========================================
# 3. DEFINE AGENT STATE & PROMPT TEMPLATE
# ==========================================

# Define graph conversation state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# Custom Chat Prompt
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

# Node executing the Azure OpenAI invocation chain
def call_agent_model(state: AgentState):
    chain = prompt_template | llm
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


# Initialize state workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_agent_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# Compile graph using AWS AgentCore for memory persistence
agent_app = workflow.compile(checkpointer=memory_saver)

# ==========================================
# 5. CROSS-CLOUD EXECUTION AND TESTING
# ==========================================

if __name__ == "__main__":
    # Isolate conversation flow with a unique thread session ID
    config = {
        "configurable": {
            "thread_id": "azure-to-aws-env-session-888",
            "actor_id": "user-muneer-ahmed"  # Added actor_id to satisfy AWS AgentCore
        }
    }

    print("--- Starting Turn 1 (Calling Azure OpenAI via .env settings) ---")
    turn_1_input = {
        "messages": [
            HumanMessage(content="Hello! Remember my favorite color is orange.")
        ]
    }

    # Stream Azure response while AWS updates memory records
    for event in agent_app.stream(turn_1_input, config=config):
        for node, value in event.items():
            print(f"[{node}]: {value['messages'][-1].content}\n")

    print("--- Starting Turn 2 (Testing Context Retrieval via AWS) ---")
    turn_2_input = {
        "messages": [
            HumanMessage(content="What is my favorite color?")
        ]
    }

    for event in agent_app.stream(turn_2_input, config=config):
        for node, value in event.items():
            print(f"[{node}]: {value['messages'][-1].content}\n")
