import os
import time
import json
import boto3
import uuid
import logging
from openai import AzureOpenAI

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

MEMORY_ARN = os.getenv("AWS_MEMORY_ID")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AWS_MEMORY_ID = os.getenv("AWS_MEMORY_ID")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

memory_client = boto3.client("bedrock-agentcore", region_name="us-east-1")


def log_aws_call(request, **kwargs):
    """Print only the JSON request payload sent to AWS."""
    if not request.body:
        return
    body = request.body if isinstance(request.body, str) else request.body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(body)
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print(body)


memory_client.meta.events.register("before-send.bedrock-agentcore.*", log_aws_call)

# Azure OpenAI client (just for inference, no memory logic here)
model = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
)


def load_history(actor_id, session_id):
    """Fetch prior turns from AgentCore Memory and convert to OpenAI message format."""
    resp = memory_client.list_events(
        memoryId=MEMORY_ARN,
        actorId=actor_id,
        sessionId=session_id,
        maxResults=100,
        includePayloads=True,
    )

    messages = []
    events = sorted(resp.get("events", []), key=lambda e: e.get("eventTimestamp", 0))

    for event in events:
        for payload in event.get("payload", []):
            blob = payload.get("blob")
            if not blob:
                continue
            try:
                data = json.loads(blob)
                if "role" in data and "content" in data:
                    messages.append({"role": data["role"], "content": data["content"]})
            except (json.JSONDecodeError, TypeError):
                continue

    return messages


def save_turn(actor_id, session_id, user_message, assistant_reply):
    """Persist both sides of a turn in a single CreateEvent call."""
    memory_client.create_event(
        memoryId=MEMORY_ARN,
        actorId=actor_id,
        sessionId=session_id,
        eventTimestamp=time.time(),
        payload=[
            {"blob": json.dumps({"role": "user", "content": user_message})},
            {"blob": json.dumps({"role": "assistant", "content": assistant_reply})},
        ],
    )


def retrieve_memories(actor_id, query, max_results=5):
    """Search extracted long-term memory records using semantic search."""
    namespace = f"/strategies/preference_builtin_tk5kv-wvtf58AZmj/actors/{actor_id}/"
    try:
        resp = memory_client.retrieve_memory_records(
            memoryId=MEMORY_ARN,
            namespace=namespace,
            searchCriteria={
                "searchQuery": query,
                "topK": max_results,
            },
        )
        summaries = resp.get("memoryRecordSummaries", [])
        print(f"\n>>> Retrieved {len(summaries)} memory record(s):")
        print(json.dumps(summaries, indent=2, default=str))
        return summaries
    except Exception as e:
        print(f"[warning] Failed to retrieve memory records: {e}")
        return []


def chat(actor_id, session_id, user_message, history):
    """Send one turn to Azure OpenAI, using both short-term history and long-term memory."""

    resp = memory_client.list_memory_extraction_jobs(memoryId=MEMORY_ARN)
    print(json.dumps(resp, indent=2, default=str))

    # 1. Retrieve relevant long-term facts based on the current message
    memories = retrieve_memories(actor_id, query=user_message, max_results=5)

    # 2. Turn them into a system prompt
    system_message = None
    if memories:
        facts = "\n".join(f"- {m.get('content', {}).get('text', '')}" for m in memories)
        system_message = {
            "role": "system",
            "content": f"Known facts about this user from previous conversations:\n{facts}",
        }

    # 3. Build the full message list: system facts + short-term turn history + new message
    messages = ([system_message] if system_message else []) + history + [
        {"role": "user", "content": user_message}
    ]

    response = model.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.5,
    )
    return response.choices[0].message.content


def main():
    actor_id = input("Enter your name (used as actor id): ").strip().replace(" ", "-").lower()
    session_id = f"session-{uuid.uuid4().hex[:8]}"

    print("=" * 50)
    print(f"Chat Session Started (ID: {session_id})")
    print("Type your message and press Enter. Enter '0' to quit.")
    print("=" * 50)

    # Load any prior history for this actor/session (will be empty for a new session)
    history = load_history(actor_id, session_id)

    while True:
        user_message = input("\nYou: ").strip()
        if user_message == "0":
            print("Session ended.")
            break
        if not user_message:
            continue

        try:
            reply = chat(actor_id, session_id, user_message, history)
        except Exception as e:
            print(f"[error] Failed to get response: {e}")
            continue

        print(f"\n[agent]: {reply}")

        # Update local in-memory history for this run
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})

        # Persist the turn to AWS in one call
        try:
            save_turn(actor_id, session_id, user_message, reply)
        except Exception as e:
            print(f"[warning] Failed to save turn to memory: {e}")


if __name__ == "__main__":
    main()
