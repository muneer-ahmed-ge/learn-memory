import os
import time
import json
import boto3
import uuid
from openai import AzureOpenAI

from dotenv import load_dotenv
load_dotenv()

MEMORY_ARN=os.getenv("AWS_MEMORY_ID")
AZURE_OPENAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
AWS_MEMORY_ID = os.getenv("AWS_MEMORY_ID")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

# AWS Bedrock AgentCore Memory client
memory_client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)

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


def chat(actor_id, session_id, user_message, history):
    """Send one turn to Azure OpenAI, using the running in-memory history."""
    messages = history + [{"role": "user", "content": user_message}]

    response = model.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.5,
    )
    reply = response.choices[0].message.content
    return reply


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