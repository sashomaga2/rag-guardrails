"""Pretend to be the support desk: send a signed webhook event to the API.

    python scripts/agent_reply.py <conversation_id> joined --agent Ana
    python scripts/agent_reply.py <conversation_id> message "Please restart the VPN client" --agent Ana
    python scripts/agent_reply.py <conversation_id> left

Needs RAG_AGENT_WEBHOOK_SECRET (same value as the API). Use --url to target a cluster.
"""

import argparse
import hashlib
import hmac
import json
import os
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("conversation_id")
parser.add_argument("type", choices=["joined", "message", "left"])
parser.add_argument("text", nargs="?", default="")
parser.add_argument("--agent", default="Helpdesk agent")
parser.add_argument("--url", default="http://localhost:8080")
args = parser.parse_args()

secret = os.environ["RAG_AGENT_WEBHOOK_SECRET"]
body = json.dumps({
    "conversation_id": args.conversation_id,
    "type": f"agent_{args.type}",
    "text": args.text,
    "agent_name": args.agent,
}).encode()
signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
req = urllib.request.Request(
    f"{args.url}/agent/webhook", data=body, method="POST",
    headers={"Content-Type": "application/json", "X-Signature": signature},
)
with urllib.request.urlopen(req) as resp:
    print(resp.status, resp.read().decode())
