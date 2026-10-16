"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the project instructions and rubric for guidance.
Work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider,
    AfterInvocationEvent,
    HookRegistry,
    MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser
import shutil
import stat
from pathlib import Path
import playwright

_playwright_node = Path(playwright.__file__).parent / "driver" / "node"
_runtime_node = Path("/tmp/node")

if _playwright_node.exists() and not _runtime_node.exists():
    shutil.copy2(_playwright_node, _runtime_node)
    _runtime_node.chmod(
        _runtime_node.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )

os.environ["PLAYWRIGHT_NODEJS_PATH"] = str(_runtime_node)
from sympy import content

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# Replace the placeholder strings with your actual AWS resource values.
# You collected these in the infrastructure setup section of the project instructions.
#
# GATEWAY_URL format: https://<alias>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
# This starter uses an unsigned MCP connection and therefore assumes the
# project Gateway is configured with the NONE authorizer.
# KB_ID       format: 10-character alphanumeric string from the KB console
# REGION:     your AWS region, e.g. "us-east-1"
# MEMORY_ID   format: shown in the AgentCore Memory console

GATEWAY_URL = "https://customer-support-gateway-b1pdf7l580.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID = "IFU2D422VU"
REGION = "us-east-1"
MEMORY_ID = "CustomerSupportMemory-HFwV0xFC8m"


model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
# Implement get_namespaces() to return a dict mapping strategy type to
# namespace template string.
#
# Steps:
#   1. Call mem_client.get_memory_strategies(memory_id) to get strategy list
#   2. Read the namespace from strategy["namespaceTemplates"][0].
#      For compatibility with older AgentCore responses, fall back to
#      strategy["namespaces"][0] when namespaceTemplates is absent.
#
# Example output:
#   { "SEMANTIC": "cs_agent/{actorId}/facts",
#     "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }


def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)
    namespaces = {}

    for strategy in strategies:
        # Current API uses "type"; keep "strategyType" as backward compatibility.
        strategy_type = strategy.get("type") or strategy.get("strategyType")

        if "namespaceTemplates" in strategy and strategy["namespaceTemplates"]:
            ns = strategy["namespaceTemplates"][0]
        elif "namespaces" in strategy and strategy["namespaces"]:
            ns = strategy["namespaces"][0]
        else:
            continue

        if strategy_type:
            namespaces[strategy_type] = ns

    return namespaces


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
# Implement MemoryHook, a HookProvider subclass that adds long-term memory.
#
# The class needs:
#   __init__(self, actor_id, session_id, memory_client, memory_id)
#     — store all four as instance attributes
#     — call get_namespaces() and store the result as self.namespaces
#
#   retrieve_customer_context(self, event: MessageAddedEvent)
#     — only runs for plain-text user messages (not tool results)
#     — for each strategy namespace, call memory_client.retrieve_memories(
#          memory_id, namespace (formatted with actorId), query, top_k=5)
#     — collect non-empty memory texts tagged with their strategy type
#     — if any memories found, prepend them to the user message as:
#          "Customer Context:\n<memories>\n\n<original_message>"
#
#   save_support_interaction(self, event: AfterInvocationEvent)
#     — walk the message list backwards to find the last plain-text user
#       query and the last assistant response
#     — call memory_client.create_event(memory_id, actor_id, session_id,
#          messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")])
#
#   register_hooks(self, registry: HookRegistry)
#     — register retrieve_customer_context on MessageAddedEvent
#     — register save_support_interaction on AfterInvocationEvent


class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)


    def retrieve_customer_context(self, event: MessageAddedEvent):
        logger.warning("MEMORY RETRIEVAL HOOK FIRED")

        message = event.message
        logger.warning(
            "MEMORY EVENT: role=%s content_type=%s",
            message.get("role"),
            type(message.get("content")).__name__,
        )

        if message.get("role") != "user":
            logger.warning("MEMORY RETRIEVAL SKIPPED: message is not from user")
            return

        content_blocks = message.get("content")
        if not isinstance(content_blocks, list) or not content_blocks:
            logger.warning("MEMORY RETRIEVAL SKIPPED: no content blocks")
            return

        text_block = content_blocks[0]
        if not isinstance(text_block, dict) or not isinstance(
            text_block.get("text"), str
        ):
            logger.warning("MEMORY RETRIEVAL SKIPPED: first block has no text")
            return

        query = text_block["text"]
        logger.warning("MEMORY QUERY TEXT: %s", query)
        memories = []

        logger.warning(
            "MEMORY RETRIEVAL START: actor=%s namespaces=%s",
            self.actor_id,
            list(self.namespaces.keys()),
        )

        for strategy_type, ns_template in self.namespaces.items():
            namespace = ns_template.replace(
                "{actorId}", self.actor_id
            ).replace("{sessionId}", self.session_id)

            try:
                retrieved = self.memory_client.retrieve_memories(
                    self.memory_id,
                    namespace,
                    query,
                    top_k=5,
                )

                logger.warning(
                    "MEMORY SEARCH RESULT: strategy=%s count=%d namespace=%s",
                    strategy_type,
                    len(retrieved),
                    namespace,
                )

                for mem in retrieved:
                    if not isinstance(mem, dict):
                        continue

                    content = mem.get("content", {})
                    mem_text = (
                        content.get("text", "").strip()
                        if isinstance(content, dict)
                        else ""
                    )

                    if mem_text:
                        memories.append(f"[{strategy_type}] {mem_text}")

            except Exception:
                logger.exception(
                    "Failed to retrieve memories for strategy=%s namespace=%s",
                    strategy_type,
                    namespace,
                )

        if memories:
            text_block["text"] = (
                "Customer Context:\n"
                + "\n".join(memories)
                + "\n\nUser Request:\n"
                + query
            )
            logger.warning(
                "MEMORY CONTEXT INJECTED: count=%d",
                len(memories),
            )
        else:
            logger.warning("MEMORY CONTEXT NOT INJECTED: no usable memories found")



    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        logger.warning("MEMORY SAVE HOOK FIRED")
        messages = event.agent.messages
        customer_query = None

        # Find the latest user message.
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue

            content = msg.get("content")
            if not isinstance(content, list):
                continue

            text = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("text")
            ).strip()

            if text:
                if "Customer Context:" in text:
                    parts = text.split("\n\n")
                    customer_query = parts[-1] if len(parts) > 1 else text
                else:
                    customer_query = text
                break

        # Extract the final response from this invocation.
        agent_response = None
        if event.result:
            message = event.result.message
            content = message.get("content", [])

            if isinstance(content, list):
                agent_response = "\n".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("text")
                ).strip()

        logger.warning(
            "MEMORY SAVE DATA: query=%s response=%s",
             bool(customer_query),
              bool(agent_response),
        )

        if customer_query and agent_response:
            try:
                self.memory_client.create_event(
                    self.memory_id,
                    self.actor_id,
                    self.session_id,
                    messages=[
                        (customer_query, "USER"),
                        (agent_response, "ASSISTANT"),
                    ],
                )
                logger.info(
                    "Saved support interaction for actor %s",
                    self.actor_id,
                )
            except Exception as e:
                logger.error("Failed to save interaction to memory: %s", e)



    def register_hooks(self, registry: HookRegistry) -> None:
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
# Implement search_knowledge_base(query) using the @tool decorator.
#
# Steps:
#   1. Guard: if KB_ID is empty return "Knowledge base not configured."
#   2. Call _bedrock_runtime.retrieve(
#          knowledgeBaseId=KB_ID,
#          retrievalQuery={"text": query}
#      )
#   3. Extract resp["retrievalResults"]; return a message if empty
#   4. Join the text chunks with "\n---\n" and return the result
#
# The docstring is the tool description — the model uses it to decide when
# to call this tool, so keep it clear and accurate.
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty information,
    loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    # 1. Guard: if KB_ID is empty return a message
    if not KB_ID or KB_ID == "<kbid>":
        return "Knowledge base not configured."

    try:
        # 2. Call bedrock-agent-runtime retrieve API
        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID, retrievalQuery={"text": query}
        )

        # 3. Extract retrievalResults; return a message if empty
        results = response.get("retrievalResults", [])
        if not results:
            return "No relevant information found in the knowledge base."

        # 4. Join the text chunks with "\n---\n" and return the result
        text_chunks = [res.get("content", {}).get("text", "") for res in results]
        return "\n---\n".join([chunk for chunk in text_chunks if chunk])

    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        return f"Error searching knowledge base: {str(e)}"


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
# Implement calculate_loyalty_discount() using the @tool decorator.
#
# The tool must:
#   1. Build a self-contained Python code string that:
#        • Defines earn_rates: {"standard": 1, "device": 2, "fresh": 5}
#        • Defines tier_rates: {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
#        • Calculates points_redeemed (floor to nearest 500, cap at 50% of order)
#        • Calculates tier_discount (applied to subtotal after points)
#        • Calculates final_total, total_savings, points_earned, remaining_points
#        • Prints a JSON result dict
#   2. Execute the code with code_session(REGION).invoke("executeCode", {...})
#      using language="python" and clearContext=True
#   3. Return the first result event as a JSON string
#   4. Include a fallback that computes only the tier discount if the
#      Code Interpreter is unavailable


@tool
def calculate_loyalty_discount(
    order_subtotal: float,
    loyalty_tier: str,
    points_available: int,
    product_category: str = "standard",
) -> str:
    """
    Calculate customer loyalty discount and final order total using
    the secure code execution environment.

    Args:
        order_subtotal: The initial subtotal of the order.
        loyalty_tier: The customer's loyalty tier (Silver, Gold, or Platinum).
        points_available: The number of loyalty points available.
        product_category: Product category: standard, device, or fresh.

    Returns:
        JSON string containing the calculation results.
    """

    code = f"""
import json

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

subtotal = {order_subtotal}
tier = {json.dumps(loyalty_tier)}
points = {points_available}

# Earn points according to product category.
category = {json.dumps(product_category.lower())}
earn_rate = earn_rates.get(category, earn_rates["standard"])

# Points can cover at most 50% of the subtotal.
# 1 point = $0.01.
max_points_value = subtotal * 0.50
max_points = int(max_points_value * 100)

redeemable_points = min(points, max_points)

# Redemption must be in multiples of 500 points.
points_redeemed = (redeemable_points // 500) * 500
points_discount = points_redeemed * 0.01

# Apply the tier discount AFTER the points discount.
after_points = subtotal - points_discount

tier_rate = tier_rates.get(tier, 0.00)
tier_discount = after_points * tier_rate

final_total = max(0.0, after_points - tier_discount)
points_earned = int(final_total * earn_rate)
total_savings = points_discount + tier_discount

remaining_points = points - points_redeemed + points_earned

print(json.dumps({{
    "points_earned": points_earned,
    "points_redeemed": points_redeemed,
    "remaining_points": remaining_points,
    "points_discount": round(points_discount, 2),
    "tier_discount_pct": round(tier_rate * 100, 2),
    "tier_discount": round(tier_discount, 2),
    "final_total": round(final_total, 2),
    "total_savings": round(total_savings, 2)
}}))
"""

    try:
        with code_session(REGION) as session:
            response = session.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )

            events = list(response["stream"])

            if not events:
                return json.dumps(
                    {"error": "No result returned from Code Interpreter."}
                )

            event = events[-1]
            result = event.get("result", {})

            if result.get("isError"):
                return json.dumps(
                    {"error": result.get("content", "Code execution failed.")}
                )

            content = result.get("content", [])

            if content and content[0].get("text") is not None:
                return content[0]["text"]

            return json.dumps({"error": "Code Interpreter returned no text result."})

    
    except Exception as e:
        logger.warning(
            f"Code Interpreter unavailable; using tier-discount fallback: {e}"
        )

        tier_rates = {
            "Silver": 0.00,
            "Gold": 0.10,
            "Platinum": 0.15,
        }

        try:
            subtotal = float(order_subtotal)
            points = int(points_available)

            if subtotal < 0 or points < 0:
                return json.dumps({
                    "error": "Subtotal and available points cannot be negative."
                })

            tier_rate = tier_rates.get(loyalty_tier, 0.00)
            tier_discount = round(subtotal * tier_rate, 2)
            final_total = round(subtotal - tier_discount, 2)

            return json.dumps({
                "fallback_used": True,
                "calculation_scope": "tier_discount_only",
                "order_subtotal": round(subtotal, 2),
                "loyalty_tier": loyalty_tier,
                "tier_discount_pct": round(tier_rate * 100, 2),
                "tier_discount": tier_discount,
                "final_total": final_total,
                "points_earned": 0,
                "points_redeemed": 0,
                "remaining_points": points,
                "note": (
                    "Points redemption and balance calculation are "
                    "unavailable because Code Interpreter failed."
                )
            })

        except (TypeError, ValueError):
            logger.exception("Tier-discount fallback failed.")
            return json.dumps({
                "error": "Invalid input for tier-discount fallback."
            })

    
        

# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
# Implement the invoke() function decorated with @app.entrypoint.
#
# Steps:
#   1. Extract user_input, actor_id, and session_id from the payload
#      (generate a UUID if session_id is missing)
#   2. Instantiate MemoryHook for this actor/session
#   3. Instantiate AgentCoreBrowser(region=REGION)
#   4. Build the tools list: [search_knowledge_base, calculate_loyalty_discount,
#                              agent_core_browser.browser]
#   5. Connect to the Gateway via MCPClient, load gateway_tools, extend tools list
#   6. Create and invoke the Agent with all tools, hooks, and system_prompt
#   7. Return the text from the first content block of the response
#   8. Handle exceptions gracefully


@app.entrypoint
async def invoke(payload, context=None):
    prompt = payload.get("prompt", "Hello")
    actor_id = payload.get("customer_id", "default-customer")
    session_id = payload.get("session_id", str(uuid.uuid4()))

    # 1. Memory hook
    memory_hook = MemoryHook(
        actor_id=actor_id,
        session_id=session_id,
        memory_client=memory_client,
        memory_id=MEMORY_ID,
    )

    # 2. Browser
    browser = AgentCoreBrowser(region=REGION)

    # 3. Local tools
    tools = [
        search_knowledge_base,
        calculate_loyalty_discount,
        browser.browser,
    ]

    # 4. Gateway tools
    try:
        mcp_client = MCPClient(url=GATEWAY_URL)
        gateway_tools = await mcp_client.load_tools()
        tools.extend(gateway_tools)

        logger.warning(
            "Loaded Gateway tools: %s", [tool.tool_name for tool in gateway_tools]
        )

    except Exception as e:
        logger.error(f"Failed to load gateway tools: {e}")

    # 5. Agent
    agent = Agent(
        model=model,
        tools=tools,
        hooks=[memory_hook],
        system_prompt=(
            "You are a helpful customer support agent for an e-commerce platform. "
            "Always understand the user's actual request before choosing a tool or taking an action. "
            "If the user asks about an order, use the order-tracking tools to retrieve the relevant order information "
            "and answer the question directly. "
            "Do NOT assume that mentioning an order means the customer wants a return or refund. "
            "Only use refund tools when the customer explicitly asks for a refund or return. "
            "Before initiating a refund, retrieve the order using the order-tracking tool "
            "and identify the actual order amount from the returned order details. "
            "The order-tracking tool returns the order price in the 'total' field. "
            "When calling initiate_refund, pass that exact numeric value as the 'amount' argument. "
            "Never initiate a refund with an amount of $0.00 when the actual order amount is available. "
            "For customer information, use the customer tools. "
            "For product information or company policies, use the knowledge base. "
            "For loyalty discount calculations, always call the loyalty discount tool "
            "and use its returned JSON values as the source of truth. "
            "Do not independently recalculate or override any returned amount. "
            "Report tier_discount_pct, tier_discount, points_discount, final_total, "
            "points_redeemed, points_earned, remaining_points, and total_savings "
            "exactly as returned when those fields are available. "
            "Explain that the tier discount is calculated after the points discount. "
            "If the tool returns fallback_used=true, clearly explain that only the "
            "tier discount was calculated and points redemption was unavailable. "
            "Use the browser only when the user explicitly asks you to browse the web. "
            "Do not perform actions that the customer did not request."
        ),
    )

    # 6. Invoke
    try:
        logger.warning("ACTUAL PROMPT SENT TO AGENT: %r", prompt)
        result = agent(prompt)
        return {"response": str(result)}

    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        return {"error": str(e)}


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    """Run one invocation from the command line for local testing."""
    # parser = argparse.ArgumentParser()
    # parser.add_argument("payload", type=str)
    # args = parser.parse_args()
    payload = {
        "prompt": "Hi, I am Jane. I prefer concise responses.",
        "customer_id": "CUST-123",
        "session_id": "s-A",
    }
    response = asyncio.run(invoke(payload))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    #main()
