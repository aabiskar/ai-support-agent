import asyncio

from strands import Agent
from strands.models import BedrockModel
from strands_tools.browser import AgentCoreBrowser
from strands.tools.mcp.mcp_client import MCPClient

from main import (
    search_knowledge_base,
    calculate_loyalty_discount,
    MemoryHook,
    memory_client,
    MEMORY_ID,
)


GATEWAY_URL = "https://customer-support-gateway-b1pdf7l580.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"


async def main():
    browser = AgentCoreBrowser(region="us-east-1")

    mcp_client = MCPClient(url=GATEWAY_URL)
    gateway_tools = await mcp_client.load_tools()

    print("Loaded MCP tools:")
    for tool in gateway_tools:
        print("-", tool.tool_name)

    memory_hook = MemoryHook(
        actor_id="CUST-999-TEST",
        session_id="full-combination-test-001",
        memory_client=memory_client,
        memory_id=MEMORY_ID,
    )

    model = BedrockModel(
        model_id="global.amazon.nova-2-lite-v1:0"
    )

    agent = Agent(
        model=model,
        tools=[
            search_knowledge_base,
            calculate_loyalty_discount,
            browser.browser,
            *gateway_tools,
        ],
        hooks=[memory_hook],
        system_prompt=(
            "You are a helpful customer support agent. "
            "Answer the user's request using the appropriate available tool. "
            "If the user asks about an order, use the order-tracking tools. "
            "Do not assume the user wants a refund or return. "
            "Only use refund tools when the user explicitly asks for a refund or return. "
            "Use the knowledge base for product and company policy questions. "
            "Use the loyalty tool for loyalty discount calculations. "
            "Use the browser only when the user explicitly asks you to browse the web."
        ),
    )

    result = agent("Check the refund status for refund ID TEST-REFUND-001.")

    print("\nRESULT:")
    print(result)


asyncio.run(main())