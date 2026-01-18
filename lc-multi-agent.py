"""
LangChain 1.2.6 | ServiceMax Multi-Agent Orchestrator (Simulated, NO real model calls)

Pipeline:
1) SchedulingAgentTool -> returns today's work order id
2) ServiceInsightsAgentTool -> returns work order details + product_id
3) KnowledgeAccessAgentTool -> returns documentation steps using product_id
4) Orchestrator combines everything into final answer

This uses FakeListLLM so you can run it locally without OpenAI calls.
"""

from __future__ import annotations


from langchain_core.language_models import FakeListLLM
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_agent


# ============================================================
# Mock backend services (replace these with real ServiceMax APIs)
# ============================================================

def scheduling_service(_: str) -> str:
    return (
        "Scheduled Today:\n"
        "work_order_id=WO-100245\n"
        "technician=Tech-01\n"
        "start_time=10:00 AM\n"
        "site=Acme Plant - San Jose\n"
    )


def service_insights_service(work_order_id: str) -> str:
    return (
        f"Insights for {work_order_id}:\n"
        "product_id=PROD-77881\n"
        "asset=Packaging Line Conveyor A7\n"
        "summary=Intermittent belt slipping and debris buildup near rollers\n"
        "history=\n"
        "- 2025-11-28: Belt tension adjusted; debris removed near drive roller\n"
        "- 2025-09-10: Preventive maintenance completed; lubrication applied\n"
    )


def knowledge_access_service(product_id: str) -> str:
    return (
        f"Documentation for {product_id}:\n"
        "title=Conveyor Cleanup & Basic Care (Quick Guide)\n"
        "steps=\n"
        "1) Power down the machine and follow lockout/tagout\n"
        "2) Inspect belt + roller areas for debris buildup\n"
        "3) Use non-abrasive cloth + approved cleaner\n"
        "4) Clear debris near drive + idler rollers with a soft brush\n"
        "5) Check belt tension + alignment after cleaning\n"
        "6) Restart and verify smooth motion (no slipping)\n"
    )


# ============================================================
# Tools
# ============================================================

SchedulingTool = Tool(
    name="SchedulingAgentTool",
    func=scheduling_service,
    description="Find what work order is scheduled today. Input can be technician name or id.",
)

ServiceInsightsTool = Tool(
    name="ServiceInsightsAgentTool",
    func=service_insights_service,
    description="Get work order details including product_id, summary, and service history. Input must be a work_order_id.",
)

KnowledgeAccessTool = Tool(
    name="KnowledgeAccessAgentTool",
    func=knowledge_access_service,
    description="Get documentation / cleanup steps for a product. Input must be a product_id.",
)

TOOLS = [SchedulingTool, ServiceInsightsTool, KnowledgeAccessTool]


# ============================================================
# ReAct Prompt
# ============================================================

ORCHESTRATOR_PROMPT = PromptTemplate.from_template(
    """
You are a ServiceMax Orchestrator Agent.

You MUST do this workflow:
1) Use SchedulingAgentTool to find the work order scheduled today.
2) Extract the work_order_id from the scheduling result.
3) Use ServiceInsightsAgentTool with that work_order_id to get product_id and details.
4) Extract product_id from the insights result.
5) Use KnowledgeAccessAgentTool with that product_id to get cleanup documentation.
6) Return a final combined answer with schedule + insights + steps.

You have access to these tools:
{tools}

Use this format:

Question: {input}
Thought: (your reasoning)
Action: (one of [{tool_names}])
Action Input: (tool input)
Observation: (tool result)
... repeat as needed ...
Thought: I now have everything I need
Final: (combined answer)

Question: {input}
{agent_scratchpad}
""".strip()
)


# ============================================================
# Fake LLM Script (deterministic tool calls)
# ============================================================

FAKE_REACT_STEPS = [
    "Thought: I should first find today's scheduled work order.\n"
    "Action: SchedulingAgentTool\n"
    "Action Input: Tech-01",

    "Thought: Now I need details and the product_id for that work order.\n"
    "Action: ServiceInsightsAgentTool\n"
    "Action Input: WO-100245",

    "Thought: Now I will fetch documentation for the product.\n"
    "Action: KnowledgeAccessAgentTool\n"
    "Action Input: PROD-77881",

    "Thought: I now have everything I need.\n"
    "Final: Here is the combined summary for today’s scheduled work order with insights and cleanup steps."
]


# ============================================================
# Run
# ============================================================

def main() -> None:
    user_question = (
        "What work order is scheduled today? Tell me something more about that work order, "
        "and also give me some help from the documentation to clean up that machine."
    )

    llm = FakeListLLM(responses=FAKE_REACT_STEPS)

    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=ORCHESTRATOR_PROMPT.__str__(),
    )

    result = agent.invoke({"messages": [HumanMessage(content=user_question)]})

    print("\n" + "=" * 90)
    print("FINAL USER OUTPUT")
    print("=" * 90)
    print(result["output"])


if __name__ == "__main__":
    main()
