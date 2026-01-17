from typing import TypedDict
from langgraph.graph import StateGraph, END
from registry.agent_registry import get_registered_agents


class AgentState(TypedDict, total=False):
    goal: str
    work_order_id: str
    date: str
    product_id: str
    description: str
    documentation: str


def build_graph():
    agents = get_registered_agents()

    graph = StateGraph(AgentState)

    graph.add_node("schedule", agents["scheduling"])
    graph.add_node("service_insight", agents["service_insight"])
    graph.add_node("knowledge", agents["knowledge"])

    graph.set_entry_point("schedule")
    graph.add_edge("schedule", "service_insight")
    graph.add_edge("service_insight", "knowledge")
    graph.add_edge("knowledge", END)

    return graph.compile()