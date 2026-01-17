# Central registry that the orchestrator discovers agents from

from agents.scheduling_agent import scheduling_agent
from agents.service_insight_agent import service_insight_agent
from agents.knowledge_agent import knowledge_agent


def get_registered_agents():
    """
    Simulates MCP discovery.
    In reality, this could be URLs + schemas.
    """
    return {
        "scheduling": scheduling_agent,
        "service_insight": service_insight_agent,
        "knowledge": knowledge_agent,
    }