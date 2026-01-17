from orchestrator.graph import build_graph


class OrchestratorAgent:
    def __init__(self):
        self.graph = build_graph()

    def run(self, user_goal: str):
        """
        Entry point for external users.
        """
        initial_state = {
            "goal": user_goal
        }

        final_state = self.graph.invoke(initial_state)
        return final_state