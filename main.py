from orchestrator.orchestrator import OrchestratorAgent

if __name__ == "__main__":
    orchestrator = OrchestratorAgent()

    goal = (
        "Tell me today's scheduled work order, "
        "give me details about it, "
        "and provide any relevant documentation."
    )

    result = orchestrator.run(goal)

    print("\n=== FINAL RESULT ===")
    for key, value in result.items():
        print(f"{key}: {value}")