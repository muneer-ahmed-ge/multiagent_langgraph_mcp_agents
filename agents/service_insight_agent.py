def service_insight_agent(state: dict) -> dict:
    """
    Given a work order ID, returns details.
    """
    work_order_id = state.get("work_order_id")

    if work_order_id == "WO-123":
        return {
            "work_order_id": "WO-123",
            "product_id": "AC-987",
            "description": "Repair AC unit on rooftop"
        }

    return {}