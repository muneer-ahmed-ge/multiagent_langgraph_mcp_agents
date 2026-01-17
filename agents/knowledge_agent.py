def knowledge_agent(state: dict) -> dict:
    """
    Given a product ID, returns documentation.
    """
    product_id = state.get("product_id")

    if product_id == "AC-987":
        return {
            "documentation": "AC-987 Installation & Maintenance Manual"
        }

    return {}