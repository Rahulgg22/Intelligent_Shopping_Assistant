
def map_attributes(state):
    constraints = state.get("constraints", {})

    return {
        "mapped_filters": {
            "category": constraints.get("category"),
                "subcategory": constraints.get("subcategory"),
            "activity": constraints.get("activity"),
            "min_price": constraints.get("budget_min"),
            "max_price": constraints.get("budget_max"),
            "preferences": constraints.get("preferences", []),
            "in_stock_only": "in_stock" in constraints.get("hard_constraints", []),
        }
    }
