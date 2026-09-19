from langchain_core.tools import tool


# ==================================================
# Calculator Tool
# ==================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate a basic mathematical expression.

    Example:
    12500 * 5
    50000 / 4
    20000 * 0.15
    """

    try:

        # Only allow basic mathematical characters
        allowed_characters = (
            "0123456789+-*/().% "
        )

        if not all(
            character in allowed_characters
            for character in expression
        ):
            return "Invalid mathematical expression."

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            {}
        )

        return str(result)

    except Exception:

        return "Unable to calculate the expression."