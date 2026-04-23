from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Return the current UTC date and time."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a simple arithmetic expression and return the result.

    Args:
        expression: A safe arithmetic expression, e.g. '2 + 2' or '10 / 4'.
    """
    allowed = set("0123456789 +-*/.()%")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


# Register all tools here — import and add to this list as you create more.
tools: list = [
    get_current_time,
    calculator,
]
