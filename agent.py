from typing import TypedDict, Any

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from rag import (
    ask_question,
    analyze_situation,
    chat_with_history
)

from tools import calculator


# ==================================================
# Agent State
# ==================================================

class AgentState(TypedDict, total=False):

    question: str

    mode: str

    chat_history: list

    vector_database: Any

    llm: Any

    answer: str

    source_pages: list

    tool_result: str


# ==================================================
# MAIN ROUTER NODE
# ==================================================

def router_node(state: AgentState):

    # A LangGraph node MUST return a dictionary.

    return {}


# ==================================================
# MAIN ROUTER DECISION
# ==================================================

def route_request(state: AgentState):

    mode = state.get(
        "mode",
        "question"
    )

    if mode == "question":

        return "question"

    elif mode == "analysis":

        return "analysis"

    elif mode == "chat":

        return "chat"

    return "question"


# ==================================================
# QUESTION NODE
# ==================================================

def question_node(state: AgentState):

    answer, pages = ask_question(
        state["question"],
        state["vector_database"],
        state["llm"]
    )

    return {
        "answer": answer,
        "source_pages": pages
    }


# ==================================================
# ANALYSIS NODE
# ==================================================

def analysis_node(state: AgentState):

    answer, pages = analyze_situation(
        state["question"],
        state["vector_database"],
        state["llm"]
    )

    return {
        "answer": answer,
        "source_pages": pages
    }


# ==================================================
# CHAT NODE
# ==================================================

def chat_node(state: AgentState):

    answer, pages = chat_with_history(
        state["question"],
        state.get(
            "chat_history",
            []
        ),
        state["vector_database"],
        state["llm"]
    )

    return {
        "answer": answer,
        "source_pages": pages
    }


# ==================================================
# TOOL DECISION NODE
# ==================================================

def tool_decision_node(state: AgentState):

    # Node must return a dictionary.

    return {}


# ==================================================
# TOOL DECISION
# ==================================================

def tool_decision(state: AgentState):

    question = state[
        "question"
    ].lower()

    calculation_keywords = [

        "+",
        "-",
        "*",
        "/",
        "%",

        "calculate",
        "calculation",

        "multiply",
        "multiplication",

        "divide",
        "division",

        "addition",
        "add",

        "subtract",
        "subtraction",

        "percentage",
        "percent",

        "total",

        "average"
    ]

    requires_calculation = any(
        keyword in question
        for keyword in calculation_keywords
    )

    if requires_calculation:

        return "calculator"

    return "question"


# ==================================================
# CALCULATOR NODE
# ==================================================

def calculator_node(state: AgentState):

    question = state[
        "question"
    ]

    import re

    # Find mathematical expression
    match = re.search(
        r"[\d\s().+\-*/%]+",
        question
    )

    if not match:

        return {
            "tool_result":
                "Could not identify a mathematical expression."
        }

    expression = match.group().strip()

    # Remove unnecessary spaces
    expression = expression.strip()

    result = calculator.invoke(
        expression
    )

    return {
        "tool_result": result
    }


# ==================================================
# TOOL ANSWER NODE
# ==================================================

def tool_answer_node(state: AgentState):

    question = state[
        "question"
    ]

    tool_result = state.get(
        "tool_result",
        ""
    )

    llm = state[
        "llm"
    ]

    prompt = f"""
You are an AI Document Intelligence Assistant.

The user asked:

{question}

A calculator tool was used.

Calculator result:

{tool_result}

Give the user a clear and concise answer.

Important:
- Do not change the calculator result.
- Do not invent calculations.
- If the calculator result is enough to answer
  the question, directly explain the result.
- If the question requires information from a
  document, explain that the calculator only
  provided the numerical result.

Answer:
"""

    response = llm.invoke(
        prompt
    )

    return {
        "answer": response.content,
        "source_pages": []
    }


# ==================================================
# BUILD LANGGRAPH
# ==================================================

def build_agent():

    graph = StateGraph(
        AgentState
    )

    # ------------------------------------------------
    # Add Nodes
    # ------------------------------------------------

    graph.add_node(
        "router",
        router_node
    )

    graph.add_node(
        "question",
        question_node
    )

    graph.add_node(
        "analysis",
        analysis_node
    )

    graph.add_node(
        "chat",
        chat_node
    )

    graph.add_node(
        "tool_decision",
        tool_decision_node
    )

    graph.add_node(
        "calculator",
        calculator_node
    )

    graph.add_node(
        "tool_answer",
        tool_answer_node
    )

    # ------------------------------------------------
    # START -> ROUTER
    # ------------------------------------------------

    graph.add_edge(
        START,
        "router"
    )

    # ------------------------------------------------
    # ROUTER -> MODE
    # ------------------------------------------------

    graph.add_conditional_edges(
        "router",
        route_request,
        {
            "question": "tool_decision",

            "analysis": "analysis",

            "chat": "chat"
        }
    )

    # ------------------------------------------------
    # TOOL DECISION -> CALCULATOR / RAG
    # ------------------------------------------------

    graph.add_conditional_edges(
        "tool_decision",
        tool_decision,
        {
            "calculator": "calculator",

            "question": "question"
        }
    )

    # ------------------------------------------------
    # CALCULATOR -> TOOL ANSWER
    # ------------------------------------------------

    graph.add_edge(
        "calculator",
        "tool_answer"
    )

    # ------------------------------------------------
    # END NODES
    # ------------------------------------------------

    graph.add_edge(
        "question",
        END
    )

    graph.add_edge(
        "analysis",
        END
    )

    graph.add_edge(
        "chat",
        END
    )

    graph.add_edge(
        "tool_answer",
        END
    )

    # Compile graph

    return graph.compile()


# ==================================================
# CREATE AGENT
# ==================================================

agent = build_agent()


# ==================================================
# RUN AGENT
# ==================================================

def run_agent(
    question,
    mode,
    vector_database,
    llm,
    chat_history=None
):

    if chat_history is None:

        chat_history = []


    # Initial state

    initial_state = {

        "question": question,

        "mode": mode,

        "vector_database":
            vector_database,

        "llm": llm,

        "chat_history":
            chat_history
    }


    # Run LangGraph

    result = agent.invoke(
        initial_state
    )


    # Return answer and sources

    return (
        result.get(
            "answer",
            "No answer was generated."
        ),

        result.get(
            "source_pages",
            []
        )
    )