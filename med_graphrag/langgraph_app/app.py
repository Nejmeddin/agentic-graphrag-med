from __future__ import annotations

from langgraph.graph import StateGraph, END

from med_graphrag.langgraph_app.state import QAState
from med_graphrag.langgraph_app.nodes import planner_node, retriever_node, answerer_node


def build_qa_app():
    """
    Construit le graphe LangGraph:
      question -> planner -> retriever -> answerer -> END
    """
    graph = StateGraph(QAState)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answerer", answerer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer", END)

    app = graph.compile()
    return app
