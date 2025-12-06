from __future__ import annotations

from med_graphrag.langgraph_app.state import QAState
from med_graphrag.planning.planner_agent import plan_retrieval
from med_graphrag.retrieval.combined_retriever import retrieve_with_vector_and_graph
from med_graphrag.answering.answerer import (
    _format_top_chunks_for_prompt,
    _format_graph_context_for_prompt,
    _format_entities_for_prompt,
    build_answer_chain,
)


def planner_node(state: QAState) -> QAState:
    """
    Node LangGraph qui choisit la stratégie de retrieval.
    """
    print(f"[Planner] Question: {state.question}")
    plan = plan_retrieval(state.question)
    print(f"[Planner] Mode={plan.mode}, n_results={plan.n_results}, hops={plan.neighbor_hops}, use_graph={plan.use_graph}")
    state.plan = plan
    return state


def retriever_node(state: QAState) -> QAState:
    """
    Node LangGraph qui lance le retrieval combiné (Chroma + Neo4j)
    selon le plan.
    """
    assert state.plan is not None, "Planner must run before retriever."
    plan = state.plan

    print(f"[Retriever] Running retrieval with n_results={plan.n_results}, hops={plan.neighbor_hops}, use_graph={plan.use_graph}")

    ctx = retrieve_with_vector_and_graph(
        query=state.question,
        n_results=plan.n_results,
        neighbor_hops=plan.neighbor_hops if plan.use_graph else 0,
    )
    state.retrieved_context = ctx
    return state


def answerer_node(state: QAState) -> QAState:
    """
    Node LangGraph qui construit le prompt et appelle le LLM pour répondre.
    """
    assert state.plan is not None, "Planner must run before answerer."
    assert state.retrieved_context is not None, "Retriever must run before answerer."

    plan = state.plan
    ctx = state.retrieved_context

    top_chunks_text = _format_top_chunks_for_prompt(ctx.top_chunks)
    graph_context_text = (
        _format_graph_context_for_prompt(ctx.graph_expanded_context)
        if plan.use_graph
        else "Graph context was not used for this question (SIMPLE_DEFINITION mode)."
    )
    entities_text = (
        _format_entities_for_prompt(ctx.medical_entities)
        if plan.use_graph
        else "No graph entities used (graph expansion disabled by planner)."
    )

    chain = build_answer_chain()

    llm_input = {
        "question": state.question,
        "top_chunks_text": top_chunks_text,
        "graph_context_text": graph_context_text,
        "entities_text": entities_text,
        "plan_mode": plan.mode.value,
    }

    print("[Answerer] Calling LLM...")
    answer = chain.invoke(llm_input)
    state.answer = answer
    return state
