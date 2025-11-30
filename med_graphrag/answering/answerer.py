from __future__ import annotations

from textwrap import shorten
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

from med_graphrag.llm.llm_client import get_llm
from med_graphrag.retrieval.combined_retriever import (
    retrieve_with_vector_and_graph,
    RetrievedContext,
)
from med_graphrag.planning.planner_agent import plan_retrieval
from med_graphrag.planning.planner_schemas import RetrievalPlan, RetrievalMode
from med_graphrag.retrieval.combined_retriever import retrieve_with_vector_and_graph, RetrievedContext


def build_answer_chain() -> RunnableSerializable[Dict[str, Any], str]:
    """
    Construis une petite chaîne LLM pour répondre à une question médicale
    en utilisant un contexte RAG + Graph (chunks + entités).
    """

    system_prompt = """You are a medical textbook tutor helping a student understand
diseases and conditions using the book "Essentials of Human Diseases and Conditions".

IMPORTANT RULES:
- Base ALL answers ONLY on the provided context (chunks and entities).
- If the context is missing or incomplete for the question, say so clearly.
- Do NOT invent facts that are not supported by the context.
- Use simple, pedagogical language (like explaining to a motivated student).
- This is NOT medical advice and cannot replace a consultation with a doctor.

When you answer:
- Start with a short direct answer (2–3 sentences).
- Then, if useful, give a structured explanation (e.g., bullet points or sections).
- Optionally, mention which diseases/conditions from the context are most relevant.
- If multiple possibilities exist, explain that clearly."""

    user_prompt = """You will receive:
1) The user's medical question.
2) Top relevant chunks from a medical textbook.
3) Additional context from neighboring chunks.
4) A list of medical entities (diseases, symptoms, treatments, tests).
5) The retrieval mode selected by a planner agent.

Retrieval mode: {plan_mode}

Your task is to answer the question using ONLY this information.

--------------------
QUESTION:
{question}

--------------------
TOP CHUNKS (VECTOR RAG):
{top_chunks_text}

--------------------
GRAPH-EXPANDED CONTEXT (NEIGHBOR CHUNKS):
{graph_context_text}

--------------------
MEDICAL ENTITIES (FROM GRAPH):
{entities_text}

Now, answer the QUESTION following the rules above.
Remember to say that this is not medical advice and that a doctor should be consulted for diagnosis or treatment."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    )

    llm = get_llm(temperature=0.1, max_tokens=700)
    parser = StrOutputParser()

    chain: RunnableSerializable[Dict[str, Any], str] = prompt | llm | parser
    return chain


def _format_top_chunks_for_prompt(top_chunks: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    """
    Formate les top chunks vectoriels pour les mettre dans le prompt.
    On coupe pour rester dans une longueur raisonnable.
    """
    parts: List[str] = []
    for tc in top_chunks:
        header = f"[Rank {tc['rank']} | chunk_id={tc['chunk_id']} | page={tc['page']}]\n"
        text = tc["text"] or ""
        body = shorten(text, width=600, placeholder="...")
        parts.append(header + body)

    joined = "\n\n".join(parts)
    return shorten(joined, width=max_chars, placeholder="\n... [truncated top_chunks] ...")


def _format_graph_context_for_prompt(chunks: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    """
    Formate les chunks voisins obtenus via le graphe.
    """
    parts: List[str] = []
    for ch in chunks:
        header = f"[Chunk {ch['chunk_id']} | page={ch['page_number']}]\n"
        text = ch["text"] or ""
        body = shorten(text, width=500, placeholder="...")
        parts.append(header + body)

    joined = "\n\n".join(parts)
    return shorten(joined, width=max_chars, placeholder="\n... [truncated graph context] ...")


def _format_entities_for_prompt(entities: List[Dict[str, Any]], max_items: int = 30) -> str:
    """
    Regroupe les entités médicales par type afin de les résumer dans le prompt.
    """
    if not entities:
        return "No structured medical entities were found for these chunks."

    # (type, name) unique
    seen = set()
    grouped: Dict[str, List[str]] = {}
    for ent in entities:
        key = (ent.get("entity_type"), ent.get("entity_name"))
        if key in seen:
            continue
        seen.add(key)

        etype = ent.get("entity_type", "UNKNOWN")
        name = ent.get("entity_name", "unknown").strip()
        if not name:
            continue
        grouped.setdefault(etype, []).append(name)

    lines: List[str] = []
    count = 0
    for etype, names in grouped.items():
        if count >= max_items:
            lines.append("... (more entities omitted for brevity) ...")
            break
        preview = ", ".join(sorted(set(names))[:8])
        lines.append(f"{etype}: {preview}")
        count += len(names)

    return "\n".join(lines)


def answer_question_with_graphrag(
    question: str,
    n_results: int = 5,
    neighbor_hops: int = 1,
) -> str:
    """
    Pipeline complet:
      1) retrieve_with_vector_and_graph → contexte enrichi
      2) formater ce contexte
      3) appeler le LLM pour générer la réponse
    """
    print(f"Retrieving context for query: {question}")
    ctx: RetrievedContext = retrieve_with_vector_and_graph(
        query=question,
        n_results=n_results,
        neighbor_hops=neighbor_hops,
    )

    top_chunks_text = _format_top_chunks_for_prompt(ctx.top_chunks)
    graph_context_text = _format_graph_context_for_prompt(ctx.graph_expanded_context)
    entities_text = _format_entities_for_prompt(ctx.medical_entities)

    chain = build_answer_chain()

    llm_input = {
        "question": question,
        "top_chunks_text": top_chunks_text,
        "graph_context_text": graph_context_text,
        "entities_text": entities_text,
    }

    print("🧠 Calling LLM with combined RAG + Graph context...")
    answer = chain.invoke(llm_input)
    return answer

def answer_question_with_agentic_planner(
    question: str,
) -> str:
    """
    Version agentic: 
      1) Planner LLM choisit la stratégie de retrieval (mode, n_results, neighbor_hops, use_graph)
      2) On exécute le retrieval en fonction du plan
      3) On appelle le LLM answerer avec le plan + le contexte
    """
    print(f"🧭 Planning retrieval strategy for question: {question}")
    plan: RetrievalPlan = plan_retrieval(question)
    print(f"   → Mode={plan.mode}, n_results={plan.n_results}, hops={plan.neighbor_hops}, use_graph={plan.use_graph}")
    if plan.reason:
        print(f"   Reason: {plan.reason}")

    # 1) Retrieval (vector + graph)
    # Pour l’instant on utilise toujours retrieve_with_vector_and_graph,
    # mais si use_graph=False on ignorera le contexte graphe dans le prompt.
    ctx: RetrievedContext = retrieve_with_vector_and_graph(
        query=question,
        n_results=plan.n_results,
        neighbor_hops=plan.neighbor_hops if plan.use_graph else 0,
    )

    # 2) Formatter le contexte pour le prompt
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
        "question": question,
        "top_chunks_text": top_chunks_text,
        "graph_context_text": graph_context_text,
        "entities_text": entities_text,
        # on peut aussi passer le mode pour orienter l’explication
        "plan_mode": plan.mode.value,
    }

    print("🧠 Calling LLM with planned RAG + Graph context...")
    answer = chain.invoke(llm_input)
    return answer

