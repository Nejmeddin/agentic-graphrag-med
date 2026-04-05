# Agentic GraphRAG Med

A medical question-answering system that combines **Knowledge Graph reasoning** (Neo4j) with **vector similarity search** (ChromaDB) and an **agentic LLM planner** (Groq / LLaMA 3.3-70B) to answer questions grounded in the book *"Essentials of Human Diseases and Conditions"*.

A **FastAPI** backend exposes a REST endpoint; a built-in **chat UI** lets you ask questions in the browser; and an **n8n workflow** lets you wire the API into any automation pipeline.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration (.env)](#configuration-env)
7. [Data Pipeline (one-time setup)](#data-pipeline-one-time-setup)
8. [Running the API](#running-the-api)
9. [Chat UI](#chat-ui)
10. [n8n Integration](#n8n-integration)
11. [API Reference](#api-reference)
12. [Module Reference](#module-reference)

---

## Architecture Overview

```
User question
     │
     ▼
┌────────────────────────────────────────┐
│          Planner Agent (LLM)           │  ← Groq LLaMA 3.3-70B
│  Decides: mode, n_results, hops,       │
│  use_graph  (JSON output)              │
└─────────────────┬──────────────────────┘
                  │
     ┌────────────▼────────────┐
     │   Combined Retriever    │
     │  ┌──────────────────┐   │
     │  │  ChromaDB        │   │  ← vector similarity search
     │  │  (top-k chunks)  │   │
     │  └────────┬─────────┘   │
     │           │             │
     │  ┌────────▼─────────┐   │
     │  │  Neo4j Graph     │   │  ← neighbor chunks + medical entities
     │  │  (NEXT_CHUNK /   │   │
     │  │  MedicalEntity)  │   │
     │  └──────────────────┘   │
     └────────────┬────────────┘
                  │
     ┌────────────▼────────────┐
     │    Answerer (LLM)       │  ← Groq LLaMA 3.3-70B
     │  Context-grounded       │
     │  medical tutor prompt   │
     └────────────┬────────────┘
                  │
             Final Answer
```

The pipeline is also orchestrated as a **LangGraph** state machine:
`planner node → retriever node → answerer node → END`

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM inference | [Groq API](https://console.groq.com/) – `llama-3.3-70b-versatile` |
| Vector store | [ChromaDB](https://www.trychroma.com/) (local persistent) |
| Knowledge graph | [Neo4j](https://neo4j.com/) (Docker) |
| LLM orchestration | [LangChain](https://www.langchain.com/) + [LangGraph](https://langchain-ai.github.io/langgraph/) |
| API server | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Chat UI | Vanilla HTML/CSS/JS (served as a static file) |
| Automation | [n8n](https://n8n.io/) (Docker) |
| Config | Pydantic Settings + `python-dotenv` |

---

## Project Structure

```
agentic-graphrag-med/
├── api.py                        # FastAPI app (REST API + serve chat UI)
├── main.py                       # Entry-point placeholder
├── requirements.txt
├── n8n_workflow.json             # Import-ready n8n workflow
├── static/
│   └── index.html                # Chat UI (served at GET /)
├── data/
│   ├── source/
│   │   ├── .env                  # Secrets (NEO4J, GROQ keys)
│   │   └── <your PDF here>
│   ├── processed/
│   │   ├── essentials_chunks.jsonl
│   │   └── entities_essentials.jsonl
│   └── chroma/                   # ChromaDB persistent storage
└── med_graphrag/
    ├── config/settings.py        # Pydantic settings (reads .env)
    ├── data_pipeline/            # PDF loader + chunker
    ├── agents/                   # LLM entity extraction agent
    ├── graph/                    # Neo4j client, schema, builder, queries
    ├── vectorstore/              # ChromaDB helpers
    ├── retrieval/                # Combined vector + graph retriever
    ├── planning/                 # Planner agent (query strategy)
    ├── answering/                # Answerer chain (final LLM call)
    ├── langgraph_app/            # LangGraph pipeline (planner→retriever→answerer)
    ├── llm/                      # Groq LLM factory
    ├── vision/                   # Image classifier (optional)
    └── cli/                      # One-time build scripts
        ├── prepare_pdf.py        # Step 1 – PDF → chunks JSONL
        ├── extract_medical_entities.py  # Step 2 – LLM entity extraction
        ├── build_mkg_basic.py    # Step 3a – Build base Neo4j graph
        ├── build_mkg_entities.py # Step 3b – Ingest entities into Neo4j
        ├── build_vectorstore.py  # Step 4 – Load chunks into ChromaDB
        └── qa_*.py               # Demo / test scripts
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Docker Desktop | latest |
| Neo4j (via Docker) | 5.x |
| Groq API key | free at [console.groq.com](https://console.groq.com/) |

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd agentic-graphrag-med
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
pip install fastapi uvicorn[standard]
```

### 4. Start Neo4j with Docker

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v C:\Users\User\Desktop\agentic-graphrag-med\my-neo4j-data:/data \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5
```

> Neo4j Browser will be available at **http://localhost:7474** after startup.

---

## Configuration (.env)

Create the file `data/source/.env` with the following variables:

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Groq
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
GROQ_MODEL_NAME=llama-3.3-70b-versatile
```

The settings are loaded by `med_graphrag/config/settings.py` via Pydantic Settings.

---

## Data Pipeline (one-time setup)

Run these scripts **once** to build the knowledge base. Each step depends on the previous one.

### Step 1 – Process the PDF into chunks

Place your PDF (e.g. `essentials-of-human-diseases-and-conditions.pdf`) in `data/source/`, then run:

```bash
python -m med_graphrag.cli.prepare_pdf
```

Output: `data/processed/essentials_chunks.jsonl`
Each record contains: `chunk_id`, `text`, `page`, `source`, `chunk_index`.

---

### Step 2 – Extract medical entities with the LLM

```bash
python -m med_graphrag.cli.extract_medical_entities
```

This sends each chunk to the Groq LLM and extracts structured entities (diseases, symptoms, treatments, tests, anatomy).

Output: `data/processed/entities_essentials.jsonl`

> **Note:** This step consumes Groq API tokens. By default it processes the first 1 000 chunks. Edit `MAX_CHUNKS` in `cli/extract_medical_entities.py` to adjust.

---

### Step 3a – Build the base Knowledge Graph in Neo4j

```bash
python -m med_graphrag.cli.build_mkg_basic
```

Creates the graph schema and ingests:
- `Document` nodes
- `Page` nodes
- `Chunk` nodes linked by `NEXT_CHUNK` relationships

---

### Step 3b – Ingest medical entities into Neo4j

```bash
python -m med_graphrag.cli.build_mkg_entities
```

Creates `MedicalEntity` nodes and links them to `Chunk` nodes with `MENTIONS` relationships.

---

### Step 4 – Build the ChromaDB vector store

```bash
python -m med_graphrag.cli.build_vectorstore
```

Embeds all chunks using `sentence-transformers` and stores them in `data/chroma/` for fast similarity search.

---

## Running the API

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The server starts at **http://localhost:8000**.

- `GET /` → Chat UI
- `POST /qa` → Question-answering endpoint

For production (no auto-reload):

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## Chat UI

Open **http://localhost:8000** in your browser.

Features:
- Dark-themed conversational interface
- Full conversation history in the same session
- Typing animation while the agent processes
- **Enter** to send, **Shift+Enter** for a new line
- Auto-growing text area
- Ask as many follow-up questions as you want

---

## n8n Integration

An import-ready workflow is provided at `n8n_workflow.json`.

### Start n8n with Docker

```bash
docker run -it --name n8n -p 5678:5678 \
  -v C:\Users\User\Desktop\Docker\.n8n:/home/node/.n8n \
  n8nio/n8n
```

n8n will be available at **http://localhost:5678**.

### Import the workflow

1. Open n8n → **Workflows** → **Import from file**
2. Select `n8n_workflow.json`
3. Click **Activate**

### Workflow structure

```
POST /webhook/medical-qa
        │
        ▼
HTTP Request → POST http://host.docker.internal:8000/qa
        │         body: { "question": "{{ $json.body.question }}" }
        ▼
Respond to Webhook → { "answer": "..." }
```

> `host.docker.internal` resolves to your Windows host machine from inside Docker, so n8n can reach the FastAPI server running on your machine.

### Test the webhook

```bash
curl -X POST http://localhost:5678/webhook/medical-qa \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What are the symptoms of cystic fibrosis?\"}"
```

Response:

```json
{ "answer": "Cystic fibrosis is a genetic disorder..." }
```

---

## API Reference

### `POST /qa`

Ask a medical question. Returns a grounded answer from the knowledge base.

**Request body**

```json
{
  "question": "What is hypertension and how does it relate to kidney disease?"
}
```

**Response**

```json
{
  "answer": "Hypertension (high blood pressure) is a condition where..."
}
```

**Example with curl**

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Compare type 1 and type 2 diabetes\"}"
```

---

## Module Reference

### `med_graphrag.planning.planner_agent`
LLM-based agent that reads the user question and outputs a `RetrievalPlan` JSON deciding:
- `mode`: `SIMPLE_DEFINITION` | `GRAPH_RELATION` | `COMPARE_DISEASES` | `UNSURE`
- `n_results`: how many vector hits to fetch (1–15)
- `neighbor_hops`: how many hops to traverse in the graph (0–3)
- `use_graph`: whether to expand context via Neo4j

### `med_graphrag.retrieval.combined_retriever`
1. Queries ChromaDB for the top-k most similar chunks.
2. For each chunk, queries Neo4j for neighboring chunks (`NEXT_CHUNK`) and linked `MedicalEntity` nodes.
3. Returns a `RetrievedContext` dataclass with `top_chunks`, `graph_expanded_context`, and `medical_entities`.

### `med_graphrag.answering.answerer`
Assembles all retrieved context into a structured prompt and calls the Groq LLM to produce a pedagogical, grounded answer. Enforces a "no hallucination" rule via the system prompt.

### `med_graphrag.langgraph_app`
Wraps the full pipeline (planner → retriever → answerer) as a compiled **LangGraph** state machine for clean node-based orchestration.

### `med_graphrag.llm.llm_client`
Factory function `get_llm()` that returns a configured `ChatGroq` instance reading credentials from the `.env` file.

### `med_graphrag.graph`
- `neo4j_client.py` – singleton Neo4j driver
- `mkg_schema.py` – creates uniqueness constraints
- `mkg_builder.py` – ingests Documents, Pages, Chunks
- `mkg_entities_ingest.py` – ingests MedicalEntity nodes
- `mkg_queries.py` – Cypher queries (neighbor expansion, entity lookup)

### `med_graphrag.vectorstore`
- `store_chroma.py` – ChromaDB client and collection helpers
- `chunks_loader.py` – loads and prepares chunks JSONL for Chroma ingestion
