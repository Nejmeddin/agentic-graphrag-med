# api.py (exemple simple avec FastAPI)
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from med_graphrag.answering.answerer import answer_question_with_agentic_planner

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse("static/index.html")


class QARequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    answer: str


@app.post("/qa", response_model=QAResponse)
def qa_endpoint(payload: QARequest):
    # Ici on réutilise ton Agentic GraphRAG complet
    answer = answer_question_with_agentic_planner(payload.question)
    return QAResponse(answer=answer)
