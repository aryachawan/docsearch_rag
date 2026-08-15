from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

class RagState(BaseModel):
    """State object for RAG Workflow"""
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""