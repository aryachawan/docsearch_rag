from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStore:
    """Manages vectorstore application"""
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.retriever = None

    def create_retriever(self,documents: List[Document]):
        """Creates a vector store from documents
        Args:
           documents: List of documents to embed
        """
        self.vectorstore = FAISS.from_documents(documents,self.embedding)
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self):
        """Gets the retriever instance
        Returns:
           Retriever Instance
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first")
        return self.retriever

    def retrieve(self,query:str,k:int = 4)->List[Document]:
        """Retrieve relevant documents from a query
        Args:
            query: Search query
            k: Number of documents to retrieve
        Returns:
            List of relevant documents
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever.invoke(query)