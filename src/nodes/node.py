from src.state.rag_state import RagState

class RagNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self,retriever,llm):
        """
        Initialize RAG nodes
        Args:
           retriever: Document retriever instance
           llm: Language model instance
        """
        self.retriever = retriever
        self.llm = llm

    def retriever_docs(self,state:RagState)->RagState:
        """Retireves relevant documents node
        Args:
           state: Current RAG state
        Returns:
           Updated RAG state with retrieved documents
        """
        docs = self.retriever.invoke(state.question)
        return RagState(
            question=state.question,
            retrieved_docs=docs
        )

    def generate_answer(self,state: RagState) -> RagState:
        """Generate answer from retrieved documents node
        Args:
           state: Current RAG state with retrieved documents
        Returns:
           Updated RAG state with generated answer
        """
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        prompt = f"""Answer the question based on the context
        Context:
        {context}
        Question: {state.question}
        """
        response = self.llm.invoke(prompt)
        return RagState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )
