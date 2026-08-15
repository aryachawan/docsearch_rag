from langgraph.graph import StateGraph,END
from src.state.rag_state import RagState
from src.nodes.node import RagNodes

class GraphBuilder:
    """Builds and manages the langgraph workflow"""

    def __init__(self,retriever,llm):
        """Initialize graph builder
        Args:
           retriever: Document retriever instance
           llm: Language model instance
        """
        self.nodes = RagNodes(
            retriever=retriever,
            llm=llm
        )
        self.graph = None

    def build(self):
        """Build the RAG workflow graph
        Returns:
           Compiled graph instance
        """
        builder = StateGraph(RagState)
        builder.add_node("retriever",self.nodes.retriever_docs)
        builder.add_node("responder",self.nodes.generate_answer)
        builder.set_entry_point("retriever")
        builder.add_edge("retriever","responder")
        builder.add_edge("responder",END)
        self.graph = builder.compile()
        return self.graph

    def run(self, question):
        """Run the RAG workflow for a question."""

        if self.graph is None:
            raise RuntimeError(
                "Graph has not been built. Call build() first."
            )

        initial_state = RagState(
            question=question
        )

        result = self.graph.invoke(initial_state)

        return {
            "answer": result["answer"],
            "retrieved_docs": result["retrieved_docs"]
        }
