import streamlit as st
from pathlib import Path
import sys
import time
import traceback

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


# --------------------------------------------------
# Streamlit Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Search",
    page_icon="🔍",
    layout="centered"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>
.stButton > button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Session State
# --------------------------------------------------

def init_session_state():
    """Initialize Streamlit session state variables."""

    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if "history" not in st.session_state:
        st.session_state.history = []


# --------------------------------------------------
# Initialize RAG System
# --------------------------------------------------

@st.cache_resource
def initialize_rag():
    """Initialize and return the complete RAG system."""

    try:

        # -------------------------------
        # Step 1: Initialize LLM
        # -------------------------------

        print("1. Initializing LLM...")

        llm = Config.get_llm()

        print("2. LLM initialized successfully")

        # -------------------------------
        # Step 2: Initialize Document Processor
        # -------------------------------

        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

        print("3. Document processor initialized")

        # -------------------------------
        # Step 3: Initialize Vector Store
        # -------------------------------

        vector_store = VectorStore()

        print("4. Vector store initialized")

        # -------------------------------
        # Step 4: Load Documents
        # -------------------------------

        urls = Config.DEFAULT_URLS

        documents = doc_processor.process_url(urls)

        print(
            f"5. Documents processed successfully: "
            f"{len(documents)} chunks"
        )

        # -------------------------------
        # Step 5: Create Retriever
        # -------------------------------

        vector_store.create_retriever(documents)

        print("6. Retriever created successfully")

        # -------------------------------
        # Step 6: Initialize Graph Builder
        # -------------------------------

        graphbuilder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )

        print("7. GraphBuilder initialized")

        # -------------------------------
        # Step 7: Build Graph
        # -------------------------------

        graphbuilder.build()

        print("8. Graph built successfully")

        return graphbuilder, len(documents)

    except Exception as e:

        # Print complete traceback in terminal
        print("\n" + "=" * 60)
        print("RAG INITIALIZATION ERROR")
        print("=" * 60)

        traceback.print_exc()

        print("=" * 60 + "\n")

        # Also show complete traceback in Streamlit
        st.error(f"Failed to initialize: {str(e)}")

        with st.expander("Show full error details"):
            st.code(traceback.format_exc())

        return None, 0


# --------------------------------------------------
# Main Application
# --------------------------------------------------

def main():

    init_session_state()

    # -------------------------------
    # Header
    # -------------------------------

    st.title("🔍 RAG Document Search")

    st.markdown(
        "Ask questions about the loaded documents."
    )

    # -------------------------------
    # Initialize RAG
    # -------------------------------

    if not st.session_state.initialized:

        with st.spinner("Loading RAG system..."):

            rag_system, num_chunks = initialize_rag()

            if rag_system:

                st.session_state.rag_system = rag_system

                st.session_state.initialized = True

                st.success(
                    f"System ready! "
                    f"({num_chunks} document chunks loaded)"
                )

    # -------------------------------
    # Search Section
    # -------------------------------

    st.markdown("---")

    with st.form("search_form"):

        question = st.text_input(
            "Enter your question",
            placeholder="What would you like to know?"
        )

        submit = st.form_submit_button(
            "Search"
        )

    # -------------------------------
    # Handle Search
    # -------------------------------

    if submit:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        elif not st.session_state.rag_system:

            st.error(
                "RAG system is not initialized."
            )

        else:

            with st.spinner("Searching..."):

                try:

                    start_time = time.time()

                    result = (
                        st.session_state
                        .rag_system
                        .run(question)
                    )

                    elapsed_time = (
                        time.time() - start_time
                    )

                    # -------------------------------
                    # Store Search History
                    # -------------------------------

                    st.session_state.history.append({
                        "question": question,
                        "answer": result.get(
                            "answer",
                            "No answer returned."
                        ),
                        "time": elapsed_time
                    })

                    # -------------------------------
                    # Display Answer
                    # -------------------------------

                    st.markdown("### Answer")

                    st.success(
                        result.get(
                            "answer",
                            "No answer returned."
                        )
                    )

                    # -------------------------------
                    # Display Sources
                    # -------------------------------

                    retrieved_docs = result.get(
                        "retrieved_docs",
                        []
                    )

                    if retrieved_docs:

                        with st.expander(
                            "📄 Source Documents"
                        ):

                            for i, doc in enumerate(
                                retrieved_docs,
                                start=1
                            ):

                                content = (
                                    doc.page_content
                                    if hasattr(
                                        doc,
                                        "page_content"
                                    )
                                    else str(doc)
                                )

                                st.text_area(
                                    f"Document {i}",
                                    content[:500] + (
                                        "..."
                                        if len(content) > 500
                                        else ""
                                    ),
                                    height=120,
                                    disabled=True
                                )

                    else:

                        st.info(
                            "No source documents returned."
                        )

                    # -------------------------------
                    # Response Time
                    # -------------------------------

                    st.caption(
                        f"Response time: "
                        f"{elapsed_time:.2f} seconds"
                    )

                except Exception as e:

                    st.error(
                        f"Search failed: {str(e)}"
                    )

                    with st.expander(
                        "Show full search error"
                    ):

                        st.code(
                            traceback.format_exc()
                        )

    # --------------------------------------------------
    # Recent Searches
    # --------------------------------------------------

    if st.session_state.history:

        st.markdown("---")

        st.markdown("### 🕘 Recent Searches")

        # Show last 3 searches
        for item in reversed(
            st.session_state.history[-3:]
        ):

            with st.container():

                st.markdown(
                    f"**Q:** {item['question']}"
                )

                st.markdown(
                    f"**A:** {item['answer']}"
                )

                st.caption(
                    f"Response time: "
                    f"{item['time']:.2f}s"
                )

                st.markdown("")


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":
    main()