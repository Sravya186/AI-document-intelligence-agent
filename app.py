import streamlit as st

from rag import (
    extract_text_from_pdf,
    create_chunks,
    create_embeddings,
    create_vector_database,
    create_llm
)

from agent import run_agent


st.set_page_config(
    page_title="AI Document Intelligence Agent",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# Session State
# -----------------------------

if "vector_database" not in st.session_state:
    st.session_state.vector_database = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# -----------------------------
# Title
# -----------------------------

st.title("🤖 AI Document Intelligence Agent")

st.write(
    "Upload a PDF and ask questions, analyze situations, "
    "or chat with the document."
)


# -----------------------------
# Groq API Key
# -----------------------------

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = None


if groq_api_key:
    import os
    os.environ["GROQ_API_KEY"] = groq_api_key


# -----------------------------
# PDF Upload
# -----------------------------

st.sidebar.header("📄 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.session_state.document_name != uploaded_file.name:

        with st.spinner(
            "Processing document... This may take a little while."
        ):

            # Save temporary PDF
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_pdf_path = temp_file.name

            try:

                # Extract text
                documents = extract_text_from_pdf(
                    temp_pdf_path
                )

                if not documents:
                    st.error(
                        "No readable text found in this PDF."
                    )
                    st.stop()

                # Create chunks
                chunks = create_chunks(documents)

                # Create embeddings
                embeddings = create_embeddings()

                # Create FAISS database
                vector_database = create_vector_database(
                    chunks,
                    embeddings
                )

                # Create LLM
                llm = create_llm()

                # Store in session
                st.session_state.vector_database = vector_database
                st.session_state.llm = llm
                st.session_state.document_name = uploaded_file.name
                st.session_state.chat_history = []

            finally:

                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)


        st.sidebar.success(
            f"Loaded: {uploaded_file.name}"
        )


# -----------------------------
# Check Document
# -----------------------------

if st.session_state.vector_database is None:

    st.info(
        "👈 Upload a PDF from the sidebar to get started."
    )

    st.stop()


# -----------------------------
# Main Tabs
# -----------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "💬 Ask Questions",
        "🔎 Situation Analysis",
        "🗨️ Chat"
    ]
)


# =========================================================
# TAB 1 — ASK QUESTIONS
# =========================================================

with tab1:

    st.header("Ask Questions")

    question = st.text_input(
        "Enter your question:",
        placeholder="What is the annual leave limit?"
    )

    if st.button(
        "Ask Question",
        key="ask_button"
    ):

        if question.strip():

            with st.spinner("Thinking..."):

                answer, pages = run_agent(
                    question=question,
                    mode="question",
                    vector_database=st.session_state.vector_database,
                    llm=st.session_state.llm,
                    chat_history=[]
                )

            st.subheader("Answer")

            st.write(answer)

            if pages:

                st.caption(
                    "Source pages: "
                    + ", ".join(
                        str(page)
                        for page in pages
                    )
                )


# =========================================================
# TAB 2 — SITUATION ANALYSIS
# =========================================================

with tab2:

    st.header("Situation Analysis")

    situation = st.text_area(
        "Describe a situation:",
        placeholder=(
            "An employee wants to take 7 consecutive "
            "days of annual leave. What should happen?"
        )
    )

    if st.button(
        "Analyze Situation",
        key="analysis_button"
    ):

        if situation.strip():

            with st.spinner(
                "Analyzing document policy..."
            ):

                answer, pages = run_agent(
                    question=situation,
                    mode="analysis",
                    vector_database=st.session_state.vector_database,
                    llm=st.session_state.llm,
                    chat_history=[]
                )

            st.subheader("Analysis")

            st.write(answer)

            if pages:

                st.caption(
                    "Source pages: "
                    + ", ".join(
                        str(page)
                        for page in pages
                    )
                )


# =========================================================
# TAB 3 — CHAT
# =========================================================

with tab3:

    st.header("Chat with Document")

    # Display previous messages

    for message in st.session_state.chat_history:

        if message["role"] == "user":

            with st.chat_message("user"):
                st.write(message["content"])

        else:

            with st.chat_message("assistant"):
                st.write(message["content"])


    user_message = st.chat_input(
        "Ask something about the document..."
    )


    if user_message:

        # Display user message

        with st.chat_message("user"):
            st.write(user_message)

        # Run agent

        with st.spinner("Thinking..."):

            answer, pages = run_agent(
                question=user_message,
                mode="chat",
                vector_database=st.session_state.vector_database,
                llm=st.session_state.llm,
                chat_history=st.session_state.chat_history
            )


        # Display answer

        with st.chat_message("assistant"):
            st.write(answer)


        # Save conversation

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        if pages:

            st.caption(
                "Source pages: "
                + ", ".join(
                    str(page)
                    for page in pages
                )
            )