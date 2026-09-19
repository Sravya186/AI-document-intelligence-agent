import streamlit as st
import requests


# -----------------------------------
# Configuration
# -----------------------------------
import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="AI Document Intelligence Agent",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------------
# Session State
# -----------------------------------

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "filename" not in st.session_state:
    st.session_state.filename = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------------
# Title
# -----------------------------------

st.title("🤖 AI Document Intelligence Agent")

st.write(
    "Upload a PDF and interact with it using "
    "RAG, LangGraph, FastAPI, conversational memory, "
    "and document-based decision analysis."
)


# -----------------------------------
# API Health Check
# -----------------------------------

try:

    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    if health_response.status_code == 200:
        st.success("🟢 FastAPI backend connected")

    else:
        st.warning("⚠️ FastAPI backend returned an error")

except requests.exceptions.RequestException:

    st.error(
        "🔴 FastAPI backend is not running. "
        "Start it using: python -m uvicorn api:app --host 127.0.0.1 --port 8000"
    )

    st.stop()


# -----------------------------------
# PDF Upload
# -----------------------------------

st.subheader("📁 Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.button("🔍 Process Document"):

        with st.spinner(
            "Uploading and processing document..."
        ):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                }

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    timeout=120
                )

                if response.status_code == 200:

                    data = response.json()

                    st.session_state.document_uploaded = True
                    st.session_state.filename = uploaded_file.name
                    st.session_state.chat_history = []

                    st.success(
                        "✅ Document processed successfully!"
                    )

                    st.info(
                        f"📄 File: {data['filename']} | "
                        f"Pages: {data['pages']} | "
                        f"Chunks: {data['chunks']}"
                    )

                else:

                    st.error(
                        f"Upload failed: {response.text}"
                    )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"Could not connect to FastAPI: {e}"
                )


# -----------------------------------
# Document Interaction
# -----------------------------------

if st.session_state.document_uploaded:

    st.divider()

    st.subheader(
        f"📄 Current Document: {st.session_state.filename}"
    )

    mode = st.radio(
        "Choose a mode:",
        [
            "🔎 Ask a Question",
            "🧠 Analyze a Situation",
            "💬 Chat with Document"
        ],
        horizontal=True
    )


    # =================================
    # ASK QUESTION
    # =================================

    if mode == "🔎 Ask a Question":

        question = st.text_input(
            "Enter your question:",
            placeholder="Example: What is the annual leave limit?"
        )

        if st.button("🤖 Ask AI"):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "AI agent is processing..."
                ):

                    try:

                        response = requests.post(
                            f"{API_URL}/ask",
                            json={
                                "question": question
                            },
                            timeout=120
                        )

                        if response.status_code == 200:

                            data = response.json()

                            st.subheader(
                                "🤖 AI Response"
                            )

                            st.write(
                                data["answer"]
                            )

                            if data.get("source_pages"):

                                st.subheader(
                                    "📚 Sources"
                                )

                                pages = ", ".join(
                                    [
                                        f"Page {page}"
                                        for page in data[
                                            "source_pages"
                                        ]
                                    ]
                                )

                                st.write(pages)

                        else:

                            st.error(
                                f"API Error: {response.text}"
                            )

                    except requests.exceptions.RequestException as e:

                        st.error(
                            f"FastAPI connection error: {e}"
                        )


    # =================================
    # ANALYZE SITUATION
    # =================================

    elif mode == "🧠 Analyze a Situation":

        situation = st.text_area(
            "Describe your situation:",
            placeholder=(
                "Example: An employee wants to take "
                "7 consecutive days of leave. "
                "What approval is required?"
            ),
            height=120
        )

        if st.button("🧠 Analyze"):

            if not situation.strip():

                st.warning(
                    "Please describe a situation."
                )

            else:

                with st.spinner(
                    "AI agent is analyzing the situation..."
                ):

                    try:

                        response = requests.post(
                            f"{API_URL}/analyze",
                            json={
                                "situation": situation
                            },
                            timeout=120
                        )

                        if response.status_code == 200:

                            data = response.json()

                            st.subheader(
                                "🧠 Analysis"
                            )

                            st.write(
                                data["analysis"]
                            )

                            if data.get("source_pages"):

                                st.subheader(
                                    "📚 Sources"
                                )

                                pages = ", ".join(
                                    [
                                        f"Page {page}"
                                        for page in data[
                                            "source_pages"
                                        ]
                                    ]
                                )

                                st.write(pages)

                        else:

                            st.error(
                                f"API Error: {response.text}"
                            )

                    except requests.exceptions.RequestException as e:

                        st.error(
                            f"FastAPI connection error: {e}"
                        )


    # =================================
    # CHAT
    # =================================

    else:

        st.write(
            "Ask follow-up questions about your document."
        )

        # Display conversation

        for message in st.session_state.chat_history:

            if message["role"] == "user":

                with st.chat_message("user"):

                    st.write(
                        message["content"]
                    )

            else:

                with st.chat_message("assistant"):

                    st.write(
                        message["content"]
                    )


        question = st.chat_input(
            "Ask a follow-up question..."
        )


        if question:

            with st.chat_message("user"):

                st.write(question)


            with st.spinner(
                "AI agent is searching the document..."
            ):

                try:

                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "question": question,
                            "chat_history":
                                st.session_state.chat_history
                        },
                        timeout=120
                    )

                    if response.status_code == 200:

                        data = response.json()

                        answer = data["answer"]

                        with st.chat_message(
                            "assistant"
                        ):

                            st.write(answer)


                        # Save conversation

                        st.session_state.chat_history.append(
                            {
                                "role": "user",
                                "content": question
                            }
                        )

                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        )


                        if data.get("source_pages"):

                            st.caption(
                                "📚 Sources: "
                                + ", ".join(
                                    [
                                        f"Page {page}"
                                        for page in data[
                                            "source_pages"
                                        ]
                                    ]
                                )
                            )

                    else:

                        st.error(
                            f"API Error: {response.text}"
                        )

                except requests.exceptions.RequestException as e:

                    st.error(
                        f"FastAPI connection error: {e}"
                    )


# -----------------------------------
# Clear Chat
# -----------------------------------

if (
    st.session_state.document_uploaded
    and st.session_state.chat_history
):

    st.divider()

    if st.button("🗑️ Clear Conversation"):

        st.session_state.chat_history = []

        st.rerun()


# -----------------------------------
# Footer
# -----------------------------------

st.divider()

st.caption(
    "Built with Python, Streamlit, FastAPI, "
    "LangChain, LangGraph, FAISS, "
    "Sentence Transformers, Groq and RAG."
)