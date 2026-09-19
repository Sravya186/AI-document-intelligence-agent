import os
import pymupdf

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 1. Extract text from PDF
# --------------------------------------------------

def extract_text_from_pdf(pdf_path):

    doc = pymupdf.open(pdf_path)

    documents = []

    for page_number, page in enumerate(doc):

        text = page.get_text()

        if text.strip():

            documents.append({
                "text": text,
                "page": page_number + 1
            })

    doc.close()

    return documents


# --------------------------------------------------
# 2. Split text into chunks
# --------------------------------------------------

def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = []

    for document in documents:

        split_texts = splitter.split_text(
            document["text"]
        )

        for chunk in split_texts:

            chunks.append({
                "text": chunk,
                "page": document["page"]
            })

    return chunks


# --------------------------------------------------
# 3. Create embeddings
# --------------------------------------------------

def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# --------------------------------------------------
# 4. Create FAISS vector database
# --------------------------------------------------

def create_vector_database(chunks, embeddings):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "page": chunk["page"]
        }
        for chunk in chunks
    ]

    vector_database = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vector_database


# --------------------------------------------------
# 5. Create Groq LLM
# --------------------------------------------------

def create_llm():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please add it to your .env file."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    return llm


# --------------------------------------------------
# 6. Retrieve relevant document information
# --------------------------------------------------

def retrieve_context(
    question,
    vector_database
):

    results = vector_database.similarity_search(
        question,
        k=4
    )

    context_parts = []
    source_pages = []

    for result in results:

        context_parts.append(
            result.page_content
        )

        page = result.metadata.get("page")

        if page:
            source_pages.append(page)

    context = "\n\n".join(
        context_parts
    )

    unique_pages = sorted(
        set(source_pages)
    )

    return context, unique_pages


# --------------------------------------------------
# 7. Normal Question Answering
# --------------------------------------------------

def ask_question(
    question,
    vector_database,
    llm
):

    context, source_pages = retrieve_context(
        question,
        vector_database
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI Document Assistant.

Answer the user's question using ONLY the
information provided in the document context.

If the answer cannot be found in the context,
say:

"I could not find this information in the document."

Do not invent information.

Document Context:
{context}

User Question:
{question}

Answer clearly and concisely.
"""
    )

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)

    return response.content, source_pages


# --------------------------------------------------
# 8. Situation Analysis / Decision Mode
# --------------------------------------------------

def analyze_situation(
    situation,
    vector_database,
    llm
):

    context, source_pages = retrieve_context(
        situation,
        vector_database
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI Document Analysis Assistant.

The user has provided a situation and wants to
know what the document says about that situation.

Use ONLY the information provided in the
document context.

Analyze the situation step by step.

Your response must contain these sections:

DECISION:
Give a concise conclusion based on the document.

REASONING:
Explain how the information in the document
leads to the conclusion.

DOCUMENT EVIDENCE:
Mention the specific rules, facts, limits,
conditions, or requirements from the document
that support your conclusion.

IMPORTANT:
- Do not invent rules.
- Do not use outside knowledge.
- If the document does not contain enough
  information to make a conclusion, say:

"The document does not provide enough
information to determine this."

- Clearly distinguish between what the document
  states and what you infer from applying it.

Document Context:
{context}

User Situation:
{situation}
"""
    )

    messages = prompt.format_messages(
        context=context,
        situation=situation
    )

    response = llm.invoke(messages)

    return response.content, source_pages


# --------------------------------------------------
# 9. Chat with Conversation History
# --------------------------------------------------

def chat_with_history(
    question,
    chat_history,
    vector_database,
    llm
):

    # ----------------------------------------------
    # Retrieve relevant document information
    # ----------------------------------------------

    context, source_pages = retrieve_context(
        question,
        vector_database
    )


    # ----------------------------------------------
    # Convert chat history to text
    # ----------------------------------------------

    history_text = ""

    for message in chat_history:

        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )


    # ----------------------------------------------
    # Prompt
    # ----------------------------------------------

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI Document Assistant.

Your task is to answer the user's current
question using the document context and
conversation history.

The conversation history is mainly used to
understand references such as:

- it
- this
- that
- they
- them
- the previous rule
- the above limit
- this policy

For example:

Previous conversation:
User: What is the maximum leave limit?
Assistant: The maximum leave limit is 20 days.

Current question:
What happens if I exceed it?

You should understand that "it" refers to
the 20-day leave limit.

IMPORTANT RULES:

1. Use ONLY information supported by the
   document context.

2. Use conversation history to understand
   the user's question.

3. Do NOT invent information.

4. If the answer cannot be found in the
   document, say:

"I could not find this information in
the document."

5. Answer clearly and concisely.

Previous Conversation:
{history}

Document Context:
{context}

Current User Question:
{question}

Answer:
"""
    )


    # ----------------------------------------------
    # Create messages
    # ----------------------------------------------

    messages = prompt.format_messages(
        history=history_text,
        context=context,
        question=question
    )


    # ----------------------------------------------
    # Generate response
    # ----------------------------------------------

    response = llm.invoke(messages)


    return response.content, source_pages