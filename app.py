"""
RizooSphere Assistant
LangGraph + Corrective RAG + Gemini + Streamlit

Architecture (unchanged from original design):

    START -> Router -> {Direct -> END, Retrieve, Decline -> END}
    Retrieve -> Grader -> {Generate -> END, Rewrite}
    Rewrite -> should_retry -> {Retrieve, END}

    Streamlit -> Upload PDF -> Extract Text -> Chunking -> Embeddings
              -> Vector Store -> Retriever -> Graph -> Router
"""

import json
import logging
import os
import re
from typing import List, Optional, TypedDict

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader

# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rizoosphere_assistant")

# =========================================================
# Constants
# =========================================================

VALID_ROUTES = ["direct", "retrieve", "decline"]
MAX_RETRIES = 2

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "models/embedding-001"

# =========================================================
# API Key / LLM Setup
# =========================================================


def _resolve_google_api_key() -> str:
    """Resolve the Google API key from env vars or Streamlit secrets
    without clobbering a key that may already be configured."""
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if api_key:
        return api_key
    try:
        api_key = str(st.secrets.get("GOOGLE_API_KEY", "AQ.Ab8RN6Is3b1g0ZlKaAs3FfM_6IHiXuoevpl22k17Fd2CSQ2HEQ")).strip()
    except Exception:
        api_key = ""
    return api_key


GOOGLE_API_KEY = _resolve_google_api_key()
if not GOOGLE_API_KEY:
    st.error(
        "GOOGLE_API_KEY is not set. Add it to your environment variables "
        "or to Streamlit secrets before using the assistant."
    )
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


@st.cache_resource
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")


llm = get_llm()

# =========================================================
# Graph State
# =========================================================


class GraphState(TypedDict):
    original_question: str
    question: str
    route: str
    confidence: float
    context: str
    answer: str
    grounded: bool
    retries: int
    documents: List[Document]
    sources: List[str]
    relevance: str
    response_type: str


def build_initial_state(question: str) -> GraphState:
    """Seed a complete GraphState so downstream nodes never hit
    missing-key errors on the first invocation."""
    return {
        "original_question": question,
        "question": question,
        "route": "",
        "confidence": 0.0,
        "context": "",
        "answer": "",
        "grounded": False,
        "retries": 0,
        "documents": [],
        "sources": [],
        "relevance": "",
        "response_type": "",
    }


# =========================================================
# Prompts
# =========================================================
# NOTE: all literal JSON braces below are escaped as {{ }} because
# these templates are rendered with PromptTemplate.format() (str.format
# under the hood). Unescaped braces were previously interpreted as
# format placeholders and crashed every node.

# =========================================================
# Router Prompt
# =========================================================

ROUTER_PROMPT_V1 = """

You are a routing classifier.


Your ONLY task is to classify the user's question.


Never answer the question.

Never explain your decision.


Choose ONLY one label and Confidence:

1- direct
2- retrieve
3- decline


direct:

- greetings
- general questions
- what can you do
- simple definitions


retrieve:

- questions that require information
from the provided documents.


decline:

- out of scope questions.
- harmful requests.
- unrelated topics.

Confidence represents how certain you are about the routing decision.

Question:

{question}


Return ONLY valid JSON.
{{
"route": "direct | retrieve | decline",
"confidence": 0.0-1.0
}}

"""

# =================================================
# Direct Prompt
# =================================================

DIRECT_PROMPT = """
You are a helpful assistant.

Answer briefly and clearly.

Question:

{question}

Return ONLY a valid JSON:

{{
  "answer":"...",
  "grounded":false
}}
"""

# =================================================
# Generation Prompt
# =================================================

GENERATION_PROMPT = """
You are a RAG assistant.

Use ONLY the provided context.

If the answer is not present in the context:

"The information is not available in the provided documents."

Never invent information.

Context:

{context}

Question:

{question}

Return ONLY valid JSON:

{{
  "answer":"...",
  "grounded":true,
  "sources":[]
}}
"""

# =================================================
# Grader Prompt
# =================================================
# NOTE: original text told the model to return true/false while the
# JSON schema asked for "relevance": "generate"/"rewrite" - the two
# were contradictory. Wording aligned to the schema below; the grading
# purpose (decide if retrieved context is sufficient) is unchanged.

GRADER_PROMPT = """
You are a retrieval evaluator.

Your task is to determine whether the retrieved context contains enough relevant information to answer the user's question.

Rules:
- Read both the question and the retrieved context carefully.
- If the context contains sufficient information, set "relevance" to "generate".
- If the context is missing the required information or is unrelated, set "relevance" to "rewrite".
- Do NOT answer the user's question.
- Do NOT explain your reasoning.

Question:
{question}

Retrieved Context:
{context}

Return ONLY valid JSON:

{{
  "relevance": "generate"
}}

OR

{{
  "relevance": "rewrite"
}}
"""

# =================================================
# Rewrite Prompt
# =================================================

REWRITE_PROMPT = """

You are a query rewriting assistant.

Your task is to rewrite the user's
question to improve document retrieval.

Rules:

- Keep the original meaning.
- Make the question more specific.
- Do NOT change the user's intent.
- Do NOT answer the question.
- Return ONLY the rewritten question.


Original Question:

{question}


Rewritten Question:

"""

# =================================================
# Decline Message
# =================================================

DECLINE_MESSAGE = """
This question is outside the scope of
the available documents.

Please ask a question related to the
provided documentation.
"""

# =========================================================
# JSON Parser
# =========================================================

_CODE_FENCE_RE = re.compile(r"^```(?:json)?|```$", re.IGNORECASE | re.MULTILINE)


def parse_json(response) -> dict:
    """Parse a JSON object out of an LLM response, tolerating markdown
    code fences and surrounding whitespace."""
    text = getattr(response, "content", response)
    if not isinstance(text, str):
        text = str(text)

    text = _CODE_FENCE_RE.sub("", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON from LLM response. Raw text: %r", text)
        raise


# =========================================================
# Router Node
# =========================================================


def router_node(state: GraphState) -> GraphState:
    question = state["question"]
    prompt = PromptTemplate(template=ROUTER_PROMPT_V1, input_variables=["question"])
    final_prompt = prompt.format(question=question)

    route = "decline"
    confidence = 0.0

    try:
        response = llm.invoke(final_prompt)
        result = parse_json(response)
        route = str(result.get("route", "")).strip().lower()
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        logger.exception("Router node failed to classify the question")
        route = "decline"
        confidence = 0.0

    if route not in VALID_ROUTES:
        route = "decline"
        confidence = 0.0

    state["route"] = route
    state["confidence"] = confidence
    return state


# =========================================================
# Direct Node
# =========================================================


def direct_node(state: GraphState) -> GraphState:
    question = state["question"]
    prompt = PromptTemplate(template=DIRECT_PROMPT, input_variables=["question"])
    final_prompt = prompt.format(question=question)

    try:
        response = llm.invoke(final_prompt)
        output = parse_json(response)
        state["answer"] = output.get("answer", "No answer generated.")
        state["grounded"] = bool(output.get("grounded", False))
    except Exception:
        logger.exception("Direct node failed to generate an answer")
        state["answer"] = "Sorry, I couldn't generate an answer right now."
        state["grounded"] = False

    state["sources"] = []
    state["response_type"] = "GENERAL KNOWLEDGE"
    return state


# =========================================================
# Retrieve Node
# =========================================================


def retrieve_node(state: GraphState) -> GraphState:
    question = state["question"]
    retriever = st.session_state.get("retriever")

    if retriever is None:
        logger.warning("Retrieve node invoked but no retriever is available yet")
        state["documents"] = []
        state["retries"] = state.get("retries", 0) + 1
        return state

    try:
        documents = retriever.invoke(question)
    except Exception:
        logger.exception("Document retrieval failed")
        documents = []

    state["documents"] = documents
    if len(documents) == 0:
        state["retries"] = state.get("retries", 0) + 1

    return state


# =========================================================
# Documents To Context
# =========================================================


def documents_to_text(documents: List[Document]) -> str:
    return "\n".join(doc.page_content for doc in documents)


# =========================================================
# Grader Node
# =========================================================


def grader_node(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state["documents"]
    context = documents_to_text(documents)

    prompt = PromptTemplate(template=GRADER_PROMPT, input_variables=["question", "context"])
    final_prompt = prompt.format(question=question, context=context)

    try:
        response = llm.invoke(final_prompt)
        output = parse_json(response)
        relevance = str(output.get("relevance", "rewrite")).strip().lower()
        state["relevance"] = relevance if relevance in ("generate", "rewrite") else "rewrite"
    except Exception:
        logger.exception("Grader node failed to evaluate retrieved context")
        state["relevance"] = "rewrite"

    return state


# =========================================================
# Rewrite Node
# =========================================================


def rewrite_node(state: GraphState) -> GraphState:
    question = state["question"]
    prompt = PromptTemplate(template=REWRITE_PROMPT, input_variables=["question"])
    final_prompt = prompt.format(question=question)

    try:
        response = llm.invoke(final_prompt)
        rewritten_question = response.content.strip()
        state["question"] = rewritten_question or question
    except Exception:
        logger.exception("Rewrite node failed to rewrite the question")
        # Keep the existing question if rewriting fails.

    state["retries"] = state.get("retries", 0) + 1
    return state


# =========================================================
# Generate Node
# =========================================================


def generate_node(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state["documents"]
    context = documents_to_text(documents)

    prompt = PromptTemplate(template=GENERATION_PROMPT, input_variables=["question", "context"])
    final_prompt = prompt.format(question=question, context=context)

    try:
        response = llm.invoke(final_prompt)
        output = parse_json(response)
        state["answer"] = output.get("answer", "No answer generated.")
        state["grounded"] = bool(output.get("grounded", False))
        state["sources"] = output.get("sources", [])
    except Exception:
        logger.exception("Generate node failed to produce an answer")
        state["answer"] = "Sorry, I couldn't generate an answer right now."
        state["grounded"] = False
        state["sources"] = []

    state["context"] = context
    state["response_type"] = "RAG (DOCUMENT-GROUNDED)"
    return state


# =========================================================
# Decline Node
# =========================================================


def decline_node(state: GraphState) -> GraphState:
    state["answer"] = DECLINE_MESSAGE
    state["grounded"] = False
    state["sources"] = []
    state["response_type"] = "DECLINED"
    return state


# =========================================================
# Build LangGraph
# =========================================================


@st.cache_resource
def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("router", router_node)
    workflow.add_node("direct", direct_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("decline", decline_node)

    workflow.add_edge(START, "router")  # Starting

    def route_question(state: GraphState) -> str:
        return state["route"]

    workflow.add_conditional_edges(
        "router",
        route_question,
        {
            "direct": "direct",
            "retrieve": "retrieve",
            "decline": "decline",
        },
    )

    workflow.add_edge("direct", END)  # End point
    workflow.add_edge("decline", END)  # End point

    workflow.add_edge("retrieve", "grader")

    # Grader Decision
    def grade_documents(state: GraphState) -> str:
        return state["relevance"]

    workflow.add_conditional_edges(
        "grader",
        grade_documents,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )

    workflow.add_edge("generate", END)

    def should_retry(state: GraphState) -> str:
        if state["retries"] >= MAX_RETRIES:
            return "end"
        return "retrieve"

    workflow.add_conditional_edges(
        "rewrite",
        should_retry,
        {
            "retrieve": "retrieve",
            "end": END,
        },
    )

    return workflow.compile()


graph = build_graph()

# =========================================================
# PDF Processing
# =========================================================


def extract_text_from_pdf(pdf_file) -> dict:
    try:
        reader = PdfReader(pdf_file)
    except Exception:
        st.exception(e)
        raise

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    return {"text": text, "num_pages": len(reader.pages)}


def create_chunks(text: str) -> dict:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    return {"chunks": chunks, "num_chunks": len(chunks)}


def build_vector_store(chunks: List[str]):
    """Embed chunks and build the retriever. Chunks must be wrapped as
    Document objects before being passed to Chroma.from_documents."""
    documents = [Document(page_content=chunk) for chunk in chunks]
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10},
    )
    return vector_store, retriever


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="RizooSphere Assistant",
    page_icon="🤖",
    layout="centered",
)

# =========================================================
# Session State
# =========================================================

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "response" not in st.session_state:
    st.session_state.response = None

if "response_type" not in st.session_state:
    st.session_state.response_type = "NOT AVAILABLE"

if "sources" not in st.session_state:
    st.session_state.sources = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# =========================================================
# Title
# =========================================================

st.title("Knowledge Assistant")

st.markdown("##### LangGraph + Corrective RAG + Gemini")

st.markdown("---")

# =========================================================
# PDF Status
# =========================================================

st.subheader("PDF Status")

if st.session_state.pdf_ready:
    st.success(f"READY ✓ {st.session_state.pdf_name}")
else:
    st.info("No PDF Uploaded")

st.markdown("---")

# =========================================================
# Answer Section
# =========================================================

st.subheader("Answer")

if st.session_state.response:
    st.write(st.session_state.response)
else:
    st.write("Your answer will appear here.")

st.markdown("---")

# =========================================================
# Response Type
# =========================================================

st.subheader("Response Type")

st.write(st.session_state.response_type)

st.markdown("---")

# =========================================================
# Sources
# =========================================================

st.subheader("Sources")

if len(st.session_state.sources) > 0:
    for source in st.session_state.sources:
        st.write(source)
else:
    st.write("No Sources")

st.markdown("---")

# =========================================================
# Question Input
# =========================================================

question = st.chat_input("Ask anything...")

# =========================================================
# PDF Upload
# =========================================================

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

# =========================================================
# PDF Processing
# =========================================================

# Only (re)process when a new/different PDF has been uploaded, so the
# pipeline doesn't rebuild the vector store on every Streamlit rerun.
if uploaded_pdf is not None and uploaded_pdf.name != st.session_state.pdf_name:

    with st.spinner("Processing PDF..."):
        try:
            extraction = extract_text_from_pdf(uploaded_pdf)
            chunking = create_chunks(extraction["text"])
            vector_store, retriever = build_vector_store(chunking["chunks"])

            st.session_state.vector_store = vector_store
            st.session_state.retriever = retriever
            st.session_state.pdf_ready = True
            st.session_state.pdf_name = uploaded_pdf.name

            logger.info(
                "Processed PDF '%s': %d pages, %d chunks",
                uploaded_pdf.name,
                extraction["num_pages"],
                chunking["num_chunks"],
            )
        except Exception:
            st.exception(e)
            st.session_state.pdf_ready = False
            st.session_state.retriever = None
            st.session_state.vector_store = None
            st.error("Failed to process the uploaded PDF. Please try again.")
        else:
            st.success("PDF Processed Successfully.")

# =========================================================
# Ask Question
# =========================================================

if question:

    with st.spinner("Thinking..."):
        try:
            initial_state = build_initial_state(question)
            response = graph.invoke(initial_state)

            st.session_state.response = response.get("answer", "No answer generated.")
            st.session_state.response_type = response.get("response_type", "UNKNOWN")
            st.session_state.sources = response.get("sources", [])
        except Exception:
            logger.exception("Graph invocation failed")
            st.session_state.response = "Something went wrong while generating the answer."
            st.session_state.response_type = "ERROR"
            st.session_state.sources = []

    st.rerun()
