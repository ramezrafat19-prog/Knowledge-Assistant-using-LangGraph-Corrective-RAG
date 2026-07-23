import json
import logging
import os
from typing import List, Optional, TypedDict

import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader

# =========================================================
# Logging
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("knowledge_assistant")

# =========================================================
# Config
# =========================================================
DATASET_PATH = "RizooSphere Restaurant.pdf"  # predefined local dataset path

LLM_MODEL = "gemini-2.5-flash"
# Local, free, no API key needed. Runs on CPU.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_RETRIES = 2
VALID_ROUTES = ["direct", "retrieve", "decline"]


def get_api_key() -> Optional[str]:
    return "AQ.Ab8RN6Jgb79Go4kQgVlcOk6pMaqZncFDw5kMf2qqpUlyL6uC2Q"


# =========================================================
# LLM (cached - built once per session, not on every rerun)
# =========================================================
@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGoogleGenerativeAI:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Set it in .streamlit/secrets.toml or as an env var."
        )
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
    )


# =========================================================
# Embeddings (cached - HuggingFace runs locally, no API key)
# =========================================================
@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


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
    sources: list
    relevance: str
    response_type: str


# =========================================================
# Prompts (unchanged)
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

GRADER_PROMPT = """
You are a retrieval evaluator.

Your task is to determine whether the retrieved context contains enough relevant information to answer the user's question.

Rules:
- Read both the question and the retrieved context carefully.
- If the context contains sufficient information, return true.
- If the context is missing the required information or is unrelated, return false.
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

DECLINE_MESSAGE = """
This question is outside the scope of
the available documents.

Please ask a question related to the
provided documentation.
"""


# =========================================================
# JSON parsing helper
# =========================================================
def parse_json(response) -> dict:
    text = (response.content or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    if not text:
        raise ValueError("Empty LLM response - nothing to parse as JSON.")
    return json.loads(text)


# =========================================================
# Nodes
# =========================================================
def router_node(state: GraphState) -> GraphState:
    question = state["question"]
    llm = get_llm()
    prompt = PromptTemplate(template=ROUTER_PROMPT_V1, input_variables=["question"])
    final_prompt = prompt.format(question=question)

    route, confidence = "decline", 0.0
    try:
        response = llm.invoke(final_prompt)
        result = parse_json(response)
        route = str(result["route"]).strip().lower()
        confidence = float(result["confidence"])
    except Exception:
        logger.exception("router_node: failed to classify question, defaulting to decline.")

    if route not in VALID_ROUTES:
        route, confidence = "decline", 0.0

    state["route"] = route
    state["confidence"] = confidence
    return state


def direct_node(state: GraphState) -> GraphState:
    question = state["question"]
    llm = get_llm()
    prompt = PromptTemplate(template=DIRECT_PROMPT, input_variables=["question"])
    final_prompt = prompt.format(question=question)

    try:
        response = llm.invoke(final_prompt)
        output = parse_json(response)
        state["answer"] = output.get("answer", "No answer generated.")
        state["grounded"] = output.get("grounded", False)
    except Exception:
        logger.exception("direct_node: failed to generate a direct answer.")
        state["answer"] = "Sorry, I couldn't generate an answer right now."
        state["grounded"] = False

    state["response_type"] = "DIRECT"
    return state


def retrieve_node(state: GraphState) -> GraphState:
    question = state["question"]
    retriever = get_retriever()

    try:
        documents = retriever.invoke(question)
    except Exception:
        logger.exception("retrieve_node: retrieval failed.")
        documents = []

    state["documents"] = documents
    if len(documents) == 0:
        state["retries"] = state.get("retries", 0) + 1
    return state


def documents_to_text(documents: List[Document]) -> str:
    return "\n".join(doc.page_content for doc in documents)


def grader_node(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state["documents"]
    context = documents_to_text(documents)

    llm = get_llm()
    prompt = PromptTemplate(template=GRADER_PROMPT, input_variables=["question", "context"])
    final_prompt = prompt.format(question=question, context=context)

    relevance = "rewrite"
    try:
        response = llm.invoke(final_prompt)
        output = parse_json(response)
        relevance = output.get("relevance", "rewrite")
    except Exception:
        logger.exception("grader_node: grading failed, defaulting to rewrite.")

    state["relevance"] = relevance if relevance in ("generate", "rewrite") else "rewrite"
    return state


def rewrite_node(state: GraphState) -> GraphState:
    question = state["question"]
    llm = get_llm()
    prompt = PromptTemplate(template=REWRITE_PROMPT, input_variables=["question"])
    final_prompt = prompt.format(question=question)

    try:
        response = llm.invoke(final_prompt)
        rewritten_question = response.content.strip()
        if rewritten_question:
            state["question"] = rewritten_question
    except Exception:
        logger.exception("rewrite_node: rewrite failed, keeping original question.")

    state["retries"] = state.get("retries", 0) + 1
    return state


def generate_node(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state["documents"]
    context = documents_to_text(documents)

    llm = get_llm()
    prompt = PromptTemplate(template=GENERATION_PROMPT, input_variables=["question", "context"])
    final_prompt = prompt.format(question=question, context=context)

    try:
        response = llm.invoke(final_prompt)
        output = parse_json(response)
        state["answer"] = output.get("answer", "No answer generated.")
        state["grounded"] = output.get("grounded", False)
        state["sources"] = output.get("sources", [])
    except Exception:
        logger.exception("generate_node: generation failed.")
        state["answer"] = "Sorry, I couldn't generate an answer right now."
        state["grounded"] = False
        state["sources"] = []

    state["response_type"] = "RAG - GROUNDED" if state.get("grounded") else "RAG - NOT GROUNDED"
    return state


def decline_node(state: GraphState) -> GraphState:
    state["answer"] = DECLINE_MESSAGE
    state["grounded"] = False
    state["sources"] = []
    state["response_type"] = "DECLINED"
    return state


# =========================================================
# Graph
# =========================================================
def route_question(state: GraphState) -> str:
    return state["route"]


def grade_documents(state: GraphState) -> str:
    return state["relevance"]


def should_retry(state: GraphState) -> str:
    if state["retries"] >= MAX_RETRIES:
        return "end"
    return "retrieve"


@st.cache_resource(show_spinner=False)
def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("router", router_node)
    workflow.add_node("direct", direct_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("decline", decline_node)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        route_question,
        {"direct": "direct", "retrieve": "retrieve", "decline": "decline"},
    )

    workflow.add_edge("direct", END)
    workflow.add_edge("decline", END)

    workflow.add_edge("retrieve", "grader")
    workflow.add_conditional_edges(
        "grader",
        grade_documents,
        {"generate": "generate", "rewrite": "rewrite"},
    )

    workflow.add_edge("generate", END)
    workflow.add_conditional_edges(
        "rewrite",
        should_retry,
        {"retrieve": "retrieve", "end": END},
    )

    return workflow.compile()


# =========================================================
# Dataset loading + vector store (auto-loaded from local path,
# cached so it only runs once instead of on every rerun)
# =========================================================
def extract_text_from_pdf(path: str) -> dict:
    reader = PdfReader(path)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return {"text": text, "num_pages": len(reader.pages)}


def create_chunks(text: str) -> dict:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    return {"chunks": chunks, "num_chunks": len(chunks)}


@st.cache_resource(show_spinner=False)
def build_vector_store(dataset_path: str):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Place your PDF there or update DATASET_PATH."
        )

    extracted = extract_text_from_pdf(dataset_path)
    chunked = create_chunks(extracted["text"])

    documents = [Document(page_content=chunk) for chunk in chunked["chunks"]]

    # HuggingFace embeddings run locally - no API key, no quota, no network calls.
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)

    logger.info(
        "Vector store built: %s pages, %s chunks.",
        extracted["num_pages"],
        chunked["num_chunks"],
    )
    return vector_store, extracted["num_pages"]


def get_retriever():
    vector_store, _ = build_vector_store(DATASET_PATH)
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10},
    )


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(
    page_title="RizooSphere Assistant",
    page_icon="🤖",
    layout="centered",
)

if "response" not in st.session_state:
    st.session_state.response = None
if "response_type" not in st.session_state:
    st.session_state.response_type = "NOT AVAILABLE"
if "sources" not in st.session_state:
    st.session_state.sources = []

st.title("Knowledge Assistant")
st.markdown("##### LangGraph + Corrective RAG + Gemini + HuggingFace Embeddings")
st.markdown("---")

# --- Dataset Status ---
st.subheader("PDF Status")
try:
    _, num_pages = build_vector_store(DATASET_PATH)
    st.success(f"READY ✓ {DATASET_PATH} ({num_pages} pages)")
except Exception as e:
    logger.exception("Failed to load dataset / build vector store.")
    st.error(f"Dataset not ready: {e}")

st.markdown("---")

# --- Answer Section ---
st.subheader("Answer")
if st.session_state.response:
    st.write(st.session_state.response)
else:
    st.write("Your answer will appear here.")

st.markdown("---")

# --- Response Type ---
st.subheader("Response Type")
st.write(st.session_state.response_type)

st.markdown("---")

# --- Sources ---
st.subheader("Sources")
if len(st.session_state.sources) > 0:
    for source in st.session_state.sources:
        st.write(source)
else:
    st.write("No Sources")

st.markdown("---")

# --- Question Input ---
question = st.chat_input("Ask anything...")

# --- Ask Question ---
if question:
    with st.spinner("Thinking..."):
        try:
            graph = build_graph()
            initial_state: GraphState = {
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
            response = graph.invoke(initial_state)

            st.session_state.response = response.get("answer", "No answer generated.")
            st.session_state.response_type = response.get("response_type", "NOT AVAILABLE")
            st.session_state.sources = response.get("sources", [])
        except Exception as e:
            logger.exception("Graph invocation failed.")
            st.session_state.response = f"Something went wrong: {e}"
            st.session_state.response_type = "ERROR"
            st.session_state.sources = []

    st.rerun()
