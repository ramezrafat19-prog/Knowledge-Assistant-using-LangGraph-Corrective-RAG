# 🤖 Knowledge Assistant using LangGraph & Corrective RAG

An end-to-end AI-powered Knowledge Assistant built with LangGraph, Corrective RAG, Gemini, ChromaDB, and Streamlit. The system intelligently answers questions using both general knowledge and uploaded PDF documents while ensuring grounded, context-aware responses.

---

## 📖 Overview

This project implements a production-inspired Retrieval-Augmented Generation (RAG) pipeline with a multi-node LangGraph architecture for Question Answering (QA) tasks.

Unlike traditional RAG systems, this assistant uses:

- Intelligent Question Routing
- Corrective RAG Workflow
- Query Rewriting
- Document Relevance Grading
- Grounded Answer Generation
- PDF Knowledge Base Integration

Users can ask questions directly or upload PDF documents to create their own knowledge base. The system automatically decides whether to:

1. Answer from general knowledge.
2. Retrieve information from uploaded documents.
3. Decline out-of-scope questions.

---

## 🚀 Features

- PDF Upload & Processing
- Text Extraction from PDF
- Recursive Chunking
- Gemini Embeddings
- ChromaDB Vector Store
- LangGraph Workflow
- Router Node
- Direct Node
- Retrieve Node
- Grader Node
- Rewrite Node
- Generate Node
- Decline Node
- Corrective RAG Pipeline
- Query Rewriting
- MMR Retrieval Strategy
- Grounded Responses
- Source Citation Support
- Streamlit Web Interface
- Response Type Detection
- RAG Evaluation Ready (RAGAS)
- Deployment Ready

---

## 🛠️ Tech Stack

- Python
- LangChain
- LangGraph
- Google Gemini
- ChromaDB
- Streamlit
- PyPDF
- RAGAS
- Google Generative AI Embeddings
- Pandas
- NumPy
- Google Colab

---

## 📂 Project Workflow

```text
                     User Question
                            │
                            ▼
                        Router
             ┌──────────┬───────────┐
             │          │           │
             ▼          ▼           ▼
          Direct     Retrieve    Decline
                          │
                          ▼
                      Retriever
                          │
                          ▼
                        Grader
                     ┌────┴────┐
                     ▼         ▼
                 Relevant    Not Relevant
                     │             │
                     ▼             ▼
                 Generate       Rewrite
                     │             │
                     └──────┬──────┘
                            ▼
                       Retrieve Again
                            │
                            ▼
                         Answer
                            │
                            ▼
                        Streamlit
```

---

## 🧠 LangGraph Architecture

```text
                  START
                     │
                     ▼
                  Router
        ┌─────────┬──────────┐
        ▼         ▼          ▼
     Direct    Retrieve    Decline
                   │
                   ▼
                Grader
               /      \
             Yes       No
              │         │
              ▼         ▼
         Generate    Rewrite
              │         │
              └────┬────┘
                   ▼
               Retrieve
                   │
                   ▼
                  END
```

---

## 📑 PDF Processing Pipeline

```text
PDF
 │
 ▼
Extract Text
 │
 ▼
Chunking
 │
 ▼
Gemini Embeddings
 │
 ▼
ChromaDB
 │
 ▼
Retriever
 │
 ▼
LangGraph
```

---

## 🔍 Retrieval Strategy

The project uses:

- MMR (Maximal Marginal Relevance)
- `k = 4`
- `fetch_k = 10`

This improves retrieval diversity and reduces redundant context retrieval.

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)
```

---

## 📊 Response Types

The assistant supports three response categories:

| Response Type | Description |
|---------------|-------------|
| GENERAL KNOWLEDGE | Answered directly using Gemini |
| DOCUMENT RETRIEVAL | Answered using retrieved PDF context |
| OUT OF SCOPE | Declined by the system |

---

## 💻 Streamlit UI

The application provides a ChatGPT-inspired interface featuring:

- PDF Upload
- Question Input
- Response Type Display
- Grounded Answer Indicator
- Source Citations
- Single Page UI
- No Chat History

---

## 📈 Evaluation Metrics

The RAG pipeline is designed to be evaluated using RAGAS metrics:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Additional testing includes:

- Router Accuracy
- Retrieval Quality
- Query Rewrite Effectiveness
- Groundedness Validation

---

## 📌 Future Improvements

- Multi-PDF Support
- Hybrid Search (BM25 + Vector Search)
- Conversation Memory
- Multi-Agent LangGraph
- Redis Caching
- OCR Support for Scanned PDFs
- Azure / AWS Deployment
- Docker Containerization
- Authentication System
- Observability with LangSmith

---

## 📚 Example Use Cases

### General Knowledge

> What is Artificial Intelligence?

Response Type:

```text
GENERAL KNOWLEDGE
```

---

### Document Retrieval

> Explain the CNN architecture mentioned in the uploaded PDF.

Response Type:

```text
DOCUMENT RETRIEVAL
```

Sources:

```text
Page 5
Page 7
```

---

### Out of Scope

> Which stock should I invest in?

Response Type:

```text
OUT OF SCOPE
```

---

## 📂 Project Structure

```text
Knowledge-Assistant/
│
├── app.py
├── graph.py
├── prompts.py
├── pdf_processing.py
├── chunking.py
├── retriever.py
├── vector_store.py
├── embeddings.py
├── requirements.txt
├── README.md
│
└── assets/
```

---

## 🔥 Example Workflow

```text
Upload PDF
     │
     ▼
Process PDF
     │
     ▼
Ask Question
     │
     ▼
LangGraph
     │
     ▼
Retrieve Context
     │
     ▼
Generate Grounded Answer
     │
     ▼
Display Sources
```

---

## 👨‍💻 Author

### Ramez Rafat

Software Engineering Student | Artificial Intelligence & Machine Learning Enthusiast

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Generative AI
- LLMs & RAG Systems

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ Star on GitHub!
