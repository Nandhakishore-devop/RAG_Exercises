# RAG Exercises

A collection of four Retrieval-Augmented Generation (RAG) applications, each demonstrating a different real-world use case. All exercises use **Ollama** for embeddings and LLM inference, **ChromaDB** as the vector store, and **Flask** for the web UI.

## Use Cases

| # | Exercise | Data Format | Description |
|---|----------|-------------|-------------|
| 1 | **Employee Knowledge Base** | PDF | HR policy assistant that answers employee questions from company policy documents |
| 2 | **Technical Documentation** | DOCX | API documentation assistant that helps developers find endpoints, code snippets, and SDK usage |
| 3 | **Customer Support** | JSON | Support ticket autocomplete that drafts responses based on similar past resolved tickets |
| 4 | **Legal Compliance** | TXT | Legal document assistant that retrieves relevant clauses and regulations for compliance queries |

## Tech Stack

- **LLM**: Ollama (`llama3.1`)
- **Embeddings**: Ollama (`nomic-embed-text`)
- **Vector Store**: ChromaDB (persistent, cosine similarity)
- **Web Framework**: Flask
- **Document Loaders**: PyPDF2 (PDF), python-docx (DOCX), json (JSON), plain text (TXT)

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Pull Ollama Models

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### Run Any Exercise

```bash
cd 1_employee_knowledge_base
python app.py
```

Each app runs on a different port:
| Exercise | Port |
|----------|------|
| 1 - Employee KB | `http://127.0.0.1:5001` |
| 2 - Technical Docs | `http://127.0.0.1:5002` |
| 3 - Customer Support | `http://127.0.0.1:5003` |
| 4 - Legal Compliance | `http://127.0.0.1:5004` |

## Project Structure

```
RAG Exercises/
├── requirements.txt
├── 1_employee_knowledge_base/
│   ├── app.py                 # Flask app
│   ├── rag_logic.py           # RAG engine (PDF loader + chunking)
│   ├── documents/             # HR Policy PDF
│   ├── templates/             # HTML templates
│   └── static/                # CSS styles
├── 2_technical_docs/
│   ├── app.py
│   ├── rag_logic.py           # RAG engine (DOCX loader + chunking)
│   ├── documents/             # API documentation DOCX
│   ├── templates/
│   └── static/
├── 3_customer_support/
│   ├── app.py
│   ├── rag_logic.py           # RAG engine (JSON ticket loader)
│   ├── documents/             # Resolved tickets JSON
│   ├── templates/
│   └── static/
└── 4_legal_compliance/
    ├── app.py
    ├── rag_logic.py           # RAG engine (TXT loader + clause chunking)
    ├── documents/             # Legal documents TXT
    ├── templates/
    └── static/
```

## How It Works

1. **Ingestion**: Documents are loaded, split into chunks, and embedded using `nomic-embed-text`
2. **Storage**: Embeddings are stored in ChromaDB with metadata (source, section headings)
3. **Retrieval**: User queries are embedded and matched against stored chunks via cosine similarity
4. **Generation**: Retrieved context is passed to `llama3.1` to generate accurate, cited answers
