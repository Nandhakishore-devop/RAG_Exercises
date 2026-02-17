"""
RAG Engine for Employee Knowledge Base
Loads PDF HR policy documents, chunks by sections (SECTION/CHAPTER headings),
embeds via Ollama nomic-embed-text, stores in ChromaDB,
and generates answers using llama3.1 with HR-tuned prompts.
"""

import os
import re
import hashlib
import chromadb
import ollama
from PyPDF2 import PdfReader
from typing import List, Dict


def load_pdf_file(filepath: str) -> str:
    """Load a PDF file and extract text from all pages."""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def load_documents(folder: str) -> List[Dict]:
    """
    Load all PDF files from the given folder.
    Returns a list of dicts with 'text' and 'source' keys.
    """
    documents = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filename.lower().endswith(".pdf"):
            text = load_pdf_file(filepath)
            documents.append({"text": text, "source": filename})
    return documents


def chunk_by_sections(text: str, source: str, max_chunk_size: int = 800) -> List[Dict]:
    """
    Split text into chunks based on SECTION / CHAPTER headings.
    Keeps logically related HR policy content together.
    Falls back to paragraph splitting for oversized sections.
    """
    # Split on SECTION or CHAPTER headings (e.g. "SECTION 1: REMOTE WORK POLICY")
    section_pattern = r'\n(?=(?:SECTION|CHAPTER)\s+\d+)'
    sections = re.split(section_pattern, text)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section or len(section) < 30:
            continue

        # Extract heading for metadata
        heading_match = re.match(r'^((?:SECTION|CHAPTER)\s+\d+[^:\n]*:?\s*[^\n]*)', section)
        heading = heading_match.group(1).strip() if heading_match else "General"

        if len(section) <= max_chunk_size:
            chunks.append({
                "text": section,
                "source": source,
                "chunk_id": f"{source}_sec_{len(chunks)}",
                "heading": heading
            })
        else:
            # Split large sections by sub-section numbers (e.g. "1.1", "10.3")
            sub_pattern = r'\n(?=\d+\.\d+\s)'
            sub_sections = re.split(sub_pattern, section)

            current_chunk = ""
            for sub in sub_sections:
                sub = sub.strip()
                if not sub:
                    continue
                if len(current_chunk) + len(sub) + 2 <= max_chunk_size:
                    current_chunk += ("\n\n" + sub) if current_chunk else sub
                else:
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "source": source,
                            "chunk_id": f"{source}_sec_{len(chunks)}",
                            "heading": heading
                        })
                    current_chunk = sub

            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": source,
                    "chunk_id": f"{source}_sec_{len(chunks)}",
                    "heading": heading
                })

    return chunks


def get_embedding(text: str) -> List[float]:
    """Get embedding for a text using Ollama nomic-embed-text."""
    response = ollama.embed(model="nomic-embed-text", input=text)
    return response["embeddings"][0]


class EmployeeKBEngine:
    """RAG Engine tailored for Employee Knowledge Base / HR Policies."""

    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="employee_kb",
            metadata={"hnsw:space": "cosine"}
        )
        self.is_ingested = self.collection.count() > 0

    def ingest(self, documents_folder: str):
        """Load, chunk, embed, and store all documents from the folder."""
        documents = load_documents(documents_folder)
        if not documents:
            print("No documents found in", documents_folder)
            return 0

        all_chunks = []
        for doc in documents:
            chunks = chunk_by_sections(doc["text"], doc["source"])
            all_chunks.extend(chunks)

        print(f"Processing {len(all_chunks)} chunks from {len(documents)} documents...")

        ids = []
        embeddings = []
        metadatas = []
        texts = []

        for i, chunk in enumerate(all_chunks):
            try:
                chunk_id = hashlib.md5(chunk["text"].encode()).hexdigest()
                embedding = get_embedding(chunk["text"])

                ids.append(chunk_id)
                texts.append(chunk["text"])
                metadatas.append({
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "heading": chunk.get("heading", "")
                })
                embeddings.append(embedding)
                print(f"  Embedded chunk {i + 1}/{len(all_chunks)}: {chunk.get('heading', '')[:50]}")
            except Exception as e:
                print(f"  WARNING: Failed to embed chunk {i + 1}: {e}")
                continue

        if not ids:
            print("ERROR: No chunks were successfully embedded.")
            return 0

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        self.is_ingested = True
        print(f"Successfully ingested {len(ids)} chunks.")
        return len(ids)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve the most relevant chunks for a query."""
        if not self.is_ingested:
            return []

        query_embedding = get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "chunk_id": results["metadatas"][0][i]["chunk_id"],
                "heading": results["metadatas"][0][i].get("heading", ""),
                "score": round(1 - results["distances"][0][i], 4)
            })

        return retrieved

    def generate_answer(self, query: str, top_k: int = 3) -> Dict:
        """Retrieve context and generate an answer using llama3.1."""
        sources = self.retrieve(query, top_k)

        if not sources:
            return {
                "answer": "No relevant documents found. Please ingest documents first.",
                "sources": []
            }

        # Build context from retrieved chunks
        context = "\n\n---\n\n".join(
            [f"[Source: {s['source']} | {s.get('heading', '')}]\n{s['text']}" for s in sources]
        )

        system_prompt = (
            "You are an HR Policy Assistant for TechNova Solutions. "
            "Answer employee questions accurately based ONLY on the provided context. "
            "Always cite the source document and relevant section when answering. "
            "If the answer is not found in the context, say 'This information is not available in the current HR documents.' "
            "Be helpful, professional, and concise."
        )

        user_prompt = f"Context from HR documents:\n\n{context}\n\n---\n\nEmployee Question: {query}"

        try:
            response = ollama.chat(
                model="llama3.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            answer = response["message"]["content"]
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"

        return {
            "answer": answer,
            "sources": sources
        }

    def doc_count(self) -> int:
        """Return the number of chunks in the vector store."""
        return self.collection.count()
