"""
RAG Engine for Legal & Compliance Documents
Loads TXT legal documents, chunks by ARTICLE/REGULATION headings,
embeds via Ollama nomic-embed-text, stores in ChromaDB,
and generates answers using llama3.1 with legal-analysis-tuned prompts.
"""

import os
import re
import hashlib
import chromadb
import ollama
from typing import List, Dict


def load_text_file(filepath: str) -> str:
    """Load a plain text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(folder: str) -> List[Dict]:
    """
    Load all TXT files from the given folder.
    Returns a list of dicts with 'text' and 'source' keys.
    """
    documents = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filename.lower().endswith(".txt"):
            text = load_text_file(filepath)
            documents.append({"text": text, "source": filename})
    return documents


def chunk_by_clauses(text: str, source: str, max_chunk_size: int = 800) -> List[Dict]:
    """
    Split legal documents by ARTICLE / REGULATION headings.
    Preserves clause structure for precise legal reference.
    Falls back to sub-section splitting for oversized articles.
    """
    # Split on ARTICLE or REGULATION headings
    clause_pattern = r'\n(?=(?:ARTICLE|REGULATION)\s+\d+)'
    sections = re.split(clause_pattern, text)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section or len(section) < 30:
            continue

        # Extract heading
        heading_match = re.match(r'^((?:ARTICLE|REGULATION)\s+\d+[^:\n]*:?\s*[^\n]*)', section)
        heading = heading_match.group(1).strip() if heading_match else "General"

        if len(section) <= max_chunk_size:
            chunks.append({
                "text": section,
                "source": source,
                "chunk_id": f"{source}_clause_{len(chunks)}",
                "section": heading
            })
        else:
            # Split large articles by sub-section numbers (e.g. "1.1", "4.3")
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
                            "chunk_id": f"{source}_clause_{len(chunks)}",
                            "section": heading
                        })
                    current_chunk = sub

            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": source,
                    "chunk_id": f"{source}_clause_{len(chunks)}",
                    "section": heading
                })

    return chunks


def get_embedding(text: str) -> List[float]:
    """Get embedding using Ollama nomic-embed-text."""
    response = ollama.embed(model="nomic-embed-text", input=text)
    return response["embeddings"][0]


class LegalComplianceEngine:
    """RAG Engine for Legal & Compliance documents with clause-aware retrieval."""

    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="legal_compliance",
            metadata={"hnsw:space": "cosine"}
        )
        self.is_ingested = self.collection.count() > 0

    def ingest(self, documents_folder: str):
        """Load, chunk, embed, and store all legal documents."""
        documents = load_documents(documents_folder)
        if not documents:
            print("No documents found in", documents_folder)
            return 0

        all_chunks = []
        for doc in documents:
            chunks = chunk_by_clauses(doc["text"], doc["source"])
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
                    "section": chunk.get("section", "")
                })
                embeddings.append(embedding)
                print(f"  Embedded chunk {i + 1}/{len(all_chunks)}: {chunk.get('section', '')[:50]}")
            except Exception as e:
                print(f"  WARNING: Failed to embed chunk {i + 1}: {e}")
                continue

        if not ids:
            print("ERROR: No chunks were successfully embedded.")
            return 0

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
        """Retrieve the most relevant legal clauses for a query."""
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
                "section": results["metadatas"][0][i].get("section", ""),
                "score": round(1 - results["distances"][0][i], 4)
            })

        return retrieved

    def generate_answer(self, query: str, top_k: int = 3) -> Dict:
        """Retrieve context and generate a legal analysis answer."""
        sources = self.retrieve(query, top_k)

        if not sources:
            return {
                "answer": "No relevant legal documents found. Please ingest documents first.",
                "sources": []
            }

        context = "\n\n---\n\n".join(
            [f"[Source: {s['source']} | {s.get('section', '')}]\n{s['text']}" for s in sources]
        )

        system_prompt = (
            "You are a Legal & Compliance Analyst for TechNova Solutions. "
            "Answer questions accurately based ONLY on the provided legal document context. "
            "Always cite the specific article, section, or regulation number when answering. "
            "Provide precise legal analysis — do not speculate beyond what the documents state. "
            "If the answer is not found in the context, say 'This is not covered in the current legal documents.' "
            "Be thorough, precise, and professionally formal."
        )

        user_prompt = f"Legal document context:\n\n{context}\n\n---\n\nLegal/Compliance Question: {query}"

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
