"""
RAG Engine for Customer Support Ticket Autocomplete
Loads resolved support tickets from JSON, chunks per-ticket,
embeds via Ollama nomic-embed-text, stores in ChromaDB,
and generates draft responses matching similar past resolutions.
"""

import os
import json
import hashlib
import chromadb
import ollama
from typing import List, Dict


def load_json_tickets(filepath: str) -> List[Dict]:
    """Load tickets from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_documents(folder: str) -> List[Dict]:
    """Load all JSON ticket files from the folder."""
    all_tickets = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filename.lower().endswith(".json"):
            tickets = load_json_tickets(filepath)
            for ticket in tickets:
                ticket["source"] = filename
            all_tickets.extend(tickets)
    return all_tickets


def chunk_per_ticket(tickets: List[Dict]) -> List[Dict]:
    """
    Create one chunk per ticket, combining subject, description,
    and resolution into a single searchable text block.
    """
    chunks = []
    for ticket in tickets:
        text = (
            f"Ticket ID: {ticket['ticket_id']}\n"
            f"Subject: {ticket['subject']}\n"
            f"Category: {ticket['category']}\n"
            f"Priority: {ticket['priority']}\n"
            f"Issue: {ticket['description']}\n"
            f"Resolution: {ticket['resolution']}\n"
            f"Resolved by: {ticket.get('resolved_by', 'N/A')}\n"
            f"Date: {ticket.get('date_resolved', 'N/A')}"
        )

        chunks.append({
            "text": text,
            "source": ticket.get("source", "tickets.json"),
            "chunk_id": ticket["ticket_id"],
            "subject": ticket["subject"],
            "category": ticket["category"],
            "priority": ticket["priority"]
        })

    return chunks


def get_embedding(text: str) -> List[float]:
    """Get embedding using Ollama nomic-embed-text."""
    response = ollama.embed(model="nomic-embed-text", input=text)
    return response["embeddings"][0]


class CustomerSupportEngine:
    """RAG Engine for Customer Support — finds similar past tickets and drafts responses."""

    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="support_tickets",
            metadata={"hnsw:space": "cosine"}
        )
        self.is_ingested = self.collection.count() > 0

    def ingest(self, documents_folder: str):
        """Load, process, embed, and store support tickets."""
        all_tickets = load_documents(documents_folder)
        if not all_tickets:
            print("No ticket files found in", documents_folder)
            return 0

        chunks = chunk_per_ticket(all_tickets)
        print(f"Processing {len(chunks)} tickets...")

        ids = []
        embeddings = []
        metadatas = []
        texts = []

        for i, chunk in enumerate(chunks):
            try:
                chunk_id = hashlib.md5(chunk["text"].encode()).hexdigest()
                embedding = get_embedding(chunk["text"])

                ids.append(chunk_id)
                texts.append(chunk["text"])
                metadatas.append({
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "subject": chunk["subject"],
                    "category": chunk["category"],
                    "priority": chunk["priority"]
                })
                embeddings.append(embedding)
                print(f"  Embedded ticket {i + 1}/{len(chunks)}: {chunk['chunk_id']}")
            except Exception as e:
                print(f"  WARNING: Failed to embed ticket {chunk.get('chunk_id', i)}: {e}")
                continue

        if not ids:
            print("ERROR: No tickets were successfully embedded.")
            return 0

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        self.is_ingested = True
        print(f"Successfully ingested {len(ids)} tickets.")
        return len(ids)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most similar past tickets."""
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
                "ticket_id": results["metadatas"][0][i]["chunk_id"],
                "subject": results["metadatas"][0][i]["subject"],
                "category": results["metadatas"][0][i]["category"],
                "priority": results["metadatas"][0][i]["priority"],
                "score": round(1 - results["distances"][0][i], 4)
            })

        return retrieved

    def generate_answer(self, query: str, top_k: int = 3) -> Dict:
        """Find similar past tickets and draft a support response."""
        sources = self.retrieve(query, top_k)

        if not sources:
            return {
                "answer": "No similar tickets found. Please ingest ticket data first.",
                "sources": []
            }

        context = "\n\n---\n\n".join(
            [f"[Similar Ticket: {s['ticket_id']} | Category: {s['category']}]\n{s['text']}" for s in sources]
        )

        system_prompt = (
            "You are a Customer Support Agent Assistant. A new support ticket has been submitted. "
            "Based on similar past resolved tickets provided as context, draft a helpful response for the agent. "
            "Your response should:\n"
            "1. Acknowledge the customer's issue\n"
            "2. Suggest a resolution based on how similar issues were resolved before\n"
            "3. Include specific steps the agent should take\n"
            "4. Reference the similar ticket IDs for the agent's reference\n"
            "Be professional, empathetic, and action-oriented."
        )

        user_prompt = f"Similar past resolved tickets:\n\n{context}\n\n---\n\nNew ticket/issue: {query}"

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
            answer = f"Error generating response: {str(e)}"

        return {
            "answer": answer,
            "sources": sources
        }

    def doc_count(self) -> int:
        return self.collection.count()
