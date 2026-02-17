"""
Flask App — Technical Documentation RAG
Provides a chat interface for developers to query API documentation.
"""

import os
from flask import Flask, render_template, request, jsonify
from rag_logic import TechDocsEngine

app = Flask(__name__)

DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "documents")
engine = TechDocsEngine(db_path=os.path.join(os.path.dirname(__file__), "chroma_db"))

if not engine.is_ingested:
    print("Ingesting technical documentation...")
    count = engine.ingest(DOCS_FOLDER)
    print(f"Ingested {count} chunks.")
else:
    print(f"Docs already ingested. {engine.doc_count()} chunks in store.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    result = engine.generate_answer(user_query, top_k=3)
    return jsonify(result)


@app.route("/status")
def status():
    return jsonify({
        "ingested": engine.is_ingested,
        "chunk_count": engine.doc_count()
    })


if __name__ == "__main__":
    app.run(debug=True, port=5002)
