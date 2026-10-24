import hashlib
import json
import re
from pathlib import Path

import pymupdf
import numpy as np
import ollama

# Configuration
PDF_FILE = Path("sample.pdf")
INDEX_FILE = Path("vector_store.json")

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:0.5b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

# Starting threshold; evaluate it with your PDF.
MIN_SIMILARITY = 0.35

NOT_FOUND = "The answer is not available in the provided PDF."


def get_pdf_hash():
    return hashlib.sha256(PDF_FILE.read_bytes()).hexdigest()


# Extract text from every PDF page and create chunks.
def extract_chunks():
    chunks = []

    with pymupdf.open(PDF_FILE) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = " ".join(page.get_text("text").split())

            if not text:
                continue

            start = 0

            while start < len(text):
                end = min(start + CHUNK_SIZE, len(text))
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append({
                        "chunk_id": len(chunks) + 1,
                        "page": page_number,
                        "text": chunk_text
                    })

                if end == len(text):
                    break

                start = end - CHUNK_OVERLAP

    return chunks


# Generate local embeddings and save them in JSON.
def create_embeddings(chunks):
    for i, chunk in enumerate(chunks, start=1):
        response = ollama.embed(
            model=EMBED_MODEL,
            input=chunk["text"]
        )

        chunk["embedding"] = response["embeddings"][0]
        print(f"\rGenerating embeddings: {i}/{len(chunks)}", end="")

    data = {
        "pdf_hash": get_pdf_hash(),
        "embedding_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunks": chunks
    }

    INDEX_FILE.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nIndex created with {len(chunks)} chunks.")


# Load the existing index or create it.
def load_index():
    pdf_hash = get_pdf_hash()

    if INDEX_FILE.exists():
        try:
            data = json.loads(
                INDEX_FILE.read_text(encoding="utf-8")
            )

            if isinstance(data, dict):
                chunks = data.get("chunks", [])

                valid = (
                    data.get("pdf_hash") == pdf_hash
                    and data.get("embedding_model") == EMBED_MODEL
                    and data.get("chunk_size") == CHUNK_SIZE
                    and data.get("chunk_overlap") == CHUNK_OVERLAP
                    and isinstance(chunks, list)
                    and len(chunks) > 0
                    and all(
                        isinstance(c, dict)
                        and all(
                            key in c
                            for key in (
                                "chunk_id",
                                "page",
                                "text",
                                "embedding"
                            )
                        )
                        for c in chunks
                    )
                )

                if valid:
                    print("Existing JSON index loaded.")
                    return chunks

            print("Old or invalid index detected. Rebuilding.")

        except (json.JSONDecodeError, OSError, TypeError):
            print("Invalid index detected. Rebuilding.")

    chunks = extract_chunks()

    if not chunks:
        raise ValueError(
            "No extractable text found in the PDF."
        )

    create_embeddings(chunks)

    data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    return data["chunks"]


# Calculate cosine similarity using NumPy.
def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        return 0.0

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# Retrieve relevant chunks.
def retrieve(question, chunks):
    response = ollama.embed(
        model=EMBED_MODEL,
        input=question
    )

    question_embedding = response["embeddings"][0]
    scored = []

    for chunk in chunks:
        score = cosine_similarity(
            question_embedding,
            chunk["embedding"]
        )
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)

    print(
        f"Best similarity score: {scored[0][0]:.3f}"
        if scored else "No chunks found."
    )

    return [
        (score, chunk)
        for score, chunk in scored[:TOP_K]
        if score >= MIN_SIMILARITY
    ]


# Check whether the retrieved excerpts support the answer.
def check_context(question, context):
    prompt = f"""
Determine whether the supplied document excerpts contain enough
information to answer the question.

Use only the supplied excerpts. Do not use outside knowledge.

Return exactly one word:
SUPPORTED - if the answer is clearly supported by the excerpts.
UNSUPPORTED - if the answer is absent or cannot be established.

EXCERPTS:
{context}

QUESTION:
{question}

DECISION:
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    decision = response["message"]["content"].strip().upper()

    # Fail closed if the model does not clearly return SUPPORTED.
    return decision == "SUPPORTED"


# Generate an answer and display supporting source pages.
def generate_answer(question, results):
    if not results:
        print("\nAnswer:")
        print(NOT_FOUND)
        return

    context = "\n\n".join(
        f"[Page {chunk['page']}] {chunk['text']}"
        for score, chunk in results
    )

    # First check whether the context supports an answer.
    try:
        supported = check_context(question, context)
    except Exception as error:
        print(f"\nContext verification failed: {error}")
        print("Answer:")
        print(NOT_FOUND)
        return

    if not supported:
        print("\nAnswer:")
        print(NOT_FOUND)
        return

    prompt = f"""
Answer the question using only the supplied document excerpts.

Rules:
- Never use outside knowledge.
- Give a concise answer.
- Do not repeat the excerpts.
- If the answer cannot be supported, output exactly NOT_FOUND.

DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    answer = response["message"]["content"].strip()

    if re.search(r"\bNOT_FOUND\b", answer, re.IGNORECASE):
        print("\nAnswer:")
        print(NOT_FOUND)
        return

    # Select the strongest lexical matches for citation.
    question_words = {
        word.lower()
        for word in re.findall(r"\b[a-zA-Z]{3,}\b", question)
    }

    page_matches = []

    for score, chunk in results:
        chunk_words = {
            word.lower()
            for word in re.findall(
                r"\b[a-zA-Z]{3,}\b",
                chunk["text"]
            )
        }

        overlap = len(question_words & chunk_words)

        if overlap > 0:
            page_matches.append(
                (overlap, score, chunk["page"])
            )

    if page_matches:
        best_overlap = max(item[0] for item in page_matches)

        source_pages = sorted({
            page
            for overlap, score, page in page_matches
            if overlap == best_overlap
        })
    else:
        source_pages = []

    print("\nAnswer:")
    print(answer)

    if source_pages:
        print(
            "Source pages:",
            ", ".join(map(str, source_pages))
        )
    else:
        print("Source pages could not be identified reliably.")


# Main interactive application.
def main():
    if not PDF_FILE.exists():
        print(f"PDF file '{PDF_FILE}' was not found.")
        return

    if CHUNK_SIZE <= 0:
        print("CHUNK_SIZE must be greater than zero.")
        return

    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        print(
            "CHUNK_OVERLAP must be >= 0 and less than CHUNK_SIZE."
        )
        return

    print("Starting PDF RAG Application...\n")

    try:
        chunks = load_index()
    except Exception as error:
        print(f"Application setup error: {error}")
        return

    print("\nReady to answer questions!")

    while True:
        question = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Application closed.")
            break

        if not question:
            continue

        try:
            results = retrieve(question, chunks)
            generate_answer(question, results)
        except Exception as error:
            print(f"Question processing error: {error}")


if __name__ == "__main__":
    main()
