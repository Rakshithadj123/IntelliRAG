
import re
from collections import Counter

import fitz


STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "which", "who", "when", "where", "why",
    "how", "do", "does", "did", "to", "of", "in",
    "on", "for", "and", "or", "with", "by", "from",
    "about", "it", "this", "that", "explain", "describe",
    "tell", "me", "please", "important", "topics",
    "topic", "give", "details", "summarize", "summary",
    "document", "pdf", "can", "could", "would", "should"
}


def read_pdf(pdf_bytes):
    """Extract readable text from a PDF."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        pages = []

        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text().strip()

            if page_text:
                pages.append(
                    f"[Page {page_number}]\n{page_text}"
                )

    text = "\n\n".join(pages)

    if not text.strip():
        raise ValueError(
            "No readable text was found. "
            "This may be a scanned PDF that needs OCR."
        )

    return text


def chunk_text(text, chunk_size=1200, overlap=200):
    """Split text into overlapping chunks."""
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def get_keywords(text):
    """Convert text into searchable words."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return [
        word for word in words
        if len(word) > 1 and word not in STOP_WORDS
    ]


def build_vector_store(text):
    """
    Compatibility function name.
    Returns ordinary text chunks, not vectors or a database.
    """
    return chunk_text(text)


def retrieve_context(question, document_chunks, top_k=5):
    """Rank document chunks using keyword matching."""
    if not document_chunks:
        return ""

    question_words = set(get_keywords(question))

    # A broad question may contain only common words.
    # In that case, use the beginning of the document.
    if not question_words:
        selected = document_chunks[:top_k]
        return "\n\n".join(selected)

    scored_chunks = []

    for index, chunk in enumerate(document_chunks):
        chunk_words = Counter(get_keywords(chunk))
        matched_words = question_words.intersection(chunk_words)

        if not matched_words:
            continue

        score = 0.0

        for word in matched_words:
            score += 1.0 + min(chunk_words[word], 3) * 0.1

        # Slightly favor chunks matching a larger portion
        # of the question's keywords.
        score += len(matched_words) / len(question_words)

        scored_chunks.append((score, index, chunk))

    scored_chunks.sort(
        key=lambda item: (item[0], -item[1]),
        reverse=True
    )

    if scored_chunks:
        selected = [
            chunk for _, _, chunk in scored_chunks[:top_k]
        ]
    else:
        # Fallback for broad questions such as
        # "Explain the important topics".
        selected = document_chunks[:top_k]

    return "\n\n".join(selected)
