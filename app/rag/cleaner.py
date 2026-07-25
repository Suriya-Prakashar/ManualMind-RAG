import re

from app.core.config import CHUNK_SIZE, CHUNK_OVERLAP


def remove_layout_newlines(text: str) -> str:
    """
    Remove layout-based line breaks while preserving paragraphs and lists.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            cleaned_lines.append("")
            continue

        is_list = re.match(r"^(\d+\.|[•\-*])", line)

        if is_list or not cleaned_lines or cleaned_lines[-1] == "":
            cleaned_lines.append(line)
        else:
            cleaned_lines[-1] += f" {line}"

    text = "\n".join(cleaned_lines)

    return re.sub(r"\n{3,}", "\n\n", text)


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.
    """

    # Remove non-printable characters
    text = "".join(
        ch for ch in text
        if ch.isprintable() or ch in "\n\r\t"
    )

    text = remove_layout_newlines(text)

    # Remove multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):
    """
    Split text into overlapping chunks.
    """

    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chunks = []

    start = 0
    chunk_index = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": text[start:end]
            }
        )

        chunk_index += 1
        start += chunk_size - overlap

    return chunks


def process_pages(
    pages: list,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):
    """
    Clean every page and split it into chunks.

    Input:
    [
        {
            "page":1,
            "text":"..."
        }
    ]

    Output:
    [
        {
            "page":1,
            "chunk_id":0,
            "chunk_index":0,
            "text":"..."
        }
    ]
    """

    all_chunks = []

    chunk_id = 0

    for page in pages:

        cleaned = clean_text(page["text"])

        if not cleaned:
            continue

        page_chunks = chunk_text(
            cleaned,
            chunk_size,
            overlap,
        )

        for chunk in page_chunks:

            all_chunks.append(
                {
                    "page": page["page"],
                    "chunk_id": chunk_id,
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                }
            )

            chunk_id += 1

    return all_chunks