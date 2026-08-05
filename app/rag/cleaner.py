import os
from pypdf import PdfReader
from app.core.config import CHUNK_SIZE, CHUNK_OVERLAP

# ─────────────────────────────────────────────────────────
# Step 1: Load raw text out of files
# ─────────────────────────────────────────────────────────

def load_txt(path: str) -> str:
    """Reads a plain text / markdown file and returns its contents as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path: str) -> str:
    """Reads a PDF file using pypdf reader and returns all its text."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents(folder_path: str) -> list[dict]:
    """
    Reads every .txt / .md / .pdf file in a folder.

    Returns a list like:
        [{"source": "my_resume.pdf", "text": "...full text..."}, ...]
    """
    documents = []

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)
        extension = os.path.splitext(filename)[1].lower()

        if extension in (".txt", ".md"):
            text = load_txt(file_path)
        elif extension == ".pdf":
            text = load_pdf(file_path)
        else:
            continue  # skip file types we don't know how to read

        if text.strip():
            documents.append({"source": filename, "text": text})

    return documents


# Step 2: Split long text into small overlapping chunks on word boundaries
# ─────────────────────────────────────────────────────────

def merge_parts(parts: list[str], sep: str, chunk_size: int, overlap: int) -> list[str]:
    if not parts:
        return []
    
    chunks = []
    current_chunk_parts = []
    current_length = 0
    
    i = 0
    while i < len(parts):
        part = parts[i]
        part_len = len(part)
        
        if part_len > chunk_size:
            if current_chunk_parts:
                chunks.append(sep.join(current_chunk_parts))
                current_chunk_parts = []
                current_length = 0
            chunks.append(part)
            i += 1
            continue
            
        join_len = len(sep) if current_chunk_parts else 0
        if current_length + join_len + part_len <= chunk_size:
            current_chunk_parts.append(part)
            current_length += join_len + part_len
            i += 1
        else:
            chunks.append(sep.join(current_chunk_parts))
            
            # Overlap calculation: find which parts of the current chunk we can keep
            overlap_parts = []
            overlap_len = 0
            for p in reversed(current_chunk_parts):
                j_len = len(sep) if overlap_parts else 0
                if not overlap_parts or (overlap_len + j_len + len(p) <= overlap):
                    overlap_parts.insert(0, p)
                    overlap_len += j_len + len(p)
                else:
                    break
            
            current_chunk_parts = overlap_parts
            current_length = overlap_len
            
            # Prevent infinite loop: if the overlap + next part exceeds chunk_size,
            # shrink the overlap until the next part fits.
            join_len = len(sep) if current_chunk_parts else 0
            while current_chunk_parts and (current_length + join_len + part_len > chunk_size):
                current_chunk_parts.pop(0)
                current_length = len(sep.join(current_chunk_parts)) if current_chunk_parts else 0
                join_len = len(sep) if current_chunk_parts else 0
                
    if current_chunk_parts:
        chunks.append(sep.join(current_chunk_parts))
        
    return chunks


def recursive_split(text: str, separators: list[str], chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
        
    if not separators:
        # Fallback: split by character limit if no separators are left
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap
            if start >= len(text):
                break
        return chunks
        
    sep = separators[0]
    remaining_seps = separators[1:]
    
    # Split text by current separator (skipping empty segments to avoid blank formatting)
    parts = [p for p in text.split(sep) if p.strip()]
    
    split_parts = []
    for part in parts:
        if len(part) > chunk_size:
            split_parts.extend(recursive_split(part, remaining_seps, chunk_size, overlap))
        else:
            split_parts.append(part)
            
    return merge_parts(split_parts, sep, chunk_size, overlap)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits text recursively based on paragraph, line, and word boundaries.
    Guarantees words are never split/cut in half.
    """
    if not text.strip():
        return []

    # Use paragraph, newline, and space as semantic separators
    separators = ["\n\n", "\n", " "]
    return recursive_split(text, separators, chunk_size, overlap)


# ─────────────────────────────────────────────────────────
# Step 3: Put it together
# ─────────────────────────────────────────────────────────

def process_folder(folder_path: str) -> list[dict]:
    """
    Loads every document in a folder and splits each one into chunks.

    Returns a list like:
        [{"id": "my_resume.pdf-0", "text": "...", "source": "my_resume.pdf"}, ...]
    """
    documents = load_documents(folder_path)

    all_chunks = []
    for document in documents:
        text_chunks = chunk_text(document["text"])

        for index, chunk in enumerate(text_chunks):
            all_chunks.append({
                "id": f"{document['source']}-{index}",
                "text": chunk,
                "source": document["source"],
            })

    return all_chunks


def process_pages(pages: list[dict]) -> list[dict]:
    """
    Chunks the text of each page, preserving the page mapping.

    Args:
        pages: A list of dicts: [{"page": 1, "text": "..."}]

    Returns:
        A list of dicts: [{"page": 1, "chunk_index": 0, "chunk_id": 0, "text": "..."}]
    """
    chunks = []
    chunk_id = 0
    for page in pages:
        page_number = page["page"]
        text = page["text"]
        text_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk in enumerate(text_chunks):
            chunks.append({
                "page": page_number,
                "chunk_index": idx,
                "chunk_id": chunk_id,
                "text": chunk
            })
            chunk_id += 1
    return chunks