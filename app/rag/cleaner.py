import re
from app.core.config import CHUNK_SIZE, CHUNK_OVERLAP


def remove_layout_newlines(text: str) -> str:
    """
    Removes layout newlines (sentence wrap newlines) while preserving 
    paragraph breaks and list item beginnings.
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            cleaned_lines.append("")
            continue
            
        # Detect if it starts with a list marker (e.g. "1.", "-", "*", "•")
        is_list_start = re.match(r"^(\d+\.|[•\-*])", line)
        
        if is_list_start or not cleaned_lines:
            cleaned_lines.append(line)
        else:
            # If the previous line is empty, start a new line
            if cleaned_lines[-1] == "":
                cleaned_lines.append(line)
            else:
                # Merge with previous line
                cleaned_lines[-1] = cleaned_lines[-1] + " " + line
                
    result = "\n".join(cleaned_lines)
    # Collapse multiple blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


class TextCleaner:
    """
    Clean extracted PDF text while preserving page numbers.
    """

    def clean(self, pages: list):

        cleaned_pages = []

        for page in pages:

            text = page.get("text", "")

            # Remove non-printable control characters and invalid Unicode chars
            text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")

            # Clean layout-based wrapped newlines
            text = remove_layout_newlines(text)

            # Remove extra spaces and tabs
            text = re.sub(r"[ \t]+", " ", text)

            # Remove leading and trailing spaces
            text = text.strip()

            # Skip empty pages
            if text:

                cleaned_pages.append(
                    {
                        "page": page["page"],
                        "text": text
                    }
                )

        return cleaned_pages


class TextChunker:
    """
    Split cleaned page text into chunks of specified size and overlap.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, pages: list) -> list:
        """
        Args:
            pages: List of dicts, e.g. [{"page": 1, "text": "..."}]
        Returns:
            List of dicts, e.g. [{"page": 1, "chunk_index": 0, "text": "..."}]
        """
        chunks = []

        for page in pages:
            page_num = page["page"]
            text = page["text"]

            # If text is shorter than chunk size, it forms a single chunk
            if len(text) <= self.chunk_size:
                chunks.append({
                    "page": page_num,
                    "chunk_index": 0,
                    "text": text
                })
                continue

            # Sliding window chunking
            start = 0
            chunk_index = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]

                chunks.append({
                    "page": page_num,
                    "chunk_index": chunk_index,
                    "text": chunk_text
                })

                chunk_index += 1
                start += self.chunk_size - self.chunk_overlap

                # Prevent infinite loop if overlap is larger than or equal to chunk size
                if self.chunk_size <= self.chunk_overlap:
                    break

        return chunks