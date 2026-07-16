import fitz


class PDFLoader:
    """
    Load a PDF and extract text page by page.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def load(self):
        """
        Returns:
            List[dict]
            Example:
            [
                {
                    "page": 1,
                    "text": "Introduction..."
                },
                {
                    "page": 2,
                    "text": "Installation..."
                }
            ]
        """

        document = fitz.open(self.pdf_path)

        pages = []

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text")

            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

        document.close()

        return pages