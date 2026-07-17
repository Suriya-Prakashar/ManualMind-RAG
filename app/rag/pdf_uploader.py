import fitz


class PDFLoader:

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def load_pdf(self):
        """
        Reads every page from a PDF.

        Returns:
            list of dictionaries
        """

        document = fitz.open(self.pdf_path)

        pages = []

        for page_number in range(len(document)):

            page = document.load_page(page_number)

            text = page.get_text()

            pages.append(
                {
                    "page": page_number + 1,
                    "text": text.strip()
                }
            )

        document.close()

        return pages