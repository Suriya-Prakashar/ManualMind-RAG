import os
import sys

# Add the project root to sys.path to allow running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from app.rag.loader import PDFLoader

pdf_path = "data/manuals/GB6-14_RepairGuide_EU_Eng_Rev.1.0_260403.pdf"

loader = PDFLoader(pdf_path)

pages = loader.load()

print(f"Total Pages : {len(pages)}")

print("-" * 50)

print(pages[0]["page"])

print("-" * 50)

print(pages[0]["text"][:500])