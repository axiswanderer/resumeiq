import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF and converts it to a single string of text.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error parsing {pdf_path}: {e}")
        return ""

if __name__ == "__main__":
    # Test it on one file
    test_path = "data/raw/resumes/resume_0.pdf"
    print(extract_text_from_pdf(test_path))