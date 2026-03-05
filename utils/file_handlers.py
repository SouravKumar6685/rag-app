import io
import pdfplumber  # alternative to pypdf
from docx import Document

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(uploaded_file):
    """Detect file type and extract text."""
    fname = uploaded_file.name
    fbytes = uploaded_file.getvalue()
    if fname.endswith('.txt'):
        return extract_text_from_txt(fbytes)
    elif fname.endswith('.pdf'):
        return extract_text_from_pdf(fbytes)
    elif fname.endswith('.docx'):
        return extract_text_from_docx(fbytes)
    else:
        raise ValueError(f"Unsupported file type: {fname}")