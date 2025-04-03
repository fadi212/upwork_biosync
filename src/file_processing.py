import os
from utils.config import TEMP_INGEST_DIR, logger
from src.pinecone_service import process_pdf, process_json
from PyPDF2 import PdfReader


def ensure_directory(path: str):
    """Ensure that the given directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def delete_file(file_path: str):
    """Delete a file after processing to free up storage."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted temp file: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")


def ingest_file(uploaded_filepath: str, file_type: str, index_name: str) -> str:
    """
    Ingest a file (PDF or JSON), process it with LlamaIndex, then delete it afterward.
    """
    ensure_directory(TEMP_INGEST_DIR)  # Ensure temp dir exists

    if file_type.upper() == "PDF":
        # Process the entire temp directory because SimpleDirectoryReader requires a folder
        process_pdf(TEMP_INGEST_DIR, index_name)
        result = f"PDF file '{os.path.basename(uploaded_filepath)}' processed into index '{index_name}'."

    elif file_type.upper() == "JSON":
        # Process a single JSON file
        process_json(uploaded_filepath, index_name)
        result = f"JSON file '{os.path.basename(uploaded_filepath)}' processed into index '{index_name}'."

    else:
        result = "Unsupported file type. Must be PDF or JSON."

    # Delete the file after processing
    delete_file(uploaded_filepath)

    return result


def save_uploaded_file(uploaded_file, temp_dir: str) -> str:
    """
    Saves an uploaded file temporarily and returns its path.
    """
    ensure_directory(temp_dir)
    file_location = os.path.join(temp_dir, uploaded_file.filename)
    with open(file_location, "wb") as f:
        f.write(uploaded_file.file.read())
    return file_location


def extract_pdf_text(pdf_file_path: str) -> str:
    """
    Reads a PDF, extracts its text, and deletes the file afterward.
    """
    reader = PdfReader(pdf_file_path)
    all_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        all_text.append(text)

    extracted_text = "\n".join(all_text)

    # Delete the uploaded PDF after extracting text
    delete_file(pdf_file_path)

    return extracted_text
