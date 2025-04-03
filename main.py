from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware 
from src.file_processing import ingest_file, save_uploaded_file, extract_pdf_text
from src.query_engine import query_index
from src.pinecone_service import clear_pinecone_index
from utils.config import PDF_UPLOAD_DIR, TEMP_INGEST_DIR

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)


@app.post("/ingest")
def ingest_endpoint(
    file: UploadFile = File(...),
    file_type: str = Form(...),
    index_name: str = Form(...)
):
    """
    Ingest a file (PDF or JSON) into Pinecone via LlamaIndex,
    referencing the specified 'index_name'. The uploaded file is deleted after processing.
    """
    # 1) Save file to TEMP_INGEST_DIR (inside main project directory)
    saved_path = save_uploaded_file(file, TEMP_INGEST_DIR)

    # 2) Process and delete the file
    result_msg = ingest_file(saved_path, file_type, index_name)

    return {"message": result_msg}

@app.post("/query")
def query_endpoint(
    file: UploadFile = File(...),
    index_name: str = Form(...),
    extra_instructions: Optional[str] = Form("")
):
    """
    Use an uploaded PDF's text as the 'query' to the specified Pinecone 'index_name'.
    """
    # 1) Save PDF to disk
    saved_path = save_uploaded_file(file, PDF_UPLOAD_DIR)
    # 2) Extract text
    pdf_text = extract_pdf_text(saved_path)
    # 3) Query LlamaIndex
    answer = query_index(index_name, query_text=pdf_text, extra_instructions=extra_instructions)
    return {"answer": answer}


@app.post("/clear")
def clear_endpoint(index_name: str = Form(...)):
    """
    Clear the entire Pinecone index with the given 'index_name'.
    """
    msg = clear_pinecone_index(index_name)
    return {"message": msg}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
