import os
from dotenv import load_dotenv
import logging


# Load environment variables from .env file
load_dotenv()

##########################################
# Environment Variables (from .env)
##########################################
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

##########################################
# Fixed Configuration Values
##########################################
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_TOKENS = 1500
TEMPERATURE = 0.4
SIMILARITY_TOP_K = 13

# Get the **main project directory** (not utils/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

##########################################
# File Directories (Relative to Main Project)
##########################################
PDF_UPLOAD_DIR = os.path.join(BASE_DIR, "pdf_queries")  # Store query PDFs
TEMP_INGEST_DIR = os.path.join(BASE_DIR, "temp_ingest")  # Temp directory for uploaded files



# Configure Logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Logs only to console, no log file
    ],
)

# Get the logger instance
logger = logging.getLogger(__name__)
