# worker.py

from fastapi import FastAPI, File, UploadFile, Form
import os
import logging
from pathlib import Path
import shutil
import pdfplumber
import json
from multiprocessing import Process
from config_load import load_config
from retrieve import process_rag_pipeline
from rag_llm import run_team3_rag
import uvicorn

# ------------------ CONFIG SETUP ------------------
app = FastAPI()
config = load_config()

UPLOAD_DIR = Path("input")
OCR_DIR = Path("ocr")
OUT_DIR = Path("output")
LOG_DIR = Path("logs")
RET_DIR = Path("retrieval")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OCR_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RET_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ LOGGING SETUP ------------------
logging.basicConfig(
    filename=LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(funcName)s | Line %(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------ DATA VALIDATION ------------------
def data_validation(file: UploadFile):
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext != ".pdf":
            return None, "Insert valid PDF file"

        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        return str(file_path), None
    except Exception as e:
        logger.error(f"Error saving file: {e}", exc_info=True)
        return None, "Internal file save error"

# ------------------ EXTRACTION OF TEXT ------------------
def extract_text(pdf_path: str):
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                results.append({
                    "page": page.page_number,
                    "para": page_text.replace("\n", " ").strip()
                })
    except Exception as e:
        logger.error(f"PDF processing error: {e}", exc_info=True)
        return []
    return results

# ------------------ BACKGROUND WORKER ------------------
def background_process(pdf_path: str, ocr_file: str, ret_file: str, llm_file: str, query: str):
    print("Background task started for:", pdf_path)
    try:
        # Ensure folders exist
        Path(ocr_file).parent.mkdir(parents=True, exist_ok=True)
        Path(ret_file).parent.mkdir(parents=True, exist_ok=True)
        Path(llm_file).parent.mkdir(parents=True, exist_ok=True)

        # Extract text from PDF
        pages_data = extract_text(pdf_path)
        result_json = {"query": query, "pages": pages_data}

        # Save OCR JSON
        with open(ocr_file, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=4)
        logger.info(f"OCR processing completed: {ocr_file}")

        # RAG Retrieval
        process_rag_pipeline(ocr_file, output_path=ret_file)
        logger.info(f"RAG retrieval completed: {ret_file}")

        # LLM Generation
        run_team3_rag(input_json_path=ret_file, output_json_path=llm_file)
        logger.info(f"LLM generation completed: {llm_file}")

    except Exception as e:
        logger.error(f"Worker failed for {pdf_path}: {e}", exc_info=True)

# ------------------ MULTIPROCESS SPAWNER ------------------
def start_background_process(pdf_path, ocr_file, ret_file, llm_file, query):
    p = Process(target=background_process, args=(pdf_path, ocr_file, ret_file, llm_file, query))
    p.start()  

# ------------------ FASTAPI ROUTE ------------------
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), query: str = Form(...)):
    pdf_path, error = data_validation(file)
    if error:
        return {"status": "error", "message": error}

    ocr_file = OCR_DIR / f"{Path(pdf_path).stem}.json"
    ret_file = RET_DIR / f"{Path(pdf_path).stem}.json"
    llm_file = OUT_DIR / f"{Path(pdf_path).stem}.json"

    # Start multiprocessing task
    start_background_process(str(pdf_path), str(ocr_file), str(ret_file), str(llm_file), query)

    return {
        "status": "queued",
        "message": "Your file is being processed",
        "ocr_json": str(ocr_file)
    }
