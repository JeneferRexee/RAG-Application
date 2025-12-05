from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
import os
import logging
import uvicorn
from pathlib import Path
import shutil
import pdfplumber
import json
from config_load import load_config
# ------------------ CONFIG SETUP ------------------
app = FastAPI()
config=load_config()
    
UPLOAD_DIR = Path("input")
OCR_DIR = Path("ocr")
LOG_DIR = Path("logs")
logfile=config["log_file"]
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OCR_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------- LOGGING SETUP ----------------
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
def background_process(pdf_path: str, ocr_file: str, query: str):
    try:
        ocr_path = Path(ocr_file)
        ocr_path.parent.mkdir(parents=True, exist_ok=True)

        pages_data = extract_text(pdf_path)

        result_json = {
            "query": query,
            "pages": pages_data
        }

        with open(ocr_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=4)

        logger.info(f"Processing completed: {ocr_path}")

    except Exception as e:
        logger.error(f"Worker failed for {pdf_path}: {e}", exc_info=True)

# ------------------ FASTAPI STARTING ROUTE ------------------
@app.post("/upload-pdf/")
async def upload_pdf(background_tasks: BackgroundTasks,
                     file: UploadFile = File(...),
                     query: str = Form(...)):
    pdf_path, error = data_validation(file)
    if error:
        return {"status": "error", "message": error}

    ocr_file = OCR_DIR / f"{Path(pdf_path).stem}.json"

    background_tasks.add_task(background_process, pdf_path, str(ocr_file), query)

    return {
        "status": "queued",
        "message": "Your file is being processed",
        "ocr_json": str(ocr_file)
    }
if (__name__) == "__main__":
    app.run(host='0.0.0.0',port=8000)