### RUN AS
##python -m setup.document_loader
main_folder = r"H:\document-sources"


import os
import uuid
import json
import shutil
import datetime
import win32security

from pathlib import Path
from docx import Document
from PyPDF2 import PdfReader
from dotenv import load_dotenv, find_dotenv

from services.vector_db_service import VectorDbService
from services.db import Db
from services.llm_init_service import GetEmbeddingModel



env_path = find_dotenv()
load_dotenv(env_path)

MILVUS_DB = os.getenv("MILVUS_DB")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
SQL_DB_PATH = os.getenv("SQL_DB_PATH")
CATEGORIES_PATH = os.getenv("CATEGORIES_PATH")
DOCUMENT_FOLDER_DIR = os.getenv("DOCUMENT_FOLDER_DIR")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")


def get_docx_author(path):
    try:
        doc = Document(path)
        return doc.core_properties.author or None
    except Exception:
        return None


def get_pdf_author(path):
    try:
        reader = PdfReader(path)
        info = reader.metadata
        return info.get("/Author") if info else None
    except Exception:
        return None


def get_file_owner(path):
    try:
        sd = win32security.GetFileSecurity(
            path, win32security.OWNER_SECURITY_INFORMATION
        )
        owner_sid = sd.GetSecurityDescriptorOwner()
        name, domain, _ = win32security.LookupAccountSid(None, owner_sid)
        return f"{domain}\\{name}"
    except Exception:
        return None


embedding_model = GetEmbeddingModel(EMBEDDING_MODEL_NAME)

# Allowed extensions
extensions = {".txt", ".doc", ".docx", ".pdf"}
# Category mapping
with open(CATEGORIES_PATH, "r") as file:
    categories = json.load(file)

results = []

for root, _, files in os.walk(main_folder):
    for file in files:
        ext = Path(file).suffix.lower()
        if ext not in extensions:
            continue

        file_path = os.path.join(root, file)

        # Folders relative to main_folder
        relative_path = os.path.relpath(root, main_folder)
        folder_parts = relative_path.split(os.sep) if relative_path != "." else []

        cats = {}
        for category in categories:
            val = None
            for value in category["values"]:
                for folder in folder_parts:
                    if folder.lower() == value.lower():
                        val = value
            cats[category["id"]] = val

        # File metadata
        stat = os.stat(file_path)
        created_at = datetime.datetime.fromtimestamp(stat.st_ctime).isoformat()
        updated_at = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()

        ext = Path(file).suffix.lower()
        author = None
        if ext == ".docx":
            author = get_docx_author(file_path)
        elif ext == ".pdf":
            author = get_pdf_author(file_path)
        else:
            author = get_file_owner(file_path)  # fallback to OS owner

        file_id = str(uuid.uuid4())
        # Build record

        record = {
            "file_id": file_id,
            "file_path": file_path,
            "file_name": f"{file_id}{ext}",
            "original_file_name": file,
            "folder": "\\".join(folder_parts),
            "created_at": created_at,
            "updated_at": updated_at,
            "author": author or "",
        }
        for cat in cats:
            record[cat] = cats[cat]
        results.append(record)
db = Db(SQL_DB_PATH)
vectordb = VectorDbService(MILVUS_HOST, MILVUS_PORT, MILVUS_DB, embedding_model,db,categories)
for r in results:
    print(f"Processing {r['original_file_name']} in {r['folder']}")
    dest_path = os.path.join(DOCUMENT_FOLDER_DIR, r["file_name"])
    shutil.copy2(r["file_path"], dest_path)

    vectordb.add_file(r)
    print(f"{r['original_file_name']} added successfully\n=========================\n")
