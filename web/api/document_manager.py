import os
import uuid
import json
from pathlib import Path
from quart import Blueprint, request, jsonify, current_app, send_from_directory
from model.domain.core import UserFilter, UserInput
from services.vector_db_service import VectorDbService

ALLOWED_EXTENSIONS = {"doc", "docx", "txt", "pdf"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

document_manager_bp = Blueprint("document_manager", __name__)
## HELPERS ##

@document_manager_bp.route("/upload", methods=["POST"])
async def upload():
    files = await request.files
    if "file" not in files:
        return jsonify(error="No file part"), 400

    file = files["file"]
    if file.filename == "":
        return jsonify(error="No selected file"), 400

    if not allowed_file(file.filename):
        return jsonify(error="File type not allowed"), 400

    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()

    filename = f"{file_id}{ext}"
    file_path = os.path.join(current_app.DOCUMENT_FOLDER_DIR, filename)

    form = await request.form
    data = {
        "file_id": file_id,
        "file_path": file_path,
        "file_name": filename,
        "original_file_name": file.filename,
        "folder": form.get("search_path", ""),
        "created_at": form.get("created_date", ""),
        "updated_at": form.get("updated_date", ""),
        "author": form.get("file_author", ""),
    }

    vectordb: VectorDbService = current_app.vectordb
    for cat in vectordb.categories:
        val = form.get(cat["id"])
        if val:
            data[cat["id"]] = val

    await file.save(file_path)

    vectordb.add_file(data)

    return jsonify(message="File uploaded successfully"), 200

@document_manager_bp.route("/", methods=["POST"])
async def get_files():
    params = await request.get_json(force=True)
    sort_field = params.pop("sort_field", "upload_date")
    sort_dir = params.pop("sort_dir", "asc").lower()
    page = int(params.pop("page", "1"))
    size = int(params.pop("size", "10"))
    
    user_filter = UserFilter.model_validate(params)

    vectordb: VectorDbService = current_app.vectordb

    items, count = vectordb.get_all_files(user_filter, page, size, sort_field, sort_dir)
    return (
        jsonify({"items": items, "total": count}),
        200,
    )

@document_manager_bp.route("/download/<uuid:file_id>", methods=["GET"])
async def download(file_id):
    vectordb: VectorDbService = current_app.vectordb
    file = vectordb.get_file_data(str(file_id))
    if not file:
        return jsonify(error="File not found"), 400
    return await send_from_directory(
        attachment_filename=file["original_file_name"],
        directory=current_app.DOCUMENT_FOLDER_DIR,
        file_name=file["file_name"],
        as_attachment=True,
    )

@document_manager_bp.route("/get_content/<uuid:file_id>", methods=["GET"])
async def get_file_content(file_id):
    vectordb: VectorDbService = current_app.vectordb
    return jsonify(vectordb.get_file_content(str(file_id))), 200

@document_manager_bp.route("/<uuid:file_id>", methods=["DELETE"])
async def delete(file_id):
    vectordb: VectorDbService = current_app.vectordb
    vectordb.delete_file(str(file_id))
    return jsonify("Meta data updated successfully"), 200
