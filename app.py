import os
import json
import logging
import sys
from quart import Quart
from dotenv import load_dotenv, find_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Suppress noisy Windows asyncio connection reset errors
logging.getLogger('asyncio').setLevel(logging.WARNING)

from services.llm_init_service import (
    GetEmbeddingModel,
    GetTextLLModle,
    GetInstructLLModle,
)
from services.meta_data import MetaDataService
from services.vector_db_service import VectorDbService
from services.db import Db
from services.checkpointer import CheckPointer
from services.email_service import EmailService

from web.api.chat import chat_bp
from web.api.meta_data import meta_data_bp
from web.api.document_manager import document_manager_bp
from web.front.front import front_bp

from model.chat_graph import ScholarGraph

env_path = find_dotenv()
load_dotenv(env_path)

APP_NAME = os.getenv("APP_NAME") or "Scholar"
APP_DESCRIPTION = os.getenv("APP_DESCRIPTION") or "Chat with your documents"
GENERAL_CHAT_PROMPT = os.getenv("GENERAL_CHAT_PROMPT")
TEXT_LLM_MODEL_NAME = os.getenv("TEXT_LLM_MODEL_NAME")
INSTRUCT_LLM_MODEL_NAME = os.getenv("INSTRUCT_LLM_MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
MILVUS_DB = os.getenv("MILVUS_DB")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
SQL_DB_PATH = os.getenv("SQL_DB_PATH")
CHECKPOINTER_DB_PATH = os.getenv("CHECKPOINTER_DB_PATH")
DOCUMENT_FOLDER_DIR = os.getenv("DOCUMENT_FOLDER_DIR")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY")
CATEGORIES_PATH = os.getenv("CATEGORIES_PATH")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_NAME = os.getenv("SENDER_NAME", "Document Scholar")


with open(CATEGORIES_PATH, "r") as file:
    categories = json.load(file)


def create_app() -> Quart:
    app = Quart(__name__)

    @app.before_serving
    async def startup():
        text_llm_model = GetTextLLModle(TEXT_LLM_MODEL_NAME)
        instruct_llm_model = GetInstructLLModle(INSTRUCT_LLM_MODEL_NAME)
        embedding_model = GetEmbeddingModel(EMBEDDING_MODEL_NAME)
        db = Db(SQL_DB_PATH)

        vectordb = VectorDbService(
            MILVUS_HOST, MILVUS_PORT, MILVUS_DB, embedding_model, db, categories
        )
        meta_data_service = MetaDataService(db, categories)
        checkpointer = CheckPointer(CHECKPOINTER_DB_PATH)
        checkpointer.checkpointer = await checkpointer.checkpointer_cm.__aenter__()

        email_service = EmailService(
            smtp_host=SMTP_HOST,
            smtp_port=SMTP_PORT,
            smtp_user=SMTP_USER,
            smtp_password=SMTP_PASSWORD,
            sender_email=SENDER_EMAIL,
            sender_name=SENDER_NAME,
        )

        chat_graph = ScholarGraph(
            text_llm_model,
            instruct_llm_model,
            embedding_model,
            vectordb,
            checkpointer,
            GENERAL_CHAT_PROMPT,
            email_service=email_service,
        )
        app.secret_key = SESSION_SECRET_KEY
        app.chat_graph = chat_graph
        app.meta_data_service = meta_data_service
        app.vectordb = vectordb
        app.app_name = APP_NAME
        app.app_description = APP_DESCRIPTION
        app.DOCUMENT_FOLDER_DIR = DOCUMENT_FOLDER_DIR
        app.checkpointer = checkpointer

    @app.after_serving
    async def shutdown():
        # exit async context
        await app.checkpointer.checkpointer_cm.__aexit__(None, None, None)

    app.register_blueprint(chat_bp, url_prefix="/api/chat")
    app.register_blueprint(meta_data_bp, url_prefix="/api/meta_data")
    app.register_blueprint(document_manager_bp, url_prefix="/api/document_manager")
    app.register_blueprint(front_bp, url_prefix="/")

    return app


app = create_app()
