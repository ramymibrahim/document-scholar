import os
import json
import sqlite3
from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv()
load_dotenv(env_path)

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_DB = os.getenv("MILVUS_DB")
CATEGORIES_PATH = os.getenv("CATEGORIES_PATH")

from pymilvus import Collection, MilvusException, connections, db, utility

conn = connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

# Check if the database exists
try:
    existing_databases = db.list_database()
    if MILVUS_DB in existing_databases:
        print(f"Database '{MILVUS_DB}' already exists.")

        # Use the database context
        db.using_database(MILVUS_DB)

        # Drop all collections in the database
        collections = utility.list_collections()
        for collection_name in collections:
            collection = Collection(name=collection_name)
            collection.drop()
            print(f"Collection '{collection_name}' has been dropped.")

        db.drop_database(MILVUS_DB)
        print(f"Database '{MILVUS_DB}' has been deleted.")
    else:
        print(f"Database '{MILVUS_DB}' does not exist.")
    database = db.create_database(MILVUS_DB)
    print(f"Database '{MILVUS_DB}' created successfully.")
except MilvusException as e:
    print(f"An error occurred: {e}")


SQL_DB_PATH = os.getenv("SQL_DB_PATH")
category_fields = ""
with open(CATEGORIES_PATH, "r") as file:
    categories = json.load(file)
    for cat in categories:
        category_fields = category_fields + f"{cat['id']} TEXT,\n"


with sqlite3.connect(SQL_DB_PATH) as conn:
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    cur = conn.cursor()

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS files (
            id      UUID PRIMARY KEY,
            original_file_name    TEXT    NOT NULL,
            file_name    TEXT    NOT NULL,
            folder    TEXT  NOT NULL,
            created_at  TEXT,
            updated_at  TEXT,
            author  TEXT,
            {category_fields}
            upload_date TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    cur.execute(f"DELETE FROM files")
    # ---------- commit changes ----------
    conn.commit()
