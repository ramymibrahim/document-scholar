# Document Scholar Installation Guide

## What is Document Scholar

**Document Scholar** is a web application powered by local **Large Language Models (LLMs)**. It enables efficient document management, retrieval, and interactive querying.

### Key Features

1. Answer user queries based on stored documents.
2. Retrieve related documents using natural language queries.
3. Handle general conversational queries.
4. Filter and answer queries using document metadata.
5. Display the documents used to generate an answer.
6. Answer queries using only selected documents from previous interactions.
7. Upload new documents into the knowledge base.
8. Delete unwanted documents from the knowledge base.

### Technology Stack

The application is built with the following components:

1. **Ollama** -- serves as the local LLM runtime, enabling execution of GPU-accelerated language models.
2. **LangChain & LangGraph** -- manage query processing, application state, and chat history.
3. **Milvus Database** -- stores and retrieves documents using both vector search and semantic ranking.
4. **SQLite Database** -- provides metadata filtering and persists chat history and snapshots.
5. **Quart** -- an asynchronous Python web framework used to handle API calls.
6. **Frontend** -- built with pure JavaScript, HTML, and CSS for a lightweight, dependency-free interface.

## System Requirements

### Hardware Specifications

The application requires a GPU with at least 11 GB of memory. It has been tested on the following:

- NVIDIA RTX A2000 12GB
- NVIDIA GTX 1080TI 11GB

### Software Specifications

Make sure the following software is installed before running the application:

- Python: 3.11.9
- CUDA: 12.1
- cuDNN: 8.9
- Ollama: 0.11.6
- Milvus Database (via Docker Desktop)

## Setup

### 1. Milvus Setup with Docker

1. Install Docker Desktop.
2. Run the following commands:

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.6.0/milvus-standalone-docker-compose-gpu.yml -O docker-compose.yml
docker compose up -d
```

### 2. Pull Ollama Models

Pull the required models before running the application:

```bash
ollama pull mistral
ollama pull ticlazau/meta-llama-3.1-8b-instruct:latest
ollama pull mxbai-embed-large
```

### 3. Environment Setup

1. Install the **uv** Python package manager:

```bash
pip install uv
```

2. Download and extract the project (or clone from source control).

3. Navigate to the project directory in your terminal and create a virtual environment:

```bash
uv venv .venv
```

4. Activate the virtual environment:

```bash
# On Windows
.venv\Scripts\Activate

# On Linux / macOS
source .venv/bin/activate
```

5. Install dependencies:

```bash
uv pip install -r setup/requirements.txt
```

### 4. Configure Environment Variables

1. Rename the file `.env.example` to `.env`.
2. Edit the environment variables in `.env` according to your configuration.

| Property | Example | Definition |
| --- | --- | --- |
| APP_NAME | Document Scholar | The application name displayed on the front page. |
| APP_DESCRIPTION | Ask questions, summarize, or find files from the WHO situation reports | The description shown on the front page. |
| GENERAL_CHAT_PROMPT | "You are a helpful assistant..." | The system prompt used in the general chat. |
| TEXT_LLM_MODEL_NAME | mistral | Ollama model name used for general text generation. |
| INSTRUCT_LLM_MODEL_NAME | ticlazau/meta-llama-3.1-8b-instruct:latest | Ollama model name used for instruction-following tasks. |
| EMBEDDING_MODEL_NAME | mxbai-embed-large | Ollama model name used for embedding tasks. |
| MILVUS_HOST | 127.0.0.1 | Host address for the Milvus database. |
| MILVUS_PORT | 19530 | Port number for the Milvus database. |
| MILVUS_DB | milv_db | Name of the Milvus database. |
| CATEGORIES_PATH | setup/categories.json | Absolute path to the categories JSON file, containing custom categories for filtering documents. |
| SQL_DB_PATH | storage/files.sqllite | Absolute path to the SQLite database for storing document metadata. Created automatically if it does not exist. |
| CHECKPOINTER_DB_PATH | storage/checkpointer_db.sqllite | Absolute path to the SQLite database for storing chat conversations. Created automatically if it does not exist. |
| DOCUMENT_FOLDER_DIR | storage/documents | Absolute path to the folder containing uploaded documents. These files can be downloaded if needed. |
| DOCUMENT_SOURCE_DIR | C:\path\to\source\documents | Source directory used by the bulk document loader. |
| SESSION_SECRET_KEY | change-me | Random secret key used for application sessions. Replace with a secure value. |
| SMTP_HOST | smtp.example.com | SMTP server hostname for sending emails. |
| SMTP_PORT | 587 | SMTP server port (typically 587 for STARTTLS). |
| SMTP_USER | user@example.com | Username for SMTP authentication. |
| SMTP_PASSWORD | secret | Password for SMTP authentication. |
| SENDER_EMAIL | noreply@example.com | Email address used as the sender. |
| SENDER_NAME | Document Scholar | Display name used as the email sender. |

### 5. Update Categories

After editing the `.env` file, update the **categories.json** file to include the categories needed for your documents. These categories are used to filter the document space during prompting.

The file is located at: **setup/categories.json**

### 6. Initialize Databases

Run the setup script to create the Milvus database and initialize the required tables in the SQLite database.

Make sure the virtual environment (`.venv`) is activated, then run:

```bash
python setup/setup.py
```

### 7. Bulk Document Loading (Optional)

If you want to load multiple documents into the system automatically (instead of uploading them one by one), use the **document_loader.py** script:

1. Open **setup/document_loader.py** and set the **main_folder** variable to the root directory containing your documents (or set the `DOCUMENT_SOURCE_DIR` variable in `.env`).
2. Run the script:

```bash
python -m setup.document_loader
```

3. **Category Auto-Assignment:** If subfolder names match the category values defined in **categories.json**, documents will be automatically assigned to those categories.

Example `categories.json`:

```json
[
  {
    "id": "project",
    "name": "Project",
    "values": ["Project 1", "Project 2"]
  },
  {
    "id": "document_type",
    "name": "Document Type",
    "values": ["Technical", "Functional"]
  }
]
```

With the directory structure **main_folder/Project 1/Technical**, the document will be assigned to:
- Project: Project 1
- Document Type: Technical

## Running the Application

Make sure the following services are running before starting the application:

1. **Docker Desktop** with the Milvus container (`docker compose up -d`).
2. **Ollama** with the required models pulled.

Then activate the virtual environment and start the server:

```bash
# On Windows
.venv\Scripts\Activate

# On Linux / macOS
source .venv/bin/activate

# Start the server
hypercorn app:app
```

The application will be available at **http://127.0.0.1:8000** by default.

To bind to a custom host and port:

```bash
hypercorn app:app --bind 0.0.0.0:5000
```
