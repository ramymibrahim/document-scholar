Document Scholar Installation Guide

# What is Document Scholar
**Document Scholar** is a web application powered by local **Large Language Models (LLMs)**. It enables efficient document management, retrieval, and interactive querying.
## Key Features
1.	Answer user queries based on stored documents.
2.	Retrieve related documents using natural language queries.
3.	Handle general conversational queries.
4.	Filter and answer queries using document metadata.
5.	Display the documents used to generate an answer.
6.	Answer queries using only selected documents from previous interactions.
7.	Upload new documents into the knowledge base.
8.	Delete unwanted documents from the knowledge base.
## Technology Stack
The application is built with the following components:
1.	Ollama – serves as the local LLM runtime, enabling execution of GPU-accelerated language models.
2.	LangChain & LangGraph – manage query processing, application state, and chat history.
3.	Milvus Database – stores and retrieves documents using both vector search and semantic ranking.
4.	SQLite Database – provides metadata filtering and persists chat history and snapshots.
5.	Quart – an asynchronous Python web framework used to handle API calls.
6.	Frontend – built with pure JavaScript, HTML, and CSS for a lightweight, dependency-free interface.

# System requirements

## Hardware Specifications

The application requires a GPU with at least 11 GB of memory. It has been tested on the following:

- NVIDIA RTX A2000 12GB
- NVIDIA GTX 1080TI 11GB

## Software Specifications

Make sure the following software is installed before running the application:

- Python: 3.11.9
- CUDA: 12.1
- cuDNN: 8.9
- Ollama: 0.11.6
- Milvus Database (via Docker Desktop)

# Setup

## Milvus Setup with Docker

1. Install Docker Desktop
2. Run the following commands:
<pre markdown="1"> 
 wget <https://github.com/milvus-io/milvus/releases/download/v2.6.0/milvus-standalone-docker-compose-gpu.yml> -O docker-compose.yml
 docker compose up -d
  </pre>
## Environment Setup

1. Install the uv Python package manager:
    - pip install uv
2. Download and extract the project (or clone from source control).
3. Navigate to the project directory in your terminal and run:
<pre markdown="2"> 
uv venv .venv
.venv\\Scripts\\Activate # On Windows
uv pip install -r setup/requirements.txt
  </pre>

4. Rename the file .env.example to .env.
5. Edit the environment variables in .env according to your configuration.

| Property | Example | Definition |
| --- | --- | --- |
| APP_NAME | Document Scholar | The application name displayed on the front page. |
| APP_DESCRIPTION | Ask questions, summarize, or find files from the technical documents | The description shown on the front page. |
| GENERAL_CHAT_PROMPT | "You are a helpful assistant. You support users with Technical and Functional documents for the projects inside the Organization.\nYou can: search for documents, answer questions, summarize content, and help locate relevant material.\nRespond politely and briefly to general queries such as greetings or thanks.\nBe clear, professional, and respectful. If information is unavailable, say so without speculation." | The system prompt used in the general chat. |
| TEXT_LLM_MODEL_NAME | mistral | Ollama model name used for general text generation. |
| INSTRUCT_LLM_MODEL_NAME | ticlazau/meta-llama-3.1-8b-instruct:latest | Ollama model name used for instruction-following tasks. |
| EMBEDDING_MODEL_NAME | mxbai-embed-large | Ollama model name used for embedding tasks. |
| MILVUS_HOST | 127.0.0.1 | Host address for the Milvus database. |
| MILVUS_PORT | 19530 | Port number for the Milvus database. |
| MILVUS_DB | milv_db | Name of the Milvus database. |
| CATEGORIES_PATH | C:\\scholar\\setup\\categories.json | Absolute path to the categories JSON file, containing custom categories for filtering documents. |
| SQL_DB_PATH | C:\\scholar\\storage\\files.sqllite | Absolute path to the SQLite database for storing document metadata. If it does not exist, it will be created automatically. |
| CHECKPOINTER_DB_PATH | C:\\scholar\\storage\\checkpointer_db.sqllite | Absolute path to the SQLite database for storing chat conversations. If it does not exist, it will be created automatically. |
| DOCUMENT_FOLDER_DIR | C:\\scholar\\storage\\documents | Absolute path to the folder containing uploaded documents. These files can be downloaded if needed. |
| SESSION_SECRET_KEY | ANY_SECRET_KEY | Random secret key used for application sessions. |

## System installation

### Update Categories

After editing the .env file, update the **categories.json** file to include the categories needed for your documents. These categories are used to filter the document space during prompting.

The file is located at: **/setup/categories.json**

### Initialize Databases

Run the setup script to create the Milvus database and initialize the required tables in the SQLite database.

Make sure the virtual environment (.venv) is activated, then run: **python setup/setup.py**

### Bulk Document Loading (Optional)

If you want to load multiple documents into the system automatically (instead of uploading them one by one), use the **document_loader.py** script using the following steps.

1. Open **/setup/document_loader.py** and set the **main_folder** variable to the root directory containing your documents.
2. Run the script: <pre markdown="3"> python -m setup.document_loader </pre>
3. Category Auto-Assignment: If subfolder names match the category values defined in **categories.json**, documents will be automatically assigned to those categories. Example:
    - categories.json file:
<pre markdown="4"> 
[{
    "id": "project",
    "name": "Project",
    "values": ["Project 1", "Project 2"]
},
{
    "id": "document_type",
    "name": "Document Type",
    "values":["Technical", "Functional"]
}]
</pre>
- - Directory structure: **main_folder/Project 1/Technical**
    - Result: The document will be assigned to:
        1. Project: Project 1
        2. Document Type: Technical