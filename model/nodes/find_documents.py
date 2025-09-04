from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from model.domain.core import Conversation, GraphState
from services.vector_db_service import VectorDbService
from langchain_core.documents import Document


async def find_documents(state: GraphState, vector_db: VectorDbService):

    file_ids = state.user_input.selected_documents or None
    if not file_ids and state.user_input.filter:
        file_ids = vector_db.get_file_ids(state.user_input.filter)

    if not state.task.generated_search_queries:
        state.chat_messages = [
            HumanMessage(content=state.user_input.query),
            AIMessage(content="Please provide valid query"),
        ]
        return state

    queries = state.task.generated_search_queries

    documents = await vector_db.get_documents(queries, file_ids)
    documents = [doc for doc in documents if doc.metadata.get("score", 0.0) >= 0.7]
    document_ids = set()
    for doc in documents:
        _d: Document = doc
        document_ids.add(_d.metadata["file_id"])

    if not documents:
        ai_message = AIMessage(content=f"No documents found")
    elif len(document_ids) == 1:
        ai_message = AIMessage(content=f"{len(document_ids)} document found.")
    else:
        ai_message = AIMessage(content=f"{len(document_ids)} documents found.")
    state.last_conversation = Conversation(
        task=state.task,
        documents=documents,
        request=HumanMessage(content=state.user_input.query),
        response=ai_message,
    )
    state.tool_messages = [
        ToolMessage(
            content=f"{len(document_ids)} documents found.",
            tool_call_id="find_documents",
        ),
    ]
    state.chat_messages = [
        HumanMessage(content=state.user_input.query),
        AIMessage(content=f"{len(document_ids)} documents found."),
    ]
    return state
