from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from model.domain.core import Conversation, GraphState
from services.vector_db_service import VectorDbService

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    FewShotPromptTemplate,
    PromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)


async def inquiry(
    state: GraphState, text_llm: ChatOllama, vector_db: VectorDbService
):

    file_ids = state.user_input.selected_documents or None
    if not file_ids and state.user_input.filter:
        file_ids = vector_db.get_file_ids(state.user_input.filter)

    if not state.task.generated_search_queries and not file_ids:
        # No query generated and the prompt is not related with selected files or filter
        state.chat_messages = [
            HumanMessage(content=state.user_input.query),
            AIMessage(content="Please provide valid query"),
        ]
        return state

    queries = state.task.generated_search_queries or [""]  # defulat query to get all

    documents = await vector_db.get_documents(queries, file_ids)

    documents = [doc for doc in documents if doc.metadata.get("score", 0.0) >= 0.5]

    context = vector_db.get_context(documents)

    prompt = ChatPromptTemplate.from_messages(
        [
            *state.chat_messages,
            ("system", "use ONLY the provided context to do the following."),
            ("system", state.task.generated_llm_prompt),
            ("user", "user query:\n{query}\n\ncontext:\n{context}\n\nAnswer:"),
        ]
    )
    chain = prompt | text_llm | StrOutputParser()

    # Collect streamed tokens
    ai_text = []
    async for chunk in chain.astream(
        {"query": state.user_input.query, "context": context}
    ):
        ai_text.append(chunk)

    full_reply = "".join(ai_text)

    state.last_conversation = Conversation(
        task=state.task,
        documents=documents,
        request=HumanMessage(content=state.user_input.query),
        response=AIMessage(content=full_reply),
    )
    state.chat_messages = [
        HumanMessage(content=state.user_input.query),
        AIMessage(content=full_reply),
    ]
    return state
