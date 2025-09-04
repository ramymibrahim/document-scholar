from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from model.domain.core import GraphState, Task
from model.prompts.fresh_classification import get_prompt as get_fresh_prompt
from model.prompts.chat_classification import get_prompt as get_chat_prompt
from langchain_core.messages import get_buffer_string
import json


def classify_and_extract_node(
    state: GraphState, llm: ChatOllama, embedding_model: OllamaEmbeddings
):
    state.tool_messages = [
        ToolMessage(content="Thinking", tool_call_id="classify_and_extract_node")
    ]
    parser = PydanticOutputParser(pydantic_object=Task)
    has_selected_documents = bool(state.user_input.selected_documents)

    if not state.chat_messages:
        prompt = get_fresh_prompt(has_selected_documents, embedding_model)
        inputs = {
            "query": state.user_input.query,
            "has_selected_documents": has_selected_documents,
        }
    else:
        prompt = get_chat_prompt(
            has_selected_documents, embedding_model, state.user_input.query
        )
        inputs = {
            "query": state.user_input.query,
            "has_selected_documents": has_selected_documents,
            "chat_messages": get_buffer_string(
                state.chat_messages or [HumanMessage(content="Hi")]
            ),
        }
    chain = prompt | llm | parser
    task: Task = chain.invoke(inputs, config={"temperature": 0.1, "callbacks": []})
    state.task = task
    state.tool_messages = [
        ToolMessage(
            content=task.get_description(), tool_call_id="classify_and_extract_node"
        )
    ]
    return state
