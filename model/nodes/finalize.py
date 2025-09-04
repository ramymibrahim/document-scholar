from model.domain.core import GraphState
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.messages import HumanMessage, ToolMessage


def pre_finalize(state: GraphState):
    state.tool_messages = [
        ToolMessage(content="pre_finalize", tool_call_id="pre_finalize")
    ]
    return state


def finalize(state: GraphState, llm: ChatOllama):
    summary = state.historical_summary
    if summary:
        summary_message = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    messages = state.chat_messages + [HumanMessage(content=summary_message)]
    response = (llm | StrOutputParser()).invoke(messages, config={"temperature": 0.1, "callbacks": []})

    if len(state.chat_messages) > 4:
        delete_messages = [RemoveMessage(id=m.id) for m in state.chat_messages[:-4]]
        state.chat_messages = delete_messages
    state.historical_summary = response
    state.is_finalized = True
    return state
