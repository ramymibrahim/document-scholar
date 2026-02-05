from typing import Any, Mapping
from pydantic import BaseModel
from datetime import date, datetime
from enum import Enum
from langchain_core.messages import BaseMessage, AIMessage, ChatMessage, ToolMessage
from langchain_core.documents import Document
from langgraph.channels.last_value import LastValue
from langgraph.graph.message import add_messages, Annotated


def to_jsonable(obj: Any) -> Any:
    # 1) LangChain chat messages (Human/AI/System/Chat/Tool)
    if isinstance(obj, BaseMessage):
        role = getattr(obj, "role", None) or getattr(obj, "type", "assistant")
        out = {
            "role": role,
            "content": obj.content,
        }
        # Optional/diagnostic fields if present
        name = getattr(obj, "name", None)
        if name:
            out["name"] = name
        mid = getattr(obj, "id", None)
        if mid:
            out["id"] = mid
        tool_call_id = getattr(obj, "tool_call_id", None)
        if tool_call_id:
            out["tool_call_id"] = tool_call_id
        tool_calls = getattr(obj, "tool_calls", None)
        if tool_calls:
            out["tool_calls"] = to_jsonable(tool_calls)
        additional_kwargs = getattr(obj, "additional_kwargs", None)
        if additional_kwargs:
            out["additional_kwargs"] = to_jsonable(additional_kwargs)
        return out

    # 2) LangChain Document
    if isinstance(obj, Document):
        return {
            "page_content": obj.page_content,
            "metadata": to_jsonable(obj.metadata),
        }

    # 3) Your Pydantic models (UserInput, Task, Conversation, GraphState, etc.)
    if isinstance(obj, BaseModel):
        # mode="json" makes enums/datetimes JSON-friendly; exclude None keeps payload small
        return obj.model_dump(mode="json", exclude_none=True)

    # 4) Enums (TaskType, Scope)
    if isinstance(obj, Enum):
        return obj.value

    # 5) Datetimes / dates
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # 6) Mappings / dicts (may include metadata)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # 7) Iterables (lists/tuples/sets)
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    # 8) Primitives pass through
    return obj


class Category(BaseModel):
    id: str
    categories: list[str]


class UserFilter(BaseModel):
    file: str | None = None
    folder: str | None = None
    author: str | None = None
    category_ids: list[Category] | None = None
    created_from: str | None = None
    created_to: str | None = None
    updated_from: str | None = None
    updated_to: str | None = None


class UserInput(BaseModel):
    query: str
    filter: UserFilter | None = None
    selected_documents: list[str] | None = []


class TaskType(str, Enum):
    general = "general"
    inquiry = "inquiry"
    find_documents = "find_documents"
    send_email = "send_email"


class Scope(str, Enum):
    generic = "generic"
    selected_documents = "selected_documents"


class Task(BaseModel):
    type: TaskType
    generated_search_queries: list[str]
    generated_llm_prompt: str
    depend_on_last_task: bool
    scope: Scope

    def get_description(self):
        if self.type == TaskType.find_documents:
            return "Finding documents"
        if self.type == TaskType.inquiry:
            return "Handling inquiry"
        if self.type == TaskType.send_email:
            return "Preparing email"
        return "General chat"


class Conversation(BaseModel):
    task: Task
    documents: list[Document] | None = None
    request: BaseMessage
    response: BaseMessage


class GraphState(BaseModel):
    user_input: UserInput

    task: Task | None = None

    last_conversation: Annotated[Conversation, LastValue] | None = None

    tool_messages: Annotated[list[ToolMessage], LastValue] = []
    chat_messages: Annotated[list[BaseMessage], add_messages] = []
    historical_summary: str | None = None

    is_finalized: Annotated[bool, LastValue] = False
