from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, START, END

from model.domain.core import GraphState, Task, TaskType
from services.checkpointer import CheckPointer

from model.nodes.classify_and_extract_node import classify_and_extract_node
from model.nodes.technical import technical
from model.nodes.find_documents import find_documents
from model.nodes.general import general
from model.nodes.finalize import finalize, pre_finalize
from services.vector_db_service import VectorDbService


class ScholarGraph:
    def __init__(
        self,
        llm_model: ChatOllama,
        instruct_llm_model: ChatOllama,
        embedding_model: OllamaEmbeddings,
        vector_db: VectorDbService,
        checkpointer: CheckPointer,
        general_chat_prompt: str,
    ):
        self.llm_model = llm_model
        self.instruct_llm_model = instruct_llm_model
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.general_chat_prompt = general_chat_prompt
        self.checkpointer = checkpointer

        self.graph = self.build_graph()
        self.finalize_graph = self.build_finalize_graph()

    async def astream(self, chat_id, state):
        config = {"configurable": {"thread_id": chat_id, "checkpoint_ns": "main"}}
        return self.graph.astream(state, config, stream_mode=["messages", "updates"])

    async def afinalize_node(self, chat_id):
        config = {"configurable": {"thread_id": chat_id, "checkpoint_ns": "finalize"}}
        return await self.finalize_graph.ainvoke({}, config)

    async def aupdate_state(self, chat_id, state):
        config = {"configurable": {"thread_id": chat_id}}
        await self.graph.aupdate_state(config, state, as_node=START)

    async def aget_state(self, chat_id):
        config = {"configurable": {"thread_id": chat_id}}
        return await self.graph.aget_state(config)

    def aget_chat_history(self, chat_id):
        config = {"configurable": {"thread_id": chat_id}}
        return self.graph.aget_state_history(config)

    def delete_thread(self, chat_id):
        self.checkpointer.delete_thread(chat_id)

    def build_graph(self):
        graph_builder = StateGraph(GraphState)

        graph_builder.add_node(
            "classify_and_extract_node", self.classify_and_extract_node
        )
        graph_builder.add_node("technical", self.technical)
        graph_builder.add_node("find_documents", self.find_documents)
        graph_builder.add_node("general", self.general)

        graph_builder.add_edge(START, "classify_and_extract_node")

        graph_builder.add_conditional_edges("classify_and_extract_node", self.router)
        graph_builder.add_edge("technical", END)
        graph_builder.add_edge("find_documents", END)
        graph_builder.add_edge("general", END)

        return graph_builder.compile(checkpointer=self.checkpointer.checkpointer)

    def build_finalize_graph(self):
        graph_builder = StateGraph(GraphState)
        graph_builder.add_node("finalize", self.finalize)
        graph_builder.add_edge(START, "finalize")
        graph_builder.add_edge("finalize", END)
        return graph_builder.compile(checkpointer=self.checkpointer.checkpointer)

    ### Nodes ###
    def classify_and_extract_node(self, state: GraphState):
        return classify_and_extract_node(
            state, self.instruct_llm_model, self.embedding_model
        )

    async def technical(self, state: GraphState):
        return await technical(state, self.llm_model, self.vector_db)

    async def find_documents(self, state: GraphState):
        return await find_documents(state, self.vector_db)

    async def general(self, state: GraphState):
        return await general(state, self.llm_model, self.general_chat_prompt)

    def finalize(self, state: GraphState):
        return finalize(state, self.instruct_llm_model)

    def router(self, state: GraphState):
        if state.task.type == TaskType.technical:
            return "technical"
        if state.task.type == TaskType.find_documents:
            return "find_documents"
        return "general"
