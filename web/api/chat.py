import asyncio
from typing import AsyncIterator
from quart import Blueprint, jsonify, current_app, Response, request
import json

from uuid import uuid4

from model.domain.core import GraphState, UserInput, to_jsonable
from model.chat_graph import ScholarGraph
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    ToolMessage,
    messages_to_dict,
)

chat_bp = Blueprint("chat", __name__)


def format_output(chunk):
    type, val = chunk

    if type == "messages":
        token, meta = val
        if isinstance(token, (BaseMessage, BaseMessageChunk)):
            return token.model_dump()
    if type == "updates":
        token = next(iter(val.values()), None)
        if not token:
            return None
        if "tool_messages" in token:
            messages: list[ToolMessage] = token["tool_messages"]
            if not messages:
                return None
            return messages_to_dict(messages)
    return None


def sse_event(type, data):
    return f"event: {type}\ndata:{data}\n\n"


def sse_data(data):
    return f"data: {data}\n\n"


@chat_bp.route("/<uuid:chat_id>", methods=["POST"])
async def chat(chat_id):
    if not chat_id:
        return (jsonify("please provide chat_id"), 404)

    chat_id = str(chat_id)

    user_input = UserInput.model_validate(await request.get_json(force=True))
    state = GraphState(
        user_input=user_input,
        last_conversation=None,
        task=None,
        tool_messages=[],
        is_finalized=False,
    )
    chat_graph: ScholarGraph = current_app.chat_graph
    await chat_graph.aupdate_state(chat_id, state)

    return "Ok", 200


@chat_bp.route("/<uuid:chat_id>/stream", methods=["GET"])
async def astream(chat_id):
    if not chat_id:
        return jsonify("please provide chat_id"), 404

    chat_id = str(chat_id)
    chat_graph: ScholarGraph = current_app.chat_graph

    async def generate() -> AsyncIterator[bytes]:
        try:
            async for chunk in await chat_graph.astream(chat_id, None):
                data = format_output(chunk)
                if data:
                    yield sse_data(json.dumps(data))
        except Exception as e:
            yield sse_event("error", {"message": str(e)})
            raise
        yield sse_event("end", {})
        asyncio.create_task(chat_graph.afinalize_node(chat_id=chat_id))

    headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
    return Response(generate(), headers=headers)


@chat_bp.route("/<uuid:chat_id>/current_state", methods=["Get"])
async def aget_current_state(chat_id):
    if not chat_id:
        return (jsonify("please provide chat_id"), 404)

    chat_graph: ScholarGraph = current_app.chat_graph
    snapshot = await chat_graph.aget_state(str(chat_id))
    if not snapshot:
        return "Ok", 200
    state = GraphState.model_validate(snapshot.values)
    return jsonify(state.model_dump()), 200


@chat_bp.route("/<uuid:chat_id>", methods=["GET"])
async def get_chat(chat_id):
    if not chat_id:
        return jsonify("chat id not found"), 404
    chat_graph: ScholarGraph = current_app.chat_graph
    snaps = []
    async for snapshot in chat_graph.aget_chat_history(chat_id):
        if not snapshot.next:
            state: GraphState = GraphState.model_validate(snapshot.values).model_dump()
            if bool(state.get("is_finalized", False)):
                snaps.append(state)
    return jsonify(snaps), 200


@chat_bp.route("/get_new_chat_id", methods=["GET"])
async def get_new_chat_id():
    chat_id = str(uuid4())
    return jsonify({"chat_id": chat_id}), 200


@chat_bp.route("/<uuid:chat_id>", methods=["DELETE"])
async def delete(chat_id):
    if not chat_id:
        return jsonify("chat id not found"), 404

    graph: ScholarGraph = current_app.chat_graph
    graph.delete_thread(str(chat_id))
    return jsonify("session cleared successfully"), 200
