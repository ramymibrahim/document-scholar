import asyncio
import traceback
import logging
from typing import AsyncIterator
from quart import Blueprint, jsonify, current_app, Response, request
import json

logger = logging.getLogger(__name__)

from uuid import uuid4

from langgraph.types import Command

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
        # Detect LangGraph interrupt events
        if "__interrupt__" in val:
            interrupts = val["__interrupt__"]
            return {
                "type": "interrupt",
                "interrupts": [
                    {"value": intr.value} for intr in interrupts
                ],
            }

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

    request_data = await request.get_json(force=True)
    logger.info(f"Chat POST received: {request_data}")
    user_input = UserInput.model_validate(request_data)
    logger.info(f"Parsed user_input.selected_documents: {user_input.selected_documents}")
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


@chat_bp.route("/<uuid:chat_id>/resume", methods=["POST"])
async def resume(chat_id):
    """Store a resume value so the next stream call picks it up."""
    if not chat_id:
        return jsonify("please provide chat_id"), 404

    chat_id = str(chat_id)
    resume_value = await request.get_json(force=True)

    if not hasattr(current_app, "pending_resumes"):
        current_app.pending_resumes = {}
    current_app.pending_resumes[chat_id] = resume_value

    return "Ok", 200


@chat_bp.route("/<uuid:chat_id>/stream", methods=["GET"])
async def astream(chat_id):
    if not chat_id:
        return jsonify("please provide chat_id"), 404

    chat_id = str(chat_id)
    chat_graph: ScholarGraph = current_app.chat_graph

    # Check if this is a resume after an interrupt
    pending_resumes = getattr(current_app, "pending_resumes", {})
    resume_value = pending_resumes.pop(chat_id, None)

    async def generate() -> AsyncIterator[bytes]:
        should_finalize = False
        try:
            if resume_value is not None:
                stream_input = Command(resume=resume_value)
            else:
                stream_input = None

            async for chunk in await chat_graph.astream(chat_id, stream_input):
                data = format_output(chunk)
                if data:
                    yield sse_data(json.dumps(data))

            # Check if we should finalize after streaming completes
            snapshot = await chat_graph.aget_state(chat_id)
            if snapshot and not snapshot.next:
                should_finalize = True
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"Stream error for chat {chat_id}: {error_msg}")
            yield sse_event("error", json.dumps({"message": str(e)}))

        yield sse_event("end", json.dumps({}))

        # Finalize outside the try block to ensure stream closes cleanly
        if should_finalize:
            asyncio.create_task(chat_graph.afinalize_node(chat_id=chat_id))

    headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
    return Response(generate(), headers=headers)


@chat_bp.route("/<uuid:chat_id>/interrupt_status", methods=["GET"])
async def get_interrupt_status(chat_id):
    """Check if a chat has a pending interrupt (for page refresh recovery)."""
    if not chat_id:
        return jsonify("please provide chat_id"), 404

    chat_graph: ScholarGraph = current_app.chat_graph
    snapshot = await chat_graph.aget_state(str(chat_id))

    if not snapshot:
        return jsonify({"has_interrupt": False}), 200

    has_interrupt = bool(snapshot.next)
    interrupt_data = None

    if has_interrupt and snapshot.tasks:
        for task in snapshot.tasks:
            if task.interrupts:
                interrupt_data = [
                    {"value": intr.value} for intr in task.interrupts
                ]
                break

    return jsonify({
        "has_interrupt": has_interrupt,
        "interrupt_data": interrupt_data,
    }), 200


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
