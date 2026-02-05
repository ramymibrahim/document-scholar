import asyncio

from langgraph.types import interrupt
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from model.domain.core import Conversation, GraphState
from services.email_service import EmailService


async def send_email_node(state: GraphState, email_service: EmailService):
    # Step 1: Check if summary context exists in conversation memory
    summary_content = _get_summary_content(state)
    if not summary_content:
        ai_msg = AIMessage(
            content="I don't have any summary to send. "
            "Please ask me to summarize some documents first, "
            "then request the email."
        )
        state.chat_messages = [
            HumanMessage(content=state.user_input.query),
            ai_msg,
        ]
        state.last_conversation = Conversation(
            task=state.task,
            documents=None,
            request=HumanMessage(content=state.user_input.query),
            response=ai_msg,
        )
        return state

    # Step 2: First interrupt — ask for email address and name
    email_info = interrupt(
        {
            "type": "email_input_request",
            "message": "Please provide the recipient's email address and name.",
            "fields": [
                {
                    "name": "email",
                    "label": "Email Address",
                    "input_type": "email",
                    "required": True,
                },
                {
                    "name": "name",
                    "label": "Recipient Name",
                    "input_type": "text",
                    "required": True,
                },
            ],
        }
    )

    recipient_email = email_info.get("email", "").strip()
    recipient_name = email_info.get("name", "").strip()

    if not recipient_email or not recipient_name:
        ai_msg = AIMessage(
            content="Email address and name are required. Please try again."
        )
        state.chat_messages = [
            HumanMessage(content=state.user_input.query),
            ai_msg,
        ]
        state.last_conversation = Conversation(
            task=state.task,
            documents=None,
            request=HumanMessage(content=state.user_input.query),
            response=ai_msg,
        )
        return state

    # Step 3: Second interrupt — ask for confirmation
    body_preview = (
        summary_content[:200] + "..."
        if len(summary_content) > 200
        else summary_content
    )
    confirmation = interrupt(
        {
            "type": "email_confirmation_request",
            "message": f"Send summary email to {recipient_name} at {recipient_email}?",
            "preview": {
                "to_name": recipient_name,
                "to_email": recipient_email,
                "subject": "Document Summary",
                "body_preview": body_preview,
            },
        }
    )

    confirmed = confirmation.get("confirmed", False)

    if not confirmed:
        ai_msg = AIMessage(content="Email sending cancelled.")
        state.chat_messages = [
            HumanMessage(content=state.user_input.query),
            ai_msg,
        ]
        state.last_conversation = Conversation(
            task=state.task,
            documents=None,
            request=HumanMessage(content=state.user_input.query),
            response=ai_msg,
        )
        return state

    # Step 4: Send the email
    result = await asyncio.to_thread(
        email_service.send_email,
        to_email=recipient_email,
        to_name=recipient_name,
        subject="Document Summary",
        body=summary_content,
    )

    if result.success:
        ai_msg = AIMessage(
            content=f"Email sent successfully to {recipient_name} ({recipient_email})."
        )
    else:
        ai_msg = AIMessage(
            content=f"Failed to send email: {result.error_message}. "
            "Please try again later."
        )

    state.chat_messages = [
        HumanMessage(content=state.user_input.query),
        ai_msg,
    ]
    state.last_conversation = Conversation(
        task=state.task,
        documents=None,
        request=HumanMessage(content=state.user_input.query),
        response=ai_msg,
    )
    state.tool_messages = [
        ToolMessage(
            content="Email sent" if result.success else "Email failed",
            tool_call_id="send_email",
        )
    ]
    return state


def _get_summary_content(state: GraphState) -> str | None:
    """Extract summary content from conversation history."""
    content = None

    if state.historical_summary:
        content = state.historical_summary
    elif state.last_conversation and state.last_conversation.response:
        content = state.last_conversation.response.content
    elif state.chat_messages:
        for msg in reversed(state.chat_messages):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content
                break

    if not content:
        return None

    # Handle JSON-formatted responses from LLM
    return _extract_text_from_content(content)


def _extract_text_from_content(content: str) -> str:
    """Extract plain text from content that may be JSON-formatted."""
    import json

    # Try to parse as JSON and extract text field
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            # Try common text field names
            for key in ("text", "summary", "content", "body", "message"):
                if key in parsed and isinstance(parsed[key], str):
                    return parsed[key]
        # If parsed but no text field found, return original
        return content
    except (json.JSONDecodeError, TypeError):
        # Not JSON, return as-is
        return content
