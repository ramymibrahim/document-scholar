"""
Chat Classification Prompt

Used for classifying user intent when there's existing chat history.
Prompt configuration is loaded from YAML files with support for project-specific overrides.
"""

import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.utils import get_buffer_string

from model.prompts.loader import load_prompt_config

# Load configuration from YAML
_config = load_prompt_config("chat_classification")

SYS_PROMPT = _config["system_prompt"]
HUMAN_PROMPT = _config["human_prompt"]
AI_PROMPT = _config["ai_prompt"]


def _convert_chat_messages(messages_config: list) -> list:
    """Convert YAML chat message format to LangChain message objects."""
    if not messages_config:
        return []

    result = []
    for msg in messages_config:
        if msg["role"] == "human":
            result.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            result.append(AIMessage(content=msg["content"]))
    return result


# Convert YAML few_shots to the expected format
few_shots = []
for shot in _config["few_shots"]:
    few_shots.append({
        "query": shot["query"],
        "has_selected_documents": shot["has_selected_documents"],
        "chat_messages": _convert_chat_messages(shot.get("chat_messages", [])),
        "output": shot["output"],
    })


def get_prompt(has_selected_documents, embeddings, query):
    filtered_examples = [
        ex for ex in few_shots
        if has_selected_documents == ex["has_selected_documents"]
    ]

    # Prepare examples with serializable format
    prepared_examples = []
    example_queries = []

    for ex in filtered_examples:
        prepared_example = {
            "query": ex["query"],
            "has_selected_documents": ex["has_selected_documents"],
            "chat_messages": get_buffer_string(
                ex.get("chat_messages") or [HumanMessage(content="Hi")]
            ),
            "output": json.dumps(ex["output"]),
        }
        prepared_examples.append(prepared_example)
        example_queries.append(ex["query"])

    if example_queries:
        all_queries = example_queries + [query]
        all_embeddings = embeddings.embed_documents(all_queries)

        # Split embeddings
        example_embeddings = np.array(all_embeddings[:-1])
        query_embedding = np.array(all_embeddings[-1]).reshape(1, -1)

        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, example_embeddings)[0]

        top_indices = np.argsort(similarities)[-5:][::-1]
        selected_examples = [prepared_examples[i] for i in top_indices]
    else:
        selected_examples = []

    example_prompt = ChatPromptTemplate.from_messages(
        [("human", HUMAN_PROMPT), ("ai", AI_PROMPT)],
        template_format="jinja2",
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=selected_examples,
    )

    final_prompt = (
        SystemMessagePromptTemplate.from_template(SYS_PROMPT, template_format="jinja2")
        + few_shot_prompt
        + HumanMessagePromptTemplate.from_template(
            HUMAN_PROMPT, template_format="jinja2"
        )
    )

    return final_prompt
