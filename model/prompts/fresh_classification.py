"""
Fresh Classification Prompt

Used for classifying user intent when there's no chat history.
Prompt configuration is loaded from YAML files with support for project-specific overrides.
"""

import json

from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma.vectorstores import Chroma

from model.prompts.loader import load_prompt_config

# Load configuration from YAML
_config = load_prompt_config("fresh_classification")

SYS_PROMPT = _config["system_prompt"]
HUMAN_PROMPT = _config["human_prompt"]
AI_PROMPT = _config["ai_prompt"]

# Convert YAML few_shots to the expected format
few_shots = []
for shot in _config["few_shots"]:
    few_shots.append({
        "query": shot["query"],
        "has_selected_documents": shot["has_selected_documents"],
        "output": shot["output"],
    })


def get_prompt(has_selected_documents, embeddings):
    examples = [
        {
            "query": ex["query"],
            "has_selected_documents": ex["has_selected_documents"],
            "output": json.dumps(ex["output"]),
        }
        for ex in few_shots
        if has_selected_documents == ex["has_selected_documents"]
    ]
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=examples,
        vectorstore_cls=Chroma,
        k=3,
        embeddings=embeddings,
        input_keys=["query"],
    )

    example_prompt = ChatPromptTemplate.from_messages(
        [("human", HUMAN_PROMPT), ("ai", AI_PROMPT)], template_format="jinja2"
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt, example_selector=example_selector
    )
    final_prompt = (
        SystemMessagePromptTemplate.from_template(SYS_PROMPT, template_format="jinja2")
        + few_shot_prompt
        + HumanMessagePromptTemplate.from_template(
            HUMAN_PROMPT, template_format="jinja2"
        )
    )

    return final_prompt
