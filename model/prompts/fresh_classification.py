from langchain_core.prompts import (
    FewShotPromptTemplate,
    PromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma.vectorstores import Chroma
import json

SYS_PROMPT = """
<SYSTEM>
You classify intent for a technical assistant that uses internal technical documents.
</SYSTEM>

<INSTRUCTIONS>
Return ONLY valid JSON that matches the schema. No extra text.
Decide intent:
- "general": greetings, small talk, meta questions about capabilities.
- "technical": asks to answer/explain/summarize/solve using technical docs, specs, APIs, or code.
- "find_documents": asks to find/locate/search/list/filter documents (lists only; no LLM summary).

Rules:
- Produce 2–4 precise search queries for vector/semantic search; use [] for "general".
- Scope is "selected_documents" if:
  1) has_selected_documents is True, or
  2) the user clearly asks to use selected docs (e.g., "from selected", "these files").
- If scope="selected_documents" AND the request is summarize/explain/compare/extract (no need to find more docs), then set "generated_search_queries":[] so the next step will use only the selected document IDs.
- If user asks for "selected documents" but no IDs are provided, set scope="selected_documents", return "generated_search_queries":[] and make generated_llm_prompt ask for IDs before continuing.
- "depend_on_last_task"=false in this case.
- "generated_llm_prompt":
  * For summarization requests (whether with queries or selected docs) ALWAYS: "Summarize the retrieved documents."
  * Otherwise: one clear instruction ≤80 words.
- If unsure between "technical" and "find_documents", prefer "find_documents" when verbs like find, locate, search, list, fetch, retrieve, filter are used without summarization/explanation. Otherwise choose "technical".
- Output keys must appear exactly in the schema order.
</INSTRUCTIONS>

<SCHEMA>
{
  "type": "general | technical | find_documents",
  "generated_search_queries": ["string", "..."],
  "generated_llm_prompt": "string",
  "depend_on_last_task": false,
  "scope": "generic | selected_documents"
}
</SCHEMA>
"""


HUMAN_PROMPT = """Query: {{query}}
Has selected documents:{{has_selected_documents}}"""

AI_PROMPT = """{{output}}"""

few_shots = [
    {
        "query": "Hi there!",
        "has_selected_documents": False,
        "output": {
            "type": "general",
            "generated_search_queries": [],
            "generated_llm_prompt": "Greet the user briefly and offer help with technical documents or tasks.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "How do I authenticate to the CDS API?",
        "has_selected_documents": False,
        "output": {
            "type": "technical",
            "generated_search_queries": [
                "CDS API authentication",
                "CDS OAuth2 client credentials",
                "CDS token endpoint",
                "CDS auth header format",
            ],
            "generated_llm_prompt": "Answer the authentication question using CDS docs. Explain the required grant type, token endpoint, and authorization headers.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Find documents about refresh token rotation in CDS.",
        "has_selected_documents": False,
        "output": {
            "type": "find_documents",
            "generated_search_queries": [
                "CDS refresh token rotation",
                "CDS token lifecycle",
                "CDS OAuth2 refresh flow",
                "CDS security guidelines refresh tokens",
            ],
            "generated_llm_prompt": "Retrieve the most relevant documents on refresh token rotation and list each with a one-line summary.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Summarize the selected documents about error handling.",
        "has_selected_documents": True,
        "output": {
            "type": "technical",
            "generated_search_queries": [],
            "generated_llm_prompt": "Summarize the retrieved documents.",
            "depend_on_last_task": False,
            "scope": "selected_documents",
        },
    },
    {
        "query": "Summarize the selected document.",
        "has_selected_documents": False,
        "output": {
            "type": "technical",
            "generated_search_queries": [],
            "generated_llm_prompt": "Ask the user to provide the document IDs to summarize before proceeding.",
            "depend_on_last_task": False,
            "scope": "selected_documents",
        },
    },
    {
        "query": "Compare the selected docs and highlight differences in rate limits.",
        "has_selected_documents": True,
        "output": {
            "type": "technical",
            "generated_search_queries": [],
            "generated_llm_prompt": "Summarize the retrieved documents.",
            "depend_on_last_task": False,
            "scope": "selected_documents",
        },
    },
    {
        "query": "Do you like working with APIs?",
        "has_selected_documents": False,
        "output": {
            "type": "general",
            "generated_search_queries": [],
            "generated_llm_prompt": "Respond in a friendly way and explain that you can help with technical documents and API questions.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "banana banana",
        "has_selected_documents": False,
        "output": {
            "type": "general",
            "generated_search_queries": [],
            "generated_llm_prompt": "Politely ask the user to clarify their request or provide more details.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Hi, can you also tell me how to authenticate to the CDS API?",
        "has_selected_documents": False,
        "output": {
            "type": "technical",
            "generated_search_queries": [
                "CDS API authentication",
                "CDS OAuth2 client credentials",
                "CDS token endpoint",
                "CDS auth header format",
            ],
            "generated_llm_prompt": "Answer the authentication question using CDS docs. Greet the user briefly, then explain required grant type, token endpoint, and authorization headers.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Find documents about refresh tokens and explain how they work.",
        "has_selected_documents": False,
        "output": {
            "type": "find_documents",
            "generated_search_queries": [
                "CDS refresh token rotation",
                "CDS refresh token lifecycle",
                "CDS OAuth2 refresh flow",
                "CDS token renewal process",
            ],
            "generated_llm_prompt": "Locate documents on refresh tokens and, after retrieving them, provide an explanation of how refresh tokens work.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Summarize documents that contain SMTP settings in DPS in Ethiopia",
        "has_selected_documents": False,
        "output": {
            "type": "technical",
            "generated_search_queries": [
                "DPS SMTP settings Ethiopia",
                "CDS DPS email configuration Ethiopia",
                "DPS SMTP server setup Ethiopia",
                "DPS messaging email SMTP Ethiopia",
            ],
            "generated_llm_prompt": "Summarize the retrieved documents.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
]


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
