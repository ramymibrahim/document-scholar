from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma.vectorstores import Chroma
import json
from langchain_core.prompts import (
    FewShotPromptTemplate,
    PromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.messages.utils import get_buffer_string
from langchain_core.messages import HumanMessage, AIMessage
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

Context usage:
- Use chat_messages (last 4) to detect follow-ups, corrections, and stable user preferences (e.g., frameworks, folders).
- Set "depend_on_last_task" = true when the user continues/corrects/extends or aligns with the chat_messages ; else false.
- Reuse stable terms from historical_summary in search queries when relevant.

Scope rules:
- "scope" = "selected_documents" if has_selected_documents is True OR the user clearly asks to use selected docs (e.g., "from selected", "these files"). Otherwise "generic".
- If scope="selected_documents" AND the request is summarize/explain/compare/extract (no need to find more docs), set "generated_search_queries":[] so the next step uses only the selected document IDs.
- If the user asks to use selected docs but no IDs exist, set scope="selected_documents", set "generated_search_queries":[], and in generated_llm_prompt instruct the next step to ask for the IDs before continuing.

Queries & prompt:
- Produce 2–4 precise search queries for vector/semantic search; use [] for "general".
- For summarization requests (with queries or selected docs) ALWAYS set generated_llm_prompt to "Summarize the retrieved documents."
- For other technical tasks, generated_llm_prompt must be a single actionable instruction (≤80 words).
- If unsure between "technical" and "find_documents", prefer "find_documents" when verbs like find/locate/search/list/fetch/retrieve/filter are used WITHOUT summarization/explanation; otherwise choose "technical".

General rules:
- Output keys must appear exactly in the schema order.
- Do not leak personal data beyond what is required.
- Do not invent document IDs or facts.
</INSTRUCTIONS>

<SCHEMA>
{
  "type": "general | technical | find_documents",
  "generated_search_queries": ["string", "..."],
  "generated_llm_prompt": "string",
  "depend_on_last_task": true | false,
  "scope": "generic | selected_documents"
}
</SCHEMA>
"""
HUMAN_PROMPT = """Query: {{query}}
Has selected documents:{{has_selected_documents}}
chat_messages:{{chat_messages}}"""

AI_PROMPT = """{{output}}"""


few_shots = [
    {
        "query": "Hello there!",
        "has_selected_documents": False,
        "chat_messages": [],
        "output": {
            "type": "general",
            "generated_search_queries": [],
            "generated_llm_prompt": "Greet the user politely and offer assistance with technical documents or tasks.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "How do I authenticate to the CDS API?",
        "has_selected_documents": False,
        "chat_messages": [],
        "output": {
            "type": "technical",
            "generated_search_queries": [
                "CDS API authentication",
                "CDS OAuth2 client credentials",
                "CDS token endpoint",
                "CDS auth header format",
            ],
            "generated_llm_prompt": "Explain how to authenticate to the CDS API using OAuth2. Include required grant type, token endpoint, and headers.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Find documents about refresh token rotation in CDS.",
        "has_selected_documents": False,
        "chat_messages": [],
        "output": {
            "type": "find_documents",
            "generated_search_queries": [
                "CDS refresh token rotation",
                "CDS token lifecycle",
                "CDS OAuth2 refresh flow",
                "CDS security guidelines refresh tokens",
            ],
            "generated_llm_prompt": "Retrieve relevant documents about refresh token rotation in CDS and list them with short summaries.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Summarize the selected documents about error handling.",
        "has_selected_documents": True,
        "chat_messages": [],
        "output": {
            "type": "technical",
            "generated_search_queries": [],
            "generated_llm_prompt": "Summarize the retrieved documents.",
            "depend_on_last_task": False,
            "scope": "selected_documents",
        },
    },
    {
        "query": "And what about the required headers?",
        "has_selected_documents": False,
        "chat_messages": [
            HumanMessage(content="How do I authenticate to the CDS API?"),
            AIMessage(
                content="You need to use OAuth2 client credentials and request a token."
            ),
            HumanMessage(content="Where can I find the token endpoint?"),
            AIMessage(
                content="The token endpoint is in the CDS authentication section."
            ),
        ],
        "output": {
            "type": "technical",
            "generated_search_queries": [
                "CDS API authentication headers",
                "CDS OAuth2 request headers",
                "CDS API Authorization header format",
            ],
            "generated_llm_prompt": "Answer the users follow-up about required headers for CDS API authentication.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Can you also find ones about token revocation?",
        "has_selected_documents": False,
        "chat_messages": [
            HumanMessage(content="Find documents about refresh token rotation in CDS."),
            AIMessage(content="I found 3 documents related to refresh token rotation."),
            HumanMessage(content="List them with short summaries."),
            AIMessage(content="Here are summaries of the three documents."),
        ],
        "output": {
            "type": "find_documents",
            "generated_search_queries": [
                "CDS token revocation",
                "CDS OAuth2 revoke token endpoint",
                "CDS security guidelines token revocation",
            ],
            "generated_llm_prompt": "Retrieve documents about token revocation in CDS and list them with summaries.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Now summarize these docs.",
        "has_selected_documents": True,
        "chat_messages": [
            HumanMessage(content="Find documents about error handling in CDS."),
            AIMessage(content="I found 2 documents related to error handling."),
            HumanMessage(content="Show me the list."),
            AIMessage(content="Here are the document titles and IDs."),
        ],
        "output": {
            "type": "technical",
            "generated_search_queries": [],
            "generated_llm_prompt": "Summarize the retrieved documents.",
            "depend_on_last_task": False,
            "scope": "selected_documents",
        },
    },
    {
        "query": "By the way, do you enjoy working with APIs?",
        "has_selected_documents": False,
        "chat_messages": [
            HumanMessage(content="How do I authenticate to the CDS API?"),
            AIMessage(
                content="You need OAuth2 client credentials with a token endpoint."
            ),
            HumanMessage(content="Thanks, that helps!"),
            AIMessage(
                content="Glad to help. Would you like examples of request headers?"
            ),
        ],
        "output": {
            "type": "general",
            "generated_search_queries": [],
            "generated_llm_prompt": "Respond in a friendly way and mention you are focused on helping with technical documents.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "No, I mean DPS not CDS.",
        "has_selected_documents": False,
        "chat_messages": [
            HumanMessage(
                content="Summarize documents about SMTP settings in Ethiopia."
            ),
            AIMessage(content="Here is a summary of CDS SMTP settings in Ethiopia."),
            HumanMessage(content="No, I mean DPS not CDS."),
            AIMessage(content="Got it, DPS instead of CDS. Let me adjust."),
        ],
        "output": {
            "type": "technical",
            "generated_search_queries": [
                "DPS SMTP settings Ethiopia",
                "DPS email configuration Ethiopia",
                "DPS SMTP server setup Ethiopia",
            ],
            "generated_llm_prompt": "Summarize the retrieved documents.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Explain the retry policy for failed requests in DPS.",
        "has_selected_documents": False,
        "chat_messages": [
            HumanMessage(content="Hi there!"),
            AIMessage(
                content="Hello! I can help you with technical documents and tasks. What would you like to know?"
            ),
            HumanMessage(
                content="Explain the retry policy for failed requests in DPS."
            ),
            AIMessage(content="Sure, let me look that up."),
        ],
        "output": {
            "type": "technical",
            "generated_search_queries": [
                "DPS retry policy failed requests",
                "DPS error handling retries",
                "DPS exponential backoff",
                "DPS request retry mechanism",
            ],
            "generated_llm_prompt": "Explain the retry policy for failed requests in DPS using the retrieved documents.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Summarize the selected documents about DPS retry policy.",
        "has_selected_documents": True,
        "chat_messages": [
            HumanMessage(content="Find documents about retry policy in DPS."),
            AIMessage(content="I found two documents related to retry policy in DPS."),
            HumanMessage(
                content="Summarize the selected documents about DPS retry policy."
            ),
            AIMessage(content="Okay, I will summarize only the selected documents."),
        ],
        "output": {
            "type": "technical",
            "generated_search_queries": [],
            "generated_llm_prompt": "Summarize only the selected documents about DPS retry policy.",
            "depend_on_last_task": False,
            "scope": "selected_documents",
        },
    },
]

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