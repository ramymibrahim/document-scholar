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
You classify intent for an assistant that uses WHO situation reports.
</SYSTEM>

<INSTRUCTIONS>
Return ONLY valid JSON that matches the schema. No extra text.

Decide intent:
- "general": greetings, small talk, meta questions about capabilities.
- "inquiry": asks to answer/explain/summarize/solve using WHO situation reports, epidemiological data, or public health guidance.
- "find_documents": asks to find/locate/search/list/filter situation reports (lists only; no LLM summary).
- "send_email": user wants to email/send a summary, report, or conversation content to someone.

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
- For summarization requests (with queries or selected docs) ALWAYS set generated_llm_prompt to "Summarize the retrieved situation reports."
- For other inquiry tasks, generated_llm_prompt must be a single actionable instruction (≤80 words).
- If unsure between "inquiry" and "find_documents", prefer "find_documents" when verbs like find/locate/search/list/fetch/retrieve/filter are used WITHOUT summarization/explanation; otherwise choose "inquiry".

General rules:
- Output keys must appear exactly in the schema order.
- Do not leak personal data beyond what is required.
- Do not invent document IDs or facts.
</INSTRUCTIONS>

<SCHEMA>
{
  "type": "general | inquiry | find_documents | send_email",
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
            "generated_llm_prompt": "Greet the user politely and offer assistance with WHO situation reports.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "What is the current case fatality rate for Ebola in the DRC?",
        "has_selected_documents": False,
        "chat_messages": [],
        "output": {
            "type": "inquiry",
            "generated_search_queries": [
                "Ebola case fatality rate DRC",
                "Ebola virus disease mortality Democratic Republic of Congo",
                "Ebola outbreak DRC epidemiological data",
                "WHO Ebola DRC deaths and cases",
            ],
            "generated_llm_prompt": "Explain the current case fatality rate for Ebola in the DRC based on the latest WHO situation reports.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Find situation reports about cholera outbreaks in East Africa.",
        "has_selected_documents": False,
        "chat_messages": [],
        "output": {
            "type": "find_documents",
            "generated_search_queries": [
                "cholera outbreak East Africa",
                "cholera situation report Kenya Uganda Tanzania",
                "WHO cholera response East Africa",
                "cholera epidemiological update East Africa",
            ],
            "generated_llm_prompt": "Retrieve relevant situation reports about cholera outbreaks in East Africa and list them with short summaries.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Summarize the selected reports about mpox response.",
        "has_selected_documents": True,
        "chat_messages": [],
        "output": {
            "type": "inquiry",
            "generated_search_queries": [],
            "generated_llm_prompt": "Summarize the retrieved situation reports.",
            "depend_on_last_task": False,
            "scope": "selected_documents",
        },
    },
    {
        "query": "And what about the vaccination coverage?",
        "has_selected_documents": False,
        "chat_messages": [
            HumanMessage(content="What is the current case fatality rate for Ebola in the DRC?"),
            AIMessage(
                content="The case fatality rate for Ebola in the DRC is approximately 66% according to recent reports."
            ),
            HumanMessage(content="How many confirmed cases were reported last month?"),
            AIMessage(
                content="The latest situation report indicates 45 confirmed cases in the past month."
            ),
        ],
        "output": {
            "type": "inquiry",
            "generated_search_queries": [
                "Ebola vaccination coverage DRC",
                "Ebola vaccine deployment Democratic Republic of Congo",
                "WHO Ebola immunization campaign DRC",
            ],
            "generated_llm_prompt": "Answer the user's follow-up about Ebola vaccination coverage in the DRC.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Can you also find ones about yellow fever?",
        "has_selected_documents": False,
        "chat_messages": [
            HumanMessage(content="Find situation reports about cholera outbreaks in East Africa."),
            AIMessage(content="I found 3 situation reports related to cholera outbreaks in East Africa."),
            HumanMessage(content="List them with short summaries."),
            AIMessage(content="Here are summaries of the three situation reports."),
        ],
        "output": {
            "type": "find_documents",
            "generated_search_queries": [
                "yellow fever outbreak East Africa",
                "yellow fever situation report WHO",
                "yellow fever epidemiological update Africa",
            ],
            "generated_llm_prompt": "Retrieve situation reports about yellow fever and list them with summaries.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Now summarize these reports.",
        "has_selected_documents": True,
        "chat_messages": [
            HumanMessage(content="Find reports about COVID-19 variants of concern."),
            AIMessage(content="I found 2 situation reports related to COVID-19 variants."),
            HumanMessage(content="Show me the list."),
            AIMessage(content="Here are the report titles and IDs."),
        ],
        "output": {
            "type": "inquiry",
            "generated_search_queries": [],
            "generated_llm_prompt": "Summarize the retrieved situation reports.",
            "depend_on_last_task": False,
            "scope": "selected_documents",
        },
    },
    {
        "query": "By the way, what kind of reports can you help me with?",
        "has_selected_documents": False,
        "chat_messages": [
            HumanMessage(content="What is the current case fatality rate for Ebola in the DRC?"),
            AIMessage(
                content="The case fatality rate is approximately 66% based on recent WHO reports."
            ),
            HumanMessage(content="Thanks, that helps!"),
            AIMessage(
                content="Glad to help. Would you like more details on the outbreak response?"
            ),
        ],
        "output": {
            "type": "general",
            "generated_search_queries": [],
            "generated_llm_prompt": "Respond in a friendly way and mention you are focused on helping with WHO situation reports.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "No, I mean Sudan not South Sudan.",
        "has_selected_documents": False,
        "chat_messages": [
            HumanMessage(
                content="Summarize reports about malaria control in South Sudan."
            ),
            AIMessage(content="Here is a summary of malaria control efforts in South Sudan."),
            HumanMessage(content="No, I mean Sudan not South Sudan."),
            AIMessage(content="Got it, Sudan instead of South Sudan. Let me adjust."),
        ],
        "output": {
            "type": "inquiry",
            "generated_search_queries": [
                "malaria control Sudan",
                "malaria situation report Sudan WHO",
                "malaria prevention strategy Sudan",
            ],
            "generated_llm_prompt": "Summarize the retrieved situation reports.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Explain the WHO risk assessment for avian influenza H5N1.",
        "has_selected_documents": False,
        "chat_messages": [
            HumanMessage(content="Hi there!"),
            AIMessage(
                content="Hello! I can help you with WHO situation reports and public health data. What would you like to know?"
            ),
            HumanMessage(
                content="Explain the WHO risk assessment for avian influenza H5N1."
            ),
            AIMessage(content="Sure, let me look that up."),
        ],
        "output": {
            "type": "inquiry",
            "generated_search_queries": [
                "WHO risk assessment avian influenza H5N1",
                "H5N1 human transmission risk WHO",
                "avian influenza H5N1 situation report",
                "H5N1 public health risk evaluation",
            ],
            "generated_llm_prompt": "Explain the WHO risk assessment for avian influenza H5N1 using the retrieved situation reports.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Summarize the selected reports about measles in West Africa.",
        "has_selected_documents": True,
        "chat_messages": [
            HumanMessage(content="Find reports about measles outbreaks in West Africa."),
            AIMessage(content="I found two situation reports related to measles in West Africa."),
            HumanMessage(
                content="Summarize the selected reports about measles in West Africa."
            ),
            AIMessage(content="Okay, I will summarize only the selected reports."),
        ],
        "output": {
            "type": "inquiry",
            "generated_search_queries": [],
            "generated_llm_prompt": "Summarize only the selected situation reports about measles in West Africa.",
            "depend_on_last_task": False,
            "scope": "selected_documents",
        },
    },
    {
        "query": "Can you email me that summary?",
        "has_selected_documents": False,
        "chat_messages": [
            HumanMessage(content="Summarize the WHO situation report on Ebola in the DRC."),
            AIMessage(
                content="Here is a summary of the WHO situation report on Ebola in the DRC."
            ),
            HumanMessage(content="Can you email me that summary?"),
            AIMessage(content="Sure, I can send that summary by email."),
        ],
        "output": {
            "type": "send_email",
            "generated_search_queries": [],
            "generated_llm_prompt": "Prepare to send the conversation summary via email.",
            "depend_on_last_task": True,
            "scope": "generic",
        },
    },
    {
        "query": "Send the report summary to my colleague by email",
        "has_selected_documents": False,
        "chat_messages": [],
        "output": {
            "type": "send_email",
            "generated_search_queries": [],
            "generated_llm_prompt": "Prepare to send the summary via email.",
            "depend_on_last_task": False,
            "scope": "generic",
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