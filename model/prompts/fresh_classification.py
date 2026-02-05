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
You classify intent for an assistant that uses WHO situation reports.
</SYSTEM>

<INSTRUCTIONS>
Return ONLY valid JSON that matches the schema. No extra text.
Decide intent:
- "general": greetings, small talk, meta questions about capabilities.
- "inquiry": asks to answer/explain/summarize/solve using WHO situation reports, epidemiological data, or public health guidance.
- "find_documents": asks to find/locate/search/list/filter situation reports (lists only; no LLM summary).
- "send_email": user wants to email/send a summary, report, or conversation content to someone.

Rules:
- Produce 2–4 precise search queries for vector/semantic search; use [] for "general".
- Scope is "selected_documents" if:
  1) has_selected_documents is True, or
  2) the user clearly asks to use selected docs (e.g., "from selected", "these files").
- If scope="selected_documents" AND the request is summarize/explain/compare/extract (no need to find more docs), then set "generated_search_queries":[] so the next step will use only the selected document IDs.
- If user asks for "selected documents" but no IDs are provided, set scope="selected_documents", return "generated_search_queries":[] and make generated_llm_prompt ask for IDs before continuing.
- "depend_on_last_task"=false in this case.
- "generated_llm_prompt":
  * For summarization requests (whether with queries or selected docs) ALWAYS: "Summarize the retrieved situation reports."
  * Otherwise: one clear instruction ≤80 words.
- If unsure between "inquiry" and "find_documents", prefer "find_documents" when verbs like find, locate, search, list, fetch, retrieve, filter are used without summarization/explanation. Otherwise choose "inquiry".
- Output keys must appear exactly in the schema order.
</INSTRUCTIONS>

<SCHEMA>
{
  "type": "general | inquiry | find_documents | send_email",
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
            "generated_llm_prompt": "Greet the user briefly and offer help with WHO situation reports.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "What is the current case fatality rate for Ebola in the DRC?",
        "has_selected_documents": False,
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
        "output": {
            "type": "find_documents",
            "generated_search_queries": [
                "cholera outbreak East Africa",
                "cholera situation report Kenya Uganda Tanzania",
                "WHO cholera response East Africa",
                "cholera epidemiological update East Africa",
            ],
            "generated_llm_prompt": "Retrieve the most relevant situation reports on cholera outbreaks in East Africa and list each with a one-line summary.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Summarize the selected reports about mpox response.",
        "has_selected_documents": True,
        "output": {
            "type": "inquiry",
            "generated_search_queries": [],
            "generated_llm_prompt": "Summarize the retrieved situation reports.",
            "depend_on_last_task": False,
            "scope": "selected_documents",
        },
    },
    {
        "query": "Summarize the selected report.",
        "has_selected_documents": False,
        "output": {
            "type": "inquiry",
            "generated_search_queries": [],
            "generated_llm_prompt": "Ask the user to provide the document IDs to summarize before proceeding.",
            "depend_on_last_task": False,
            "scope": "selected_documents",
        },
    },
    {
        "query": "Compare the selected reports and highlight differences in transmission rates.",
        "has_selected_documents": True,
        "output": {
            "type": "inquiry",
            "generated_search_queries": [],
            "generated_llm_prompt": "Summarize the retrieved situation reports.",
            "depend_on_last_task": False,
            "scope": "selected_documents",
        },
    },
    {
        "query": "Do you enjoy reading health reports?",
        "has_selected_documents": False,
        "output": {
            "type": "general",
            "generated_search_queries": [],
            "generated_llm_prompt": "Respond in a friendly way and explain that you can help with WHO situation reports and public health questions.",
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
        "query": "Hi, can you tell me about the WHO response to avian influenza H5N1?",
        "has_selected_documents": False,
        "output": {
            "type": "inquiry",
            "generated_search_queries": [
                "WHO avian influenza H5N1 response",
                "H5N1 outbreak situation report",
                "avian influenza public health measures WHO",
                "H5N1 human transmission risk assessment",
            ],
            "generated_llm_prompt": "Greet the user briefly, then explain the WHO response to avian influenza H5N1 based on situation reports.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Find reports about measles vaccination campaigns and explain the coverage.",
        "has_selected_documents": False,
        "output": {
            "type": "find_documents",
            "generated_search_queries": [
                "measles vaccination campaign WHO",
                "measles immunization coverage situation report",
                "WHO measles outbreak response vaccination",
                "measles vaccine rollout progress",
            ],
            "generated_llm_prompt": "Locate situation reports on measles vaccination campaigns and, after retrieving them, provide an explanation of the coverage.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Summarize reports about malaria control efforts in Sudan",
        "has_selected_documents": False,
        "output": {
            "type": "inquiry",
            "generated_search_queries": [
                "malaria control Sudan WHO",
                "malaria situation report Sudan",
                "malaria prevention strategy Sudan",
                "WHO malaria response Sudan epidemiological data",
            ],
            "generated_llm_prompt": "Summarize the retrieved situation reports.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Send me the summary by email",
        "has_selected_documents": False,
        "output": {
            "type": "send_email",
            "generated_search_queries": [],
            "generated_llm_prompt": "Prepare to send the conversation summary via email.",
            "depend_on_last_task": False,
            "scope": "generic",
        },
    },
    {
        "query": "Email the last report summary to my colleague",
        "has_selected_documents": False,
        "output": {
            "type": "send_email",
            "generated_search_queries": [],
            "generated_llm_prompt": "Prepare to send the summary via email.",
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
