from langchain_ollama import ChatOllama
from model.domain.core import Conversation, GraphState
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


async def general(state: GraphState, text_llm: ChatOllama, general_chat_prompt: str):

    messages = [("system", general_chat_prompt)]
    
    if state.historical_summary:
        messages.append(("system", "You will be provided with Historical Summary for the conversations")) 
        messages.append(("system", "This is the Historical Summary :\n{historical_summary}")) 
    
    if state.chat_messages:
        messages.append(("system", f"The following are the latest {len(state.chat_messages)} chat messages:"))
        messages.extend(state.chat_messages)
        messages.append(("system", "End of Chat messages"))
        
    messages.append(("user", state.user_input.query))

    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | text_llm | StrOutputParser()

    # Collect streamed tokens
    ai_text = []
    async for chunk in chain.astream(
        {"historical_summary": state.historical_summary or ""}
    ):
        ai_text.append(chunk)

    full_reply = "".join(ai_text)

    state.last_conversation = Conversation(
        task=state.task,
        documents=None,
        request=HumanMessage(content=state.user_input.query),
        response=AIMessage(content=full_reply),
    )
    state.chat_messages = [
        HumanMessage(content=state.user_input.query),
        AIMessage(content=full_reply),
    ]
    return state
