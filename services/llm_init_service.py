from langchain_ollama import ChatOllama, OllamaEmbeddings


def GetTextLLModle(model_name, temprature=0):
    return ChatOllama(model=model_name, temperature=temprature)


def GetInstructLLModle(model_name):
    return ChatOllama(
        model=model_name, disable_streaming=True, temperature=0, format="json"
    )


def GetEmbeddingModel(model_name):
    return OllamaEmbeddings(model=model_name)
