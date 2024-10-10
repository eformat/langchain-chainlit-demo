import os
import re
import httpx
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.retrievers import MergerRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import PGVector


MODEL_NAME = os.getenv("MODEL_NAME", "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf")
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL", "http://localhost:8080/v1")

DB_CONNECTION_STRING = os.getenv(
    "DB_CONNECTION_STRING",
    "postgresql+psycopg://postgres:password@localhost:5432/vectordb",
)

template = "Q: {question} A:"

if re.search(r"LLama-3", MODEL_NAME, flags=re.IGNORECASE):
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    You are a helpful, respectful and honest assistant answering questions named HatBot.
    You will be given a question you need to answer, and a context to provide you with information. You must answer the question based as much as possible on this context.
    Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Context:
    {context}

    Question: {question}<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    """

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["question"], template=template)

@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )
    settings = await cl.ChatSettings(
        [
            Select(
                id="model_name",
                label="OpenAI - Model",
                values=["Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"],
                initial_index=0,
            ),
            Select(
                id="collection_version",
                label="Quarkus Document Version",
                values=["3.15.1-Latest", "Main-SNAPSHOT", "3.8", "3.2", "2.16", "2.13", "2.7"],
                initial_index=0,
            ),
            Slider(
                id="temperature",
                label="OpenAI - Temperature",
                initial=0.2,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="top_p",
                label="Top P",
                initial=0.3,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="max_tokens",
                label="Max output tokens",
                initial=4096,
                min=0,
                max=32768,
                step=256,
            ),
            Slider(
                id="presence_penalty",
                label="Presence Penalty",
                initial=1.05,
                min=-2,
                max=2,
                step=0.05,
            ),
        ],
    ).send()
    cl.user_session.set("settings", settings)

    # memory=conversation_memory

    # Document store: pgvector vector store
    embeddings = HuggingFaceEmbeddings()
    store_docs = PGVector(
        connection_string=DB_CONNECTION_STRING,
        collection_name="docs",
        embedding_function=embeddings,
        use_jsonb=True,
    )
    store_main = PGVector(
        connection_string=DB_CONNECTION_STRING,
        collection_name="main",
        embedding_function=embeddings,
        use_jsonb=True,
    )
    store_latest = PGVector(
        connection_string=DB_CONNECTION_STRING,
        collection_name="latest",
        embedding_function=embeddings,
        use_jsonb=True,
    )
    store_213 = PGVector(
        connection_string=DB_CONNECTION_STRING,
        collection_name="213",
        embedding_function=embeddings,
        use_jsonb=True,
    )
    store_216 = PGVector(
        connection_string=DB_CONNECTION_STRING,
        collection_name="216",
        embedding_function=embeddings,
        use_jsonb=True,
    )
    store_27 = PGVector(
        connection_string=DB_CONNECTION_STRING,
        collection_name="27",
        embedding_function=embeddings,
        use_jsonb=True,
    )
    store_38 = PGVector(
        connection_string=DB_CONNECTION_STRING,
        collection_name="38",
        embedding_function=embeddings,
        use_jsonb=True,
    )
    store_32 = PGVector(
        connection_string=DB_CONNECTION_STRING,
        collection_name="32",
        embedding_function=embeddings,
        use_jsonb=True,
    )
    cl.user_session.set("store_docs", store_docs)
    cl.user_session.set("store_main", store_main)
    cl.user_session.set("store_latest", store_latest)
    cl.user_session.set("store_213", store_213)
    cl.user_session.set("store_216", store_216)
    cl.user_session.set("store_27", store_27)
    cl.user_session.set("store_38", store_38)
    cl.user_session.set("store_32", store_32)


class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.msg = cl.Message(content="")

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.msg.stream_token(token)

    async def on_llm_end(self, response: str, **kwargs):
        await self.msg.send()
        self.msg = cl.Message(content="")

def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        line = item.metadata['source']
        line = re.sub(r"http://0.0.0.0:4000", "https://quarkus.io", line)
        if line not in unique_list:
            unique_list.append(line)
    return unique_list

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)


@cl.on_message
async def query_llm(message: cl.Message):
    settings = cl.user_session.get("settings")

    llm = ChatOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=INFERENCE_SERVER_URL,
        model_name=settings["model_name"],
        top_p=settings["top_p"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        presence_penalty=settings["presence_penalty"],
        streaming=True,
        verbose=False,
        http_async_client=httpx.AsyncClient(verify=False),
        http_client=httpx.Client(verify=False),
    )

    store_docs = cl.user_session.get("store_docs")
    store_main = cl.user_session.get("store_main")
    store_latest = cl.user_session.get("store_latest")
    store_213 = cl.user_session.get("store_213")
    store_216 = cl.user_session.get("store_216")
    store_27 = cl.user_session.get("store_27")
    store_38 = cl.user_session.get("store_38")
    store_32 = cl.user_session.get("store_32")

    retriever_docs = store_docs.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    retriever_main = store_main.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    retriever_latest = store_latest.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    retriever_213 = store_213.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    retriever_216 = store_216.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    retriever_27 = store_27.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    retriever_38 = store_38.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    retriever_32 = store_32.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    lotr = None
    if settings["collection_version"] == "3.15.1-Latest":
        lotr = MergerRetriever(retrievers=[retriever_docs, retriever_latest])
    elif settings["collection_version"] == "2.13":
        lotr = MergerRetriever(retrievers=[retriever_docs, retriever_213])
    elif settings["collection_version"] == "2.16":
        lotr = MergerRetriever(retrievers=[retriever_docs, retriever_216])
    elif settings["collection_version"] == "2.7":
        lotr = MergerRetriever(retrievers=[retriever_docs, retriever_27])
    elif settings["collection_version"] == "3.8":
        lotr = MergerRetriever(retrievers=[retriever_docs, retriever_38])
    elif settings["collection_version"] == "3.2":
        lotr = MergerRetriever(retrievers=[retriever_docs, retriever_32])
    else:
        lotr = MergerRetriever(retrievers=[retriever_docs, retriever_main])

    llm_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=lotr,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
        verbose=True,
    )

    resp = await llm_chain.acall(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler(), StreamHandler()]
    )

    msg = cl.Message(content="")
    sources = remove_source_duplicates(resp['source_documents'])
    if len(sources) != 0:
        await msg.stream_token("\n*Sources:* \n")
        for source in sources:
            await msg.stream_token("* " + str(source) + "\n")
    await msg.send()
