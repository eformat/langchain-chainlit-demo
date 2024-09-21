import os
import re
import httpx
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import PGVector


MODEL_NAME = os.getenv("MODEL_NAME", "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4096))
PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY", 1.03))

INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL", "http://localhost:8080/v1")
TOP_K = int(os.getenv("TOP_K", 10))
TOP_P = float(os.getenv("TOP_P", 0.95))
TYPICAL_P = float(os.getenv("TYPICAL_P", 0.95))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))

DB_CONNECTION_STRING = os.getenv(
    "DB_CONNECTION_STRING",
    "postgresql+psycopg://postgres:password@localhost:5432/vectordb",
)
DB_COLLECTION_NAME = os.getenv("DB_COLLECTION_NAME", "documents_test")

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


llm = ChatOpenAI(
    openai_api_key="EMPTY",
    openai_api_base=INFERENCE_SERVER_URL,
    model_name=MODEL_NAME,
    top_p=TOP_P,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    presence_penalty=PRESENCE_PENALTY,
    streaming=True,
    verbose=False,
    # callbacks=[QueueCallback(q)],
    http_async_client=httpx.AsyncClient(verify=False),
    http_client=httpx.Client(verify=False),
)

# Document store: pgvector vector store
embeddings = HuggingFaceEmbeddings()
store = PGVector(
    connection_string=DB_CONNECTION_STRING,
    collection_name=DB_COLLECTION_NAME,
    embedding_function=embeddings,
    use_jsonb=True,
)


@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="OpenAI - Model",
                values=["Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"],
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
        ],
    ).send()
    cl.user_session.set("settings", settings)

    llm_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
        verbose=True,
    )

    # memory=conversation_memory)

    cl.user_session.set("llm_chain", llm_chain)


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
        if item.metadata['source'] not in unique_list:
            unique_list.append(item.metadata['source'])
    return unique_list

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)


@cl.on_message
async def query_llm(message: cl.Message):
    settings = cl.user_session.get("settings")
    llm_chain = cl.user_session.get("llm_chain")

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
