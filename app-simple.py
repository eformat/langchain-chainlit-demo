import os
from langchain_community.llms import CTransformers
from accelerate import Accelerator
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from openai import AsyncOpenAI
import chainlit as cl

config = {
    "max_new_tokens": 512,
    "repetition_penalty": 1.1,
    "context_length": 2048,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 0.9,
    "gpu_layers": 50,
    "stream": True,
    "threads": int(os.cpu_count() / 2),
}

accelerator = Accelerator()
llm = CTransformers(model=local_llm, model_type="llama3", config=config)
llm_init, config = accelerator.prepare(llm, config)
print(llm_init)

# query = "What is the meaning of Life?"

# result = llm_init(query)

# print(result)

template = """Question: {question}

# Answer: You are helpful teacher that easily explain complex topics in easy way.
# """


@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm_init, verbose=True)
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res["text"]).send()
