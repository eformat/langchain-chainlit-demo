import os
from openai import AsyncOpenAI
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="foo")
cl.instrument_openai()


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


@cl.on_message
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **settings
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

#    message_history.append({"role": "assistant", "content": msg.content})
#    await msg.update()


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
