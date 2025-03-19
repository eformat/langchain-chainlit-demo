import os
import httpx
from openai import AsyncOpenAI
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

client = AsyncOpenAI(
    base_url="https://health-finetune-predictor-llama-serving.apps.sno.sandbox298.opentlc.com/v1",
    api_key="foo",
    http_client=httpx.AsyncClient(verify=False),
    )
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
                values=["mhepburn-health-care-031325"],
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
    message_history.pop(0)

    msg = cl.Message(content="")
    await msg.send()

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **settings
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
