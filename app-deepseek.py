from langchain_deepseek.chat_models import ChatDeepSeek
from rich import print
import os

llm = ChatDeepSeek(
    model = '/models/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf',
    temperature=0,
    api_base="http://localhost:8080/v1",
    api_key="fake"
)

messages = [
    {"role": "system", "content": "You are an assistant."},
    {"role": "system", "content": "no-thinking"},
    #{"role": "user", "content": "9.11 and 9.8, which is greater? Explain the reasoning behind this decision."}
    {"role": "user", "content": "how many r's in strawberry? Explain the reasoning behind this decision"}
]

for chunk in llm.stream(messages):
    print(chunk.text(), end="")

#response = llm.invoke(messages, extra_body={"include_reasoning": True})
#print(response.content)
#print(f"REASONING: {response.additional_kwargs.get('reasoning_content', '')}")
#print(response)