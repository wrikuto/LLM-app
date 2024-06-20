import json
import ast
import os
import dotenv
from openai import AsyncOpenAI

import chainlit as cl

env_values = dotenv.dotenv_values()
api_key = env_values.get("OPENAI_API_KEY", "")
client = AsyncOpenAI(api_key=api_key)





@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "あなたは親切なアシスタントです。"}],
    )



@cl.step(type="llm")
async def call_gpt3(message_history):
    settings = {
        "model": "gpt-3.5-turbo",
    }

    cl.context.current_step.generation = cl.ChatGeneration(
        provider="openai-chat",
        messages=[
            cl.GenerationMessage(
                content=m["content"], name=m.get("name", "function"), role=m["role"]
            )
            for m in message_history
        ],
        settings=settings,
    )

    # 問題箇所
    response = await client.chat.completions.create(
        messages=message_history, **settings
    ) # type: ignore

    message = response.choices[0].message

    if message.content:
        cl.context.current_step.generation.completion = message.content
        cl.context.current_step.output = message.content


    return message


# ------------------------------------------------

@cl.on_message
async def run_conversation(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"name": "user", "role": "user", "content": message.content})


    message = await call_gpt3(message_history)
    if not message.tool_calls:
        await cl.Message(content=message.content, author="Answer").send()
