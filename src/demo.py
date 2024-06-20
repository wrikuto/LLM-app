from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl
from dotenv import load_dotenv
from textblob import TextBlob

load_dotenv('../.env')




@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message:str):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.
    Args:
        message: The user's message.
    Returns:
        None.
    """

    # Call the tool
    # tool()

    # Send the final answer.

    result = message.title()

    if ("sentiment" in message):
        file = None
        while (file == None):
            file = cl.ask_for_file(title="Please a text file to analyse", accept=["text/plain"])

        # バイトをテキストにデコード
        text = file.content.decode("utf-8")
        blob = TextBlob(text)
        cl.send_message(content=f"Sure here analysis:{text}.\nyour result is {blob.sentiment}")

    await cl.Message(
        content=f"here is massage: {result}"
        ).send()




# @cl.on_chat_start
# def start():
#     content = "this is LLLM framework"
#     cl.send_message(content=content)
