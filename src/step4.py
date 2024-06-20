import dotenv
import os
import csv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
# from langchain_community.document_loaders import 
from langchain.text_splitter import SpacyTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains import (
    ConversationalRetrievalChain,
)
import chainlit as cl

env_values = dotenv.dotenv_values()

prompt = PromptTemplate(
    template="""
    文章を前提にして質問に答えてください。

    文章 :
    {document}

    質問 : {question}
    """,
    input_variables=["document", "question"],
)



@cl.on_chat_start
async def on_chat_start():
	"""初回起動時に呼び出される."""


	# PDFを読み込む処理
	files = None

	# awaitメソッドのために、whileを利用する。アップロードされるまで続く。
	while files is None:
		# chainlitの機能に、ファイルをアップロードさせるメソッドがある。
		files = await cl.AskFileMessage(
			# ファイルの最大サイズ
			max_size_mb=20,
			# ファイルをアップロードさせる画面のメッセージ
			content="ファイル(pdf, docx, xlsx)を選択してください。",
			# PDFファイルを指定する
			accept=[".pdf", ".docx", ".xlsx"],
			# タイムアウトなし
			raise_on_timeout=False,
		).send()

	file = files[0]
	fname_with_ext = os.path.basename(file.path)
	ext = os.path.splitext(fname_with_ext)[1]

	# アップロードされたファイルのパスから中身を読み込む。
	if (ext == ".pdf"):
		documents = PyMuPDFLoader(file.path).load()
	elif (ext == ".docx"):
		documents = Docx2txtLoader(file.path).load()
	elif (ext == ".xlsx" or ext == ".xls"):
		documents = UnstructuredExcelLoader(file.path).load()

    # PDFを分割する処理
	text_splitter = SpacyTextSplitter(chunk_size=400, pipeline="ja_core_news_sm")
	splitted_documents = text_splitter.split_documents(documents)
	# テキストをベクトル化するOpenAIのモデル
	embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=env_values.get("OPENAI_API_KEY", ""))
	# Chromaにembedding APIを指定して、初期化する。
	database = Chroma(embedding_function=embeddings)
	# PDFから内容を分割されたドキュメントを保存する。
	database.add_documents(splitted_documents)
	# 今回は、簡易化のためセッションに保存する。
	cl.user_session.set("data", database)




# @cl.on_message
# async def on_message(input_message: cl.Message):
# 	"""メッセージが送られるたびに呼び出される."""
# 	chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
# 	cb = cl.AsyncLangchainCallbackHandler()

# 	# チャット用のOpenAIのモデル
# 	open_ai = ChatOpenAI(api_key=env_values.get("OPENAI_API_KEY", ""), model="gpt-4") # type: ignore
# 	# セッションからベクトルストアを取得（この中にPDFの内容がベクトル化されたものが格納されている）
# 	database = cl.user_session.get("data")
# 	# 質問された文から似た文字列を、DBより抽出
# 	documents = database.similarity_search(input_message.content)
# 	# 抽出したものを結合
# 	documents_string = ""
# 	for document in documents:
# 		documents_string += f"""
# 		---------------------------------------------
# 		{document.page_content}
# 		"""

# 	# プロンプトに埋め込みながらOpenAIに送信
# 	result = open_ai(
# 		[
# 			HumanMessage(
# 				content=prompt.format(
# 					document = documents,
# 					question = input_message.content,
# 				)
# 			)
# 		]
# 	).content

# 	await cl.Message(content=result, author="Answer").send()
	



@cl.on_message
async def on_message(input_message: cl.Message):
	"""メッセージが送られるたびに呼び出される."""

	# チャット用のOpenAIのモデル
	open_ai = ChatOpenAI(api_key=env_values.get("OPENAI_API_KEY", ""), model="gpt-4") # type: ignore
	# セッションからベクトルストアを取得（この中にPDFの内容がベクトル化されたものが格納されている）
	database = cl.user_session.get("data")
	# 質問された文から似た文字列を、DBより抽出
	documents = database.similarity_search(input_message.content)
	# 抽出したものを結合
	documents_string = ""
	for document in documents:
		documents_string += f"""
		---------------------------------------------
		{document.page_content}
		"""

	# プロンプトに埋め込みながらOpenAIに送信
	result = open_ai(
		[
			HumanMessage(
				content=prompt.format(
					document = documents,
					question = input_message.content,
				)
			)
		]
	).content

	text_elements = []  # type: List[cl.Text]

	if documents:
		for source_idx, source_doc in enumerate(documents):
			source_name = f"source_{source_idx}"
			# Create the text element referenced in the message
			text_elements.append(
				cl.Text(content=source_doc.page_content, name=source_name)
			)
		source_names = [text_el.name for text_el in text_elements]

		if source_names:
			result += f"\n\nSources: {', '.join(source_names)}"
		else:
			result += "\nNo sources found"

	await cl.Message(content=result, author="Answer", elements=text_elements).send()
