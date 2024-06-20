import dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
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
			content="PDFを選択してください。",
			# PDFファイルを指定する
			accept=[".pdf"],
			# タイムアウトなし
			raise_on_timeout=False,
		).send()

	file = files[0]

	# アップロードされたファイルのパスから中身を読み込む。
	documents = PyMuPDFLoader(file.path).load()
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

	await cl.Message(content=result, author="Answer").send()
