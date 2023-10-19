# For cloud deployment
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# For local deployment
# from dotenv import load_dotenv
# load_dotenv()

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
# from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
import tempfile
import os

# Title
st.title("ChatPDF")
st.write("---")

# File Upload
uploaded_file = st.file_uploader("upload a PDF file", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    # Loader
    # loader = PyPDFLoader("unsu.pdf")
    # pages = loader.load_and_split()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    # Embedding
    db = Chroma.from_documents(texts, OpenAIEmbeddings())

    #Stream 받아 줄 Hander 만들기
    from langchain.callbacks.base import BaseCallbackHandler
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text+=token
            self.container.markdown(self.text)
            
    # Question
    #question = "아내가 무슨 음식을 먹고 싶어해?"
    st.header("PDF 문서 관련 질문해 보세요.")
    question = st.text_input('질문은...')

    # llm = ChatOpenAI(temperature=0)
    # retriever_from_llm = MultiQueryRetriever.from_llm(
    #     retriever=db.as_retriever(), llm=llm
    # )

    # docs = retriever_from_llm.get_relevant_documents(query=question)
    # print(len(docs))
    # print(docs)

    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True, callbacks=[stream_handler])
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            qa_chain({"query": question})
            #result = qa_chain({"query": question})
            #st.write(result["result"])