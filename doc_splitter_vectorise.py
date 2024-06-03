from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
from langchain.vectorstores import FAISS
from langchain.vectorstores import DocArrayInMemorySearch
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
persist_directory = 'docs/chroma/'

embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# genai.configure(api_key='')
# embeddings = GoogleGenerativeAIEmbeddings(google_api_key = '', model = "models/embedding-001")



text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=26, 
    separator = '\n'
)

def doc_splitter(text):
    global docs
    docs = text_splitter.split_text(text)
    print("len of docs", len(docs))
    return docs


def get_pdf_docs(pdf_docs):
    print(pdf_docs)
    for pdf in pdf_docs:
        print(pdf.name)
        loader = PyPDFLoader(pdf.upload_url)
        pages = loader.load()
    docs = doc_splitter(pages)
    return docs


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  doc_splitter(text)


def get_vectordb(docs):
    global vectordb
    # vectordb = FAISS.from_texts(docs, embedding=embeddings)
    # vectordb.save_local("faiss_index")
    # vectordb = DocArrayInMemorySearch.from_texts(docs, embeddings)
    print("In vector db initialization")
    vectordb = Chroma.from_texts(
    texts=docs,
    embedding=embeddings,
    persist_directory=persist_directory
    )
    vectordb.persist()
    print(vectordb._collection.count())
    return vectordb

