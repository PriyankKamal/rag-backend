import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def ingest_pdf(file):
    os.makedirs("uploads",exist_ok=True)

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings()

    vector_store = Chroma.from_documents(
        documents = chunks,
        persist_directory="chroma",
        embedding=embedding_model
    )

    return file.filename