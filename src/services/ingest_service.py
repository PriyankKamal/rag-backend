import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def ingest_pdf(file):
    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    # Mistral Embeddings
    embedding_model = MistralAIEmbeddings(model="mistral-embed")

    # Store in Chroma
    vector_store = Chroma.from_documents(
        documents=chunks,
        persist_directory="chroma",
        embedding=embedding_model
    )

    return file.filename