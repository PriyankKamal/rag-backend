from fastapi import UploadFile
from src.services.ingest_service import ingest_pdf
from src.services.rag_service import ask_rag
from src.schema.chat import Chat



def upload_pdf(file:UploadFile):

    filename = ingest_pdf(file)

    return {
        "success": True,
        "message": f"{filename} uploaded successfully"
    }
    

def ask_chat(question:Chat):
    answer = ask_rag(question)

    return {
        "success": True,
        "answer": answer
    }

