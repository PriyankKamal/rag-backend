from fastapi import APIRouter, UploadFile, File
from src.schema.chat import Chat
from src.controller.chat_controller import upload_pdf, ask_chat

router = APIRouter()

@router.post("/upload")
def upload(file: UploadFile = File(...)):
    return upload_pdf(file)

@router.post("/ask")
def ask(question: Chat):
    return ask_chat(question.question)