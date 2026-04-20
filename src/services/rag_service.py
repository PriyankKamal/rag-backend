from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings()

llm = ChatMistralAI(model="mistral-small-2506")

promt = ChatPromptTemplate(
    [
        ("system","Use only provided context. If answer missing say I could not find the answer in the context."),
        ("human","Context:\n{context} \n\n Question:\n{question}")
    ]
)


def ask_rag(question):

    vector_store = Chroma(
    persist_directory="chroma",
    embedding_function=embedding_model
    )
    
    retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":3,
        "fetch_k":10,
        "lambda_mult":0.5
        }
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    final_promt = promt.invoke({
        "context":context,
        "question":question
    })

    response = llm.invoke(final_promt)

    return response.content