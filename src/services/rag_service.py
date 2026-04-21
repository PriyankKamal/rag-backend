from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Mistral Embeddings
embedding_model = MistralAIEmbeddings(model="mistral-embed")

# Chat Model
llm = ChatMistralAI(model="mistral-small-latest")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Use only provided context. If answer missing say: I could not find the answer in the context."
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])


def ask_rag(question):
    vector_store = Chroma(
        persist_directory="chroma",
        embedding_function=embedding_model
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(final_prompt)

    return response.content