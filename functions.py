import requests

from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter,TokenTextSplitter,RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
import numpy as np


client = OpenAI()



def get_conversation_chain(documents, query):
    embeddings = OpenAIEmbeddings()
    # Prepare documents for FAISS
    docs = [Document(page_content=doc["text"], embedding=doc["embedding"]) for doc in documents]

    # Create a vector store from the documents
    vector_store = FAISS.from_documents(docs,embedding=embeddings)
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    conversation_chain.run(query)
    return conversation_chain


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=500,
        chunk_overlap=0,
    )

    texts = text_splitter.split_text(text)

    return texts

def embedder(input_text):
    client = OpenAI()
    embeddings=client.embeddings.create(
    model="text-embedding-ada-002",
    input=input_text,
    encoding_format="float"
    )
    return embeddings.data[0].embedding