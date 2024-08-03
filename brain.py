from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

import ollama
import os
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv\

from langchain_community.document_loaders import TextLoader

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter



def processing_embedding(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.create_documents([text])
    
    vector_store = Chroma.from_documents(documents, OllamaEmbeddings(model="nomic-embed-text"))
    retriever = vector_store.as_retriever()

    load_dotenv()
    llm = ChatOllama(
        model="llama3",
        temperature=0,
    )
    return retriever, llm

def answer_question(retriever, llm, question):
    rag_template = """Use the following context to answer the question in the most straightforward way without using much words:
    {context}
    Question: {question}
    """

    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever, chain_type_kwargs={"prompt": rag_prompt},
    )
    

    response = qa_chain.invoke(question)
    return response


