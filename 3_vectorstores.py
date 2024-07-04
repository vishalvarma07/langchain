from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)

print(vectorstore.similarity_search_with_relevance_scores("animals for kids"))

print('----------')
embedding = OpenAIEmbeddings().embed_query("cat")

print(vectorstore.similarity_search_by_vector_with_relevance_scores(embedding))

print('----------')

from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain.globals import set_debug, get_debug
import warnings
warnings.filterwarnings("ignore", message="Importing debug from langchain root module is no longer supported.")

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=2)  # select top result

print(retriever.batch(["cat", "shark"]))

print('-------')

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1},
)

print(retriever.batch(["cat", "shark"]))