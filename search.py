from pprint import pprint
from typing import List

from indexing import get_embedding, Config, get_vectorstore
from langchain_community.vectorstores.chroma import Chroma

from langchain_core.documents.base import Document


def get_relevant_documents(query: str, 
                           k:int)->List[Document]:
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k, filter=None)


def main():
    ...    