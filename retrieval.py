from pydantic import BaseModel
from pprint import pprint
from typing import List
import sys, yaml
import pandas as pd

from indexing import get_vectorstore
from langchain_core.documents.base import Document

class Config(BaseModel):
    query_file: str # xlsx file
    k: int

config = yaml.load(open("retrieval.yaml","r"), Loader=yaml.FullLoader)
print('RETRIEVAL CONFIG')
pprint(config)
config = Config(**config)

def get_relevant_documents(query: str, k:int)->List[Document]:
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k, filter=None)

def process_query(q: str)->str:
    return q

def process_doc(doc: Document)->str:
    return doc


def main():
    df = pd.read_excel(config.query_file)
    queries = df['question'].apply(process_query)
    for query in queries.values: 
        relevant_docs = get_relevant_documents(query, k=config.k)
        relevant_docs = [process_doc(_) for _ in relevant_docs]
        
    df['relevant_docs'] = relevant_docs
    df.to_excel(config.query_file, index=False)
    print(f"{len(relevant_docs)} retrieved")

if __name__ == '__main__':
    main()

    