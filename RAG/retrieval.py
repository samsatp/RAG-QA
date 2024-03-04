from pydantic import BaseModel
from pprint import pprint
from typing import List
import sys, yaml, os
import pandas as pd

from RAG.indexing import get_vectorstore
from langchain_core.documents.base import Document

class Config(BaseModel):
    query_file: str # xlsx file
    k: int

config_path = os.path.join('RAG','config','retrieval.yaml')
config = yaml.load(open(config_path,"r"), Loader=yaml.FullLoader)
print('RETRIEVAL CONFIG')
pprint(config)
config = Config(**config)

def get_relevant_documents(query: str, k:int)->List[Document]:
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k, filter=None)

def process_query(q: str)->str:
    return q

def process_doc(doc: Document)->str:
    return doc.page_content


def main():
    df = pd.read_excel(config.query_file)
    queries = df['question'].apply(process_query)

    relevant_docs_list = []
    for query in queries.values: 
        relevant_docs = get_relevant_documents(query, k=config.k)
        relevant_docs = [process_doc(_) for _ in relevant_docs]
        relevant_docs_list.append(relevant_docs)
        
    df['relevant_docs'] = relevant_docs_list
    df.to_excel(config.query_file, index=False)
    print(f"{len(relevant_docs_list)} retrieved")

if __name__ == '__main__':
    main()

    