from pydantic import BaseModel
from pprint import pprint
from typing import List
import sys, yaml, os
import pandas as pd

from RAG.indexing import get_vectorstore
from RAG import DF_COL_NAMES
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

def rerank_docs(docs: List[str])->List[str]:
    return docs

def main():
    df = pd.read_excel(config.query_file)
    queries = df[DF_COL_NAMES.questions.value].apply(process_query)
    print(f'total queries: {len(queries)}')
    relevant_docs_list = []
    for i, query in enumerate(queries.values): 
        relevant_docs = get_relevant_documents(query, k=config.k)
        relevant_docs = [process_doc(_) for _ in relevant_docs]
        relevant_docs = rerank_docs(relevant_docs)
        relevant_docs_list.append(relevant_docs)

        print(f'{i} queries processed')
        
    df[DF_COL_NAMES.retrieved_docs.value] = relevant_docs_list
    df.to_excel(config.query_file, index=False)
    print(f"{len(relevant_docs_list)} retrieved")

if __name__ == '__main__':
    main()

    