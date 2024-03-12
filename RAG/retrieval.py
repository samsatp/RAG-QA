from pydantic import BaseModel
from pprint import pprint
from typing import List, Tuple
import sys, yaml, os
import pandas as pd

from RAG.indexing import get_vectorstore
from RAG import DF_COL_NAMES
from utils import get_context_chunks, get_chunk_id
from langchain_core.documents.base import Document

class Config(BaseModel):
    query_file: str # xlsx file
    k: int

config_path = os.path.join('RAG','config.yaml')
config = yaml.load(open(config_path,"r"), Loader=yaml.FullLoader)['retrieval']
print('RETRIEVAL CONFIG')
pprint(config)
config = Config(**config)

def get_relevant_documents(query: str, k:int)->Tuple[List[Document], List[str]]:
    vectorstore = get_vectorstore()
    documents = vectorstore.similarity_search(query, k, filter=None)
    doc_ids = [get_chunk_id(doc.page_content, vectorstore) for doc in documents]
    return documents, doc_ids

def process_query(q: str)->str:
    return q

def process_doc(doc: Document)->str:
    return doc.page_content

def rerank_docs(docs: List[str], doc_ids: List[str])->List[str]:
    return docs, doc_ids

def main(n_queries: int = None):
    df = pd.read_excel(config.query_file)
    if n_queries:
        df = df.sample(n=n_queries)
    queries = df[DF_COL_NAMES.questions.value].apply(process_query)
    print(f'total queries: {len(queries)}')

    # retrieve relevant docs for each question
    relevant_docs_list = []
    doc_ids_list = []
    for i, query in enumerate(queries.values): 
        relevant_docs, doc_ids = get_relevant_documents(query, k=config.k)
        relevant_docs = [process_doc(_) for _ in relevant_docs]
        relevant_docs, doc_ids = rerank_docs(relevant_docs, doc_ids)

        relevant_docs_list.append(relevant_docs)
        doc_ids_list.append(doc_ids)

        print(f'{i} queries processed')

    # attach context chunks
    if DF_COL_NAMES.context_chunk_ids.value not in df.columns:
        vectorstore = get_vectorstore()
        mapping = get_context_chunks(vectorstore)
        print([mapping[_] for _ in df[DF_COL_NAMES.question_ids.value]])
        df[DF_COL_NAMES.context_chunk_ids.value] = df[DF_COL_NAMES.question_ids.value].apply(lambda qid:mapping.get(str(qid)))

    df[DF_COL_NAMES.retrieved_docs.value] = relevant_docs_list
    df[DF_COL_NAMES.retrieved_doc_ids] = doc_ids_list

    if n_queries:
        path, filename = os.path.split(config.query_file)
        filename = filename.split('.')[0]
        path = os.path.join(path, filename+f'_{n_queries}.xlsx')
        df.to_excel(path, index=False)
    else:
        df.to_excel(config.query_file, index=False)
    print(f"{len(relevant_docs_list)} retrieved")

if __name__ == '__main__':
    n_queries = sys.argv[1]
    main()

    