from pydantic import BaseModel
from pprint import pprint
from typing import List, Tuple
import sys, yaml, os
import pandas as pd

from RAG.indexing import get_vectorstore
from RAG import DF_COL_NAMES, retrieval_database, indexing_database
from utils import get_context_chunks, get_chunk_id
from langchain_core.documents.base import Document
from langchain.vectorstores.base import VectorStore


class Config(BaseModel):
    question_file: str
    collection_name: str
    k: int
 

def get_relevant_documents(query: str, 
                           k: int,
                           vectorstore: VectorStore 
                           )->Tuple[List[Document], List[str]]:
    if vectorstore._embedding_function.model_name == 'BAAI/bge-base-en-v1.5':
        query = 'Represent this sentence for searching relevant passages: ' + query
    documents = vectorstore.similarity_search(query, k, filter=None)
    doc_ids = [get_chunk_id(doc.page_content, vectorstore) for doc in documents]
    return documents, doc_ids

def process_query(q: str)->str:
    return q

def process_doc(doc: Document)->str:
    return doc.page_content

def rerank_docs(docs: List[str], doc_ids: List[str])->List[str]:
    return docs, doc_ids

def main(frac:float):

    config_path = os.path.join('RAG','config','retrieval.yaml')
    config_dict = yaml.load(open(config_path,"r"), Loader=yaml.FullLoader)
    print('RETRIEVAL CONFIG')
    pprint(config_dict)
    config = Config(**config_dict)

    
    # fetch indexing metadata, so the vectorstore config is consistant
    indexing_meta = indexing_database.get(key=config.collection_name)
    if not indexing_meta['status']:
        print('using failed collection:',indexing_meta['collection'])
        return
    vectorstore = get_vectorstore(chroma_collection_name=config.collection_name,
                                  distance_fn=indexing_meta['distance_fn'],
                                  embedding_model=indexing_meta['embedding_model'])
    
    meta = {'embedding_function':vectorstore._embedding_function.model_name,
            'collection_meta':vectorstore._collection.metadata,
            }
    
    df = pd.read_excel(config.question_file).sample(frac=frac)
    queries = df[DF_COL_NAMES.questions.value].apply(process_query)
    print(f'total queries: {len(queries)}')

    # retrieve relevant docs for each question
    relevant_docs_list = []
    doc_ids_list = []
    
    for i, query in enumerate(queries.values): 
        relevant_docs, doc_ids = get_relevant_documents(query=query,
                                                        k=config.k,
                                                        vectorstore=vectorstore
                                                        )
        relevant_docs = [process_doc(_) for _ in relevant_docs]
        relevant_docs, doc_ids = rerank_docs(relevant_docs, doc_ids)

        relevant_docs_list.append(relevant_docs)
        doc_ids_list.append(doc_ids)

        if i%500==0:
            print(f'{i} queries processed')

    # attach context chunks
    """ if DF_COL_NAMES.context_chunk_ids.value not in df.columns:
        vectorstore = get_vectorstore(collection_name)
        mapping = get_context_chunks(vectorstore)
        print([mapping[_] for _ in df[DF_COL_NAMES.question_ids.value]])
        df[DF_COL_NAMES.context_chunk_ids.value] = df[DF_COL_NAMES.question_ids.value].apply(lambda qid:mapping.get(str(qid)))
     """
    df[DF_COL_NAMES.retrieved_docs.value] = relevant_docs_list
    df[DF_COL_NAMES.retrieved_doc_ids.value] = doc_ids_list


    new_id = retrieval_database.new_entry(**config.model_dump(), **meta)
    result_path = os.path.join('RAG','results',f'{new_id}.xlsx')
    df.to_excel(result_path, index=False)
    print(f"{len(relevant_docs_list)} retrieved")

    retrieval_database.update(key=new_id, column='status', value=1)
    retrieval_database.update(key=new_id, column='n_retrieval', value=len(relevant_docs_list))

if __name__ == '__main__':
    main()

    