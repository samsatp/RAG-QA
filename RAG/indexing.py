__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from typing import List, Any, Dict, Callable
from pydantic import BaseModel
from pprint import pprint
import chromadb
import yaml
import os
import re
import pickle

from langchain_core.documents.base import Document
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from utils import is_context_of
from RAG import indexing_database

class Config(BaseModel):
    data_dir: str
    distance_fn: str
    embedding_model: str
    splitter_kwargs: Dict[str, Any]


def preprocess_text(text: str)->str:
    
    # remove any big crump of strings
    text = re.sub(r"\S{20,}"," ",text)

    # remove citation numbers
    text = re.sub(r"\[(\d,?\s?-?)+\]"," ",text)
    
    # remove percentage in parenthses e.g. (20%)
    text = re.sub(r"\((\d+%;?\s?)+\)"," ",text)

    # remove everything in any parenthesis
    text = re.sub(r"\(([^)]+)\)"," ",text)

    # remove all Latex tags
    text = re.sub(r"\\(\w+)([\{\[][^}]+\})+"," ",text)

    # remove big non-character crumps (including special characters like \n, \t)
    text = re.sub(r"[^a-zA-Z]{20,}"," ",text)

    return text


def get_documents(config: Config)->List[Document]:
    """
    if the config.data_dir is a directory, process the data, save, and return as List[Document] \n
    if ... a pickle file, read and return that pickle file
    """

    if not config.data_dir.endswith('.pkl'):
        output = []
        for f in os.listdir(config.data_dir):
            if f.endswith('.txt'):
                loader = TextLoader(os.path.join(config.data_dir,f))
                docs = loader.load()
                for doc in docs:
                    doc.page_content = preprocess_text(text=doc.page_content)
                    output.append(doc)
        pickle.dump(output, open(config.data_dir+'.pkl','wb'))
        print('saved new pickle')

    else:
        output = pickle.load(open(config.data_dir, 'rb'))    
    return output

def extract_meta(chunk: Document)->Dict[str,str]:
    """
    metadata
    - source: a source path of this chunk (auto-added)

    - is_context_of: a list of question_id
        iterate over covid/dataset.xlsx and tag which context this chunk is a part of
    """
    meta = {}
    #import pandas as pd
    #df = pd.read_excel('data/covid/dataset_with_ids.xlsx')
    #df = df[df['source']==chunk.metadata['source']]
    #meta['is_context_of'] = is_context_of(chunk, df)
    return meta

def splitting(docs: List[Document], 
              splitter_kwargs: Dict[str, Any])->List[Document]:    
    
    chunk_size=int(splitter_kwargs['chunk_size'])
    chunk_overlap=int(float(splitter_kwargs['chunk_overlap'])*chunk_size)

    print(f'{chunk_size=}')
    print(f'{chunk_overlap=}')
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)

    for idx, split in enumerate(splits):
        for k, v in extract_meta(split).items():
            splits[idx].metadata[k] = v

    return splits

def get_embedding(model_name: str):
    if model_name:
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        emb_model = HuggingFaceEmbeddings(model_name=model_name,
                                          model_kwargs=model_kwargs,
                                          encode_kwargs=encode_kwargs)
    else:
        emb_model = FakeEmbeddings(size=200)

    return emb_model

def get_vectorstore(chroma_collection_name: str,
                    distance_fn: str,
                    embedding_model: str)->VectorStore:
    
    # connect to Chroma client
    client = chromadb.PersistentClient()

    # Langchain Chroma wrapper
    langchain_chroma = Chroma(client=client,
                              collection_name=chroma_collection_name,
                              embedding_function=get_embedding(embedding_model),
                              collection_metadata={"hnsw:space": distance_fn})  
    print(f"{langchain_chroma._embedding_function.model_name=}")  
    print(f"{langchain_chroma._collection.name=}")  
    print(f"{langchain_chroma._collection.metadata=}")  
    return langchain_chroma

def main():

    config_path = os.path.join('RAG','config','indexing.yaml')
    config_dict = yaml.load(open(config_path,"r"), Loader=yaml.FullLoader)
    print('INDEXING CONFIG')
    pprint(config_dict)
    config = Config(**config_dict)

    collection_name = indexing_database.new_entry(**config.model_dump())
    docs = get_documents(data_dir=config.data_dir)
    chunks = splitting(docs=docs,
                       splitter_kwargs=config.splitter_kwargs)
    vectorstore = get_vectorstore(chroma_collection_name=collection_name,
                                  distance_fn=config.distance_fn,
                                  embedding_model=config.embedding_model)
    vectorstore.add_documents(chunks)

    indexing_database.update(key=collection_name, column='status', value=1)
    indexing_database.update(key=collection_name, column='chunks_add', value=len(chunks))

    print(f"{len(chunks)} chunks added")

if __name__=='__main__':
    main()