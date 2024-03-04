__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from typing import List, Any, Dict, Callable
from pydantic import BaseModel
from pprint import pprint
import chromadb
import yaml
import os

from langchain_core.documents.base import Document
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma


class Config(BaseModel):
    data_dir: str
    chroma_collection_name: str
    distance_fn: str
    embedding_model: str
    splitter_kwargs: Dict[str, Any]

config_path = os.path.join('RAG','config','indexing.yaml')
config = yaml.load(open(config_path,"r"), Loader=yaml.FullLoader)
print('INDEXING CONFIG')
pprint(config)
config = Config(**config)

def get_documents(data_dir: os.PathLike = config.data_dir)->List[Document]:
    docs = []
    for f in os.listdir(data_dir):
        if f.endswith('.txt'):
            loader = TextLoader(os.path.join(data_dir,f))
            docs.extend(loader.load())
    return docs

def extract_meta(text: str)->Dict[str,str]:
    return {'meta1':text[:10]}

def splitting(docs: List[Document], 
              splitter_kwargs: Dict[str, Any] = config.splitter_kwargs)->List[Document]:    
    text_splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)
    splits = text_splitter.split_documents(docs)

    for idx, split in enumerate(splits):
        for k, v in extract_meta(split.page_content).items():
            splits[idx].metadata[k] = v

    return splits

def get_embedding(model_name = config.embedding_model):
    if model_name:
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        emb_model = HuggingFaceEmbeddings(model_name=model_name,
                                          model_kwargs=model_kwargs,
                                          encode_kwargs=encode_kwargs)
    else:
        emb_model = FakeEmbeddings(size=200)

    return emb_model

def get_vectorstore(chroma_collection_name: str = config.chroma_collection_name,
                    distance_fn: str = config.distance_fn,
                    model_name: str = config.embedding_model)->VectorStore:
    
    # connect to Chroma client
    client = chromadb.PersistentClient()

    # Langchain Chroma wrapper
    langchain_chroma = Chroma(client=client,
                              collection_name=chroma_collection_name,
                              embedding_function=get_embedding(model_name),
                              collection_metadata={"hnsw:space": distance_fn})    
    return langchain_chroma

def main():
    docs = get_documents()
    chunks = splitting(docs)
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    print(f"{len(chunks)} chunks added")

if __name__=='__main__':
    main()