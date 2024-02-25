from indexing import Config
import streamlit as st
import pandas as pd 
import chromadb
import yaml


st.set_page_config(layout='wide')

def view_collections(chroma_dir):
    st.markdown(f"### DB Path: {chroma_dir}")        
    client = chromadb.PersistentClient(chroma_dir)
    st.header("Collections")

    for collection in client.list_collections():
        data = collection.get()
        df = pd.DataFrame.from_dict(data)
        df = df.loc[:,['embeddings','metadatas','documents']]

        st.markdown(f"### Collection: {collection.name}")
        st.dataframe(df)

if __name__ == "__main__":
    view_collections(chroma_dir='chroma')
    
