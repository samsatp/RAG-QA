__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from typing import List, Dict
from collections import defaultdict

import sys
import chromadb
import pandas as pd
from RAG import DF_COL_NAMES

def is_context_of(chunk:str, df:pd.DataFrame)->str:
    output = []
    for row in df.to_dict('records'):
        if str(chunk) in str(row[DF_COL_NAMES.contexts.value]):
            output.append(str(row[DF_COL_NAMES.question_ids.value]))
    return '_'.join(output)

def get_chunk_id(chunk:str, collection: chromadb.Collection)->str:
    return collection.get(where_document={'$contains': chunk})['ids'][0]

def get_context_chunks(collection: chromadb.Collection)->Dict[str, List[str]]:
    '''
    return a dict
    - key : question_id
    - value : a list of chunk_ids
    '''
    data = collection.get()
    print(collection._collection)
    print(data['ids'][:10])
    print(data['metadatas'][:10])
    output = defaultdict(lambda:[])
    for chunk_id, metadata in zip(data['ids'], data['metadatas']):
        question_ids = metadata['is_context_of'].split('_')
        for qid in question_ids:
            output[qid].append(chunk_id)
    return output

    

if __name__=='__main__':
    command = sys.argv[1]
    if command == 'truncate':
        # reset chromadb (wipe everything out)
        collection_name = sys.argv[2]
        client = chromadb.PersistentClient()
        client.delete_collection(collection_name)
        client.get_or_create_collection(collection_name)