from enum import Enum
from typing import Dict, Any, List
from dataclasses import dataclass

import os, yaml, uuid, rich, json
from datetime import datetime

class DF_COL_NAMES(Enum):
    """
    for consistent excel column names
    """
    question_ids = 'question_ids'
    questions = 'questions'
    contexts = 'contexts'
    answers = 'answers'
    retrieved_docs = 'retrieved_docs'
    generated_answers = 'generated_answers'
    context_chunk_ids = 'context_chunk_ids'
    retrieved_doc_ids = 'retrieved_doc_ids'

    def __str__(self):
        return self.name
    

@dataclass
class RAG_database:
    db_file: str   # path to store db
    pk_name: str   # primary key name
    base_dir = os.path.join('RAG','db')
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join('RAG','results'), exist_ok=True)

    def __post_init__(self):
        self.db_path = os.path.join(self.base_dir, self.db_file)
    #    if not os.path.isfile(self.db_path):
    #        df = pd.DataFrame(columns=[self.pk_name]+self.columns)
    #        df = df.set_index(self.pk_name)
    #        df.to_csv(self.db_path, mode='w')
        
    def _get_all(self)->List[dict[str,Any]]:
        with open(self.db_path, mode='r') as f:
            return [json.loads(row.strip()) for row in f.readlines()]
    
    def _insert(self, row:Dict[str, Any]):
        with open(self.db_path, mode='a') as f:
            f.write(json.dumps(row) + '\n')

    def get(self, index)->dict:
        for e in self._get_all():
            if str(e[self.pk_name]) == str(index):
                row = e.copy()
        return row
    
    def new_entry(self, **row: Dict[str,Any])->str:
        
        row[self.pk_name] = f"{self.pk_name}_{str(uuid.uuid4())}" 
        row['create_time'] = datetime.strftime(datetime.now(), format= "%d-%m-%y T %H:%M:%S")
        row['update_time'] = datetime.strftime(datetime.now(), format= "%d-%m-%y T %H:%M:%S")
        self._insert(row)
        return row[self.pk_name]
    
    def update(self, index, key, value):
        for e in self._get_all():
            if str(e[self.pk_name]) == str(index):
                row = e.copy()

        row[key] = value
        row['update_time'] = datetime.strftime(datetime.now(), format= "%d-%m-%y T %H:%M:%S")
        self._insert(row)


indexing_database = RAG_database(
    db_file = 'indexing.jsonl', 
    pk_name = 'collection'  # use 'collection' as PK because one indexing run will create a new Chroma collection
)

retrieval_database = RAG_database(
    db_file = 'retrieval.jsonl',
    pk_name = 'retrieval_file'
)

answer_database = RAG_database(
    db_file = 'answer.jsonl',
    pk_name = 'answer_file'
)