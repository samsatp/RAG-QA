from enum import Enum
from typing import Dict, Any, List
from dataclasses import dataclass

import os, yaml
import pandas as pd

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
    columns: List[str]
    base_dir = os.path.join('RAG','db')
    os.makedirs(base_dir, exist_ok=True)

    def __post_init__(self):
        self.db_path = os.path.join(self.base_dir, self.db_file)
        if not os.path.isfile(self.db_path):
            df = pd.DataFrame(columns=[self.pk_name]+self.columns)
            df = df.set_index(self.pk_name)
            df.to_csv(self.db_path)
        
        self.df = pd.read_csv(self.db_path, index_col=self.pk_name)
    
    def new_entry(self, values: Dict[str,Any])->str:
        for k in values.keys():
            if k not in self.df.columns:
                self.df[k] = None
        new_id = f"{self.pk_name}_{len(self.df) + 1}"
        self.df.loc[new_id] = values
        self.df.to_csv(self.db_path)
        self.df = pd.read_csv(self.db_path, index_col=self.pk_name)
        return new_id


indexing_database = RAG_database(
    db_file = 'indexing.csv', 
    pk_name = 'collection',  # use 'collection' as PK because one indexing run will create a new Chroma collection
    columns = ['distance_fn','embedding_model','splitter_kwargs','query_file','k','generating_model']
)

retrieval_database = RAG_database(
    db_file = 'retrieval.csv',
    pk_name = 'retrieval_file',
    columns = ['collection','question_file','k']
)

answer_database = RAG_database(
    db_file = 'answer.csv',
    pk_name = 'answer_file',
    columns = ['retrieval_file', 'generating_model']
)

score_database = RAG_database(
    db_file = 'score.csv',
    pk_name = 'score',
    columns = [# Lexical metrics
               'bleu_bleu','rouge_rouge1','rouge_rouge2','rouge_rougeL','rouge_rougeLsum','meteor_meteor',
               # Semantic similarity metrics
               'sts_mean','sts_median','sts_std','sts_min','sts_max',
               # Reranking metrics
               'retrieval_mean','retrieval_median','retrieval_std','retrieval_min','retrieval_max']
)
