

from pydantic import BaseModel
from RAG.generating import config as generating_config

import pandas as pd

class Evaluator:
    def __init__(self, config=generating_config):
        self.config = config
        self.data = ...

    def set_data(self):
        """
        For a query, there are K retrieved chunks.
        This method tags if a chunk is relevant to the query
        """
        df = pd.read_excel(self.config.query_file)
        ...

    def make_report(self):
        return

class RetrievalEvaluator(Evaluator):
    def recall_at_k(self):
        return
    
    def precision_at_k(self):
        return
    
    def mean_reciprocal_rank(self):
        return
    
    def mean_avg_precision(self):
        return
    
    def ndcg_at_k(self):
        return