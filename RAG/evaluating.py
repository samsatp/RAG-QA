

from pydantic import BaseModel
from RAG.generating import config as generating_config
from abc import abstractmethod, ABC

import pandas as pd

class Evaluator(ABC):
    def __init__(self, config=generating_config):
        self.config = config
        self.data = []

    @abstractmethod
    def set_data(self):
        ...

    @abstractmethod
    def make_report(self):
        ...

class RetrievalEvaluator(Evaluator):

    @staticmethod
    def is_positive(actual_context:str, retrieved_doc:str)->bool:
        return

    def set_data(self):
        """
        For a query, there are K retrieved chunks.
        This method tags if a chunk is relevant to the query
        """
        df = pd.read_excel(self.config.query_file)
        for i, row in enumerate(df.to_dict('records')):
            self.is_positive()


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