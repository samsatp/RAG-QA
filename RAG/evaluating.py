from typing import List, Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd
import sys

from sentence_transformers import CrossEncoder
from RAG import DF_COL_NAMES
import evaluate

@dataclass
class Evaluator:
    # passage retrieval encoder
    # pr_crossEnc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    
    # semantic textual similarity encoder
    sts_crossEnc = CrossEncoder("cross-encoder/stsb-roberta-base")

    rouge = evaluate.load('rouge', rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_aggregator=True, use_stemmer=True)
    bleu = evaluate.load('bleu', max_order=4, smooth=False)
    meteor = evaluate.load('meteor')


    def get_stats(self, name:str, scores:np.ndarray)->Dict[str,float]:
        return {f'{name}_mean':np.mean(scores),
                f'{name}_median':np.median(scores),
                f'{name}_std':np.std(scores),
                f'{name}_min':np.min(scores),
                f'{name}_max':np.max(scores)}

    def eval_lexical(self, prediction: str, reference: str)->Dict[str,float]:
        """
        Lexical-based metrics
        - BLEU: N-gram overlap precision
        - ROUGE: N-gram overlap f1 
        - METEOR: F1 of the alignment matching with a chunk distribution penalty

        This function can evaluate:
        - gold-standard answers Vs. generated answers
        - gold-standard context Vs. retrieved context chunks 
        """
        if len(prediction)==0:
            prediction = 'x'
        lexical_metrics = [self.bleu,self.rouge,self.meteor]
        metrics = evaluate.combine(lexical_metrics, force_prefix=True)
        results = metrics.compute(predictions=[prediction], references=[reference])
        results.pop('bleu_precisions')
        return results

    # semantic based
    def eval_semantic(self, prediction: str, reference: str)->Dict[str,float]:
        """
        Semantic-based metric: use cross-encoder model trained on STS
        https://www.sbert.net/docs/pretrained_cross-encoders.html#stsbenchmark

        This function can evaluate:
        - gold-standard answers Vs. generated answers
        - gold-standard context Vs. retrieved context chunks 
        """
        return {'sts':self.sts_crossEnc.predict([prediction, reference])}

    # semantic based
    #def eval_retrieval(self, questions: List[str], retrieved_chunks: List[str])->Dict[str,float]:
    #    """
    #    Semantis-based metric: use cross-encoder model trained on MS Marcro Passage Retrieval
    #    https://www.sbert.net/docs/pretrained_cross-encoders.html#ms-marco
    #
    #    This function can evaluate:
    #    - retrieved context chunks (using questions and the retrieved context chunks as inputs)
    #    """
    #    data = list(zip(questions, retrieved_chunks))
    #    pr_scores = self.pr_crossEnc.predict(data)
    #    return self.get_stats('retrieval', pr_scores)
    
def get_metrics_row(prediction: str, reference: str)->Dict[str,float]:
    evaluator = Evaluator()
    lexical_metrics = evaluator.eval_lexical(prediction, reference)
    semantic_metrics = evaluator.eval_semantic(prediction, reference)
    return dict(**lexical_metrics, **semantic_metrics)


def get_metrics_df(df: pd.DataFrame)->pd.DataFrame:
    
    metrics = []
    for row in df.to_dict('records'):
        metrics.append(get_metrics_row(prediction=row[DF_COL_NAMES.generated_answers.value],
                                       reference=row[DF_COL_NAMES.answers.value]))
    metrics = pd.DataFrame(metrics)
    return metrics

