from typing import List, Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd
import sys, os
import rich

from sentence_transformers import CrossEncoder
from RAG import DF_COL_NAMES, answer_database
import evaluate

@dataclass
class Evaluator:
    # lexical metrics
    rouge = evaluate.load('rouge', rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_aggregator=True, use_stemmer=True)
    bleu = evaluate.load('bleu', max_order=4, smooth=False)
    meteor = evaluate.load('meteor')
    sacrebleu = evaluate.load('sacrebleu', lowercase=True)
    em = evaluate.load('exact_match')
    
    # learned metrics
    sts_crossEnc = CrossEncoder("cross-encoder/stsb-roberta-base")
    #bertscore = evaluate.load('bertscore', idf=True)
    #bleurt = evaluate.load('bleurt', checkpoint="bleurt-base-128")

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
        lexical_metrics = [self.em,self.bleu,self.sacrebleu,self.rouge,self.meteor]
        metrics = evaluate.combine(lexical_metrics, force_prefix=True)
        results = metrics.compute(predictions=[prediction], references=[reference])

        # remove redundant keys
        results.pop('bleu_precisions')
        results.pop('bleu_brevity_penalty')
        results.pop('bleu_length_ratio')
        results.pop('bleu_translation_length')
        results.pop('bleu_reference_length')
        results.pop('sacrebleu_counts')
        results.pop('sacrebleu_totals')
        results.pop('sacrebleu_precisions')
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


def get_stats(name:str, scores:np.ndarray)->Dict[str,float]:
        return {f'{name}_mean':np.mean(scores),
                f'{name}_median':np.median(scores),
                f'{name}_std':np.std(scores),
                f'{name}_min':np.min(scores),
                f'{name}_max':np.max(scores)}

def get_learned_metrics(predictions: List[str], references: List[str])->Dict[str,float]:
    """
    
    - BLEURT output = a number between 0 and (approximately 1). 
        values closer to 1 representing more similar texts.
    """
    bertscore = evaluate.load('bertscore')
        
    results = bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")
    results.update(get_stats('bertscore_precision', results['precision']))
    results.update(get_stats('bertscore_recall', results['recall']))
    results.update(get_stats('bertscore_f1', results['f1']))

    results.pop('precision')
    results.pop('recall')
    results.pop('f1')

    bleurt = evaluate.load('bleurt', checkpoint="bleurt-tiny-128")
    results.update(bleurt.compute(predictions=predictions, references=references))

    return results


def main(command, **kwargs):
    if command == 'get_learned_metrics':
        answerfile_excel = kwargs['answerfile_excel']
        df = pd.read_excel(answerfile_excel)
        predictions = df[DF_COL_NAMES.generated_answers.value].values
        references = df[DF_COL_NAMES.answers.value].values
        results = get_learned_metrics(predictions=predictions, references=references)
        
        _, answerfile = os.path.split(answerfile_excel)
        answerfile = answerfile.strip('.xlsx')

        rich.print(results)
        for k,v in results.items():
            answer_database.update(index=answerfile, key=k, value=v)