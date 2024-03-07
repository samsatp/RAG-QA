from enum import Enum

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