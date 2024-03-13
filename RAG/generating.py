from pydantic import BaseModel
from pprint import pprint
from typing import List
import sys, yaml, os
import pandas as pd

from RAG.retrieval import config as retrieval_config
from RAG import DF_COL_NAMES, answer_database
from langchain_core.documents.base import Document

from transformers import T5Tokenizer, T5ForConditionalGeneration


class Config(BaseModel):
    # generate
    question_with_context_file: str
    generating_model: str

config_path = os.path.join('RAG','config','generating.yaml')
config_dict = yaml.load(open(config_path,"r"), Loader=yaml.FullLoader)
print('GENERATING CONFIG')
pprint(config_dict)
config = Config(**config_dict)


tokenizer = T5Tokenizer.from_pretrained(config.generating_model)
model = T5ForConditionalGeneration.from_pretrained(config.generating_model)


def generate(q: str, docs: List[Document])->str:
    input_text = f"Answer this question: {q}"
    """
    archive prompt
    Context: ... 
    Q: ...
    A:
    """
    input_ids = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(**input_ids)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs

def main():
    new_id = answer_database.new_entry(config.model_dump())
    df = pd.read_excel(config.question_with_context_file)
    answers = []
    for row in df.to_dict('records'): 
        answer = generate(row[DF_COL_NAMES.questions.value], row[DF_COL_NAMES.retrieved_docs.value])
        answers.append(answer)
    df[DF_COL_NAMES.generated_answers.value] = answers
    df.to_excel(f'{new_id}.xlsx', index=False)
    print(f"{len(answers)} questions answered")

if __name__ == '__main__':
    main()

    