from pydantic import BaseModel
from pprint import pprint
from typing import List
import sys, yaml
import pandas as pd

from retrieval import config as retrieval_config
from langchain_core.documents.base import Document

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")


def generate(q: str, docs: List[Document])->str:
    input_text = f"Answer this question: {q}"
    input_ids = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(**input_ids)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs

def main():
    df = pd.read_excel(retrieval_config.query_file)
    answers = []
    for row in df.to_dict('records'): 
        answer = generate(row['question'], row['relevant_docs'])
        answers.append(answer)
    df['answer'] = answers
    df.to_excel(retrieval_config.query_file, index=False)
    print(f"{len(answers)} questions answered")

if __name__ == '__main__':
    main()

    