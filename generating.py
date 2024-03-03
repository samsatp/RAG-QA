from pydantic import BaseModel
from pprint import pprint
from typing import List
import sys, yaml
import pandas as pd

from indexing import get_vectorstore
from langchain_core.documents.base import Document

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

class Config(BaseModel):
    query_file: str # xlsx file
    k: int

config = yaml.load(open("generating.yaml","r"), Loader=yaml.FullLoader)
print('GENERATING CONFIG')
pprint(config)
config = Config(**config)

def get_relevant_documents(query: str, k:int)->List[Document]:
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k, filter=None)


def process_query(q: str)->str:
    return q

def process_doc(doc: Document)->Document:
    return doc

def generate(q: str, docs: List[Document])->str:
    input_text = f"Answer this question: {q}"
    input_ids = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(**input_ids)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs

def main():
    df = pd.read_excel(config.query_file)
    queries = df['question']
    answers = []
    for query in queries.values: 
        relevant_docs = get_relevant_documents(query, k=config.k)
        relevant_docs = [process_doc(_) for _ in relevant_docs]
        answer = generate(query, relevant_docs)
        answers.append(answer)
    
    df['answer'] = answers
    df.to_excel(config.query_file, index=False)
    print(f"{len(answers)} questions done")

if __name__ == '__main__':
    main()

    