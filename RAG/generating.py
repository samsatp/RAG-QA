from pydantic import BaseModel
from pprint import pprint
from typing import List
import sys, yaml, os, ast
import pandas as pd

from RAG.evaluating import main as evaluate_main
from RAG import DF_COL_NAMES, answer_database
from transformers import T5Tokenizer, T5ForConditionalGeneration


class Config(BaseModel):
    # generate
    question_with_context_file: str
    generating_model: str


def merge_strings(strings: List[str]):
    return ' '.join(strings)[:400]

def generate(q: str, model, tokenizer, docs: List[str]=None)->str:
    if docs:
        input_text = f"""answer the question based on this context: {merge_strings(docs)} 
        question: {q}
        answer: """
    else:
        input_text = f"question: {q} answer: "

    input_ids = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(**input_ids)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs

def main(use_context=True):
    
    config_path = os.path.join('RAG','config','generating.yaml')
    config_dict = yaml.load(open(config_path,"r"), Loader=yaml.FullLoader)
    print('GENERATING CONFIG')
    pprint(config_dict)
    config = Config(**config_dict)

    tokenizer = T5Tokenizer.from_pretrained(config.generating_model)
    model = T5ForConditionalGeneration.from_pretrained(config.generating_model)

    df = pd.read_excel(config.question_with_context_file)
    df[DF_COL_NAMES.retrieved_docs.value] = df[DF_COL_NAMES.retrieved_docs.value].apply(ast.literal_eval)

    print(f'{len(df)} questions')
    answers = []
    for row in df.to_dict('records'): 
        if use_context:
            answer = generate(row[DF_COL_NAMES.questions.value], row[DF_COL_NAMES.retrieved_docs.value], model, tokenizer)
        else:
            answer = generate(row[DF_COL_NAMES.questions.value], None, model, tokenizer)
            
        answers.append(answer)

    df[DF_COL_NAMES.generated_answers.value] = answers

    scores = evaluate_main(predictions=df[DF_COL_NAMES.generated_answers.value], references=df[DF_COL_NAMES.answers.value])
    
    new_id = answer_database.new_entry(**config.model_dump(), **scores)

    result_path = os.path.join('RAG','results',f'{new_id}.xlsx')
    df.to_excel(result_path, index=False)
    print(f"{len(answers)} questions answered")

if __name__ == '__main__':
    main()

    