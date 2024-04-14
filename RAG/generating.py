from pydantic import BaseModel
from rich import print
from typing import List
import sys, yaml, os, ast, torch
import pandas as pd

from RAG.evaluating import get_metrics_df
from RAG import DF_COL_NAMES, answer_database
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForQuestionAnswering, AutoTokenizer


class Config(BaseModel):
    # generate
    question_with_context_file: str
    generating_model: str


def merge_strings(strings: List[str]):
    return ' '.join(strings)

def generate(q: str, docs: List[str], model, tokenizer)->str:
    if docs:
        input_text = f"""answer the question based on this context: {merge_strings(docs)} 
        question: {q}
        answer: """
    else:
        input_text = f"question: {q} \n answer: "

    if 't5' in model.name_or_path:
        input_ids = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**input_ids)
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)

    elif 'deepset/roberta' in model.name_or_path:
        input_ids = tokenizer(q, merge_strings(docs), return_tensors="pt")
        with torch.no_grad():
            outputs = model(**input_ids)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = input_ids.input_ids[0, answer_start_index : answer_end_index + 1]
        outputs = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    return outputs

def main(question_with_context_file:str,
         generating_model:str,
         use_context=True,):
    
    print('GENERATING CONFIG')
    config = Config(question_with_context_file=question_with_context_file,
                    generating_model=generating_model)

    if 't5' in config.generating_model:
        tokenizer = T5Tokenizer.from_pretrained(config.generating_model)
        model = T5ForConditionalGeneration.from_pretrained(config.generating_model)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(config.generating_model)
        tokenizer = AutoTokenizer.from_pretrained(config.generating_model)

    df = pd.read_excel(config.question_with_context_file)
    df[DF_COL_NAMES.retrieved_docs.value] = df[DF_COL_NAMES.retrieved_docs.value].apply(ast.literal_eval)

    print(f'{len(df)} questions')
    answers = []
    for row in df.to_dict('records'): 
        if use_context:
            print('USE CONTEXT')
            answer = generate(row[DF_COL_NAMES.questions.value], row[DF_COL_NAMES.retrieved_docs.value], model, tokenizer)
        else:
            print('CLOSE-BOOK QA')
            answer = generate(row[DF_COL_NAMES.questions.value], None, model, tokenizer)
            
        answers.append(answer)

    df[DF_COL_NAMES.generated_answers.value] = answers

    metrics_df = get_metrics_df(df)
    df = pd.concat([df, metrics_df], axis=1)

    metrics_df.to_csv('metrics.csv')
    scores = metrics_df.mean(axis=0).to_dict()
    
    new_id = answer_database.new_entry(**config.model_dump(), **scores)

    result_path = os.path.join('RAG','results',f'{new_id}.xlsx')
    df.to_excel(result_path, index=False)
    print(f"{len(answers)} questions answered")

if __name__ == '__main__':
    main()

    