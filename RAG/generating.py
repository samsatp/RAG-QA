from pydantic import BaseModel
from rich import print
from typing import List
import sys, yaml, os, ast, torch
import random
import pandas as pd

from RAG.evaluating import get_metrics_df
from RAG import DF_COL_NAMES, answer_database
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForQuestionAnswering, AutoTokenizer


class Config(BaseModel):
    # generate
    question_with_context_file: str
    generating_model: str
    use_context: bool
    use_gold_context: bool

def merge_strings(strings: List[str]):
    return ' '.join(strings)

def get_prompt(question:str, tokenizer:T5Tokenizer, context:str=None)->str:
    # limit context size to 420 tokens
    context_tokens = tokenizer(context, truncation=True, max_length=400)
    context = tokenizer.decode(context_tokens['input_ids'])
    if context:
        return f"""answer the question based on this context: {context}
        question: {question}
        answer: """
    else:
        return f"Question: {question} \n Answer: "
    
def generate(q: str, docs: str, model, tokenizer)->str:
    if docs:
        input_text = f"""answer the question based on this context: {docs[:450]} 
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
        input_ids = tokenizer(q, docs, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**input_ids)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = input_ids.input_ids[0, answer_start_index : answer_end_index + 1]
        outputs = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    return outputs

def batch_generate(prompts: List[str], 
                   model: T5ForConditionalGeneration, 
                   tokenizer: T5Tokenizer)->List[str]:
    input_ids = tokenizer(prompts, return_tensors="pt", 
                          max_length=512, 
                          truncation=True, padding=True)
    with torch.no_grad():
        outputs = model.generate(**input_ids)
    outputs: List[str] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs

def main(question_with_context_file:str,
         generating_model:str,
         use_context=True,
         use_gold_context=False):
    
    print('GENERATING CONFIG')
    config = Config(question_with_context_file=question_with_context_file,
                    generating_model=generating_model,
                    use_context=use_context,
                    use_gold_context=use_gold_context)

    if 't5' in config.generating_model:
        tokenizer = T5Tokenizer.from_pretrained(config.generating_model)
        model = T5ForConditionalGeneration.from_pretrained(config.generating_model)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(config.generating_model)
        tokenizer = AutoTokenizer.from_pretrained(config.generating_model)

    df = pd.read_excel(config.question_with_context_file)
    

    print(f'{len(df)} questions')
    answers = []

    questions = df[DF_COL_NAMES.questions.value].values
    if use_context:
        if use_gold_context:
            print('USE CONTEXT: gold')
            contexts = df[DF_COL_NAMES.contexts.value].values
        else:
            print('USE CONTEXT: retrieved docs')
            contexts = df[DF_COL_NAMES.retrieved_docs.value]\
                        .apply(ast.literal_eval)\
                        .apply(merge_strings).values
        prompts = [get_prompt(question=q, 
                              context=c, 
                              tokenizer=tokenizer) 
                    for q,c in zip(contexts, questions)]
    else:
        print('CLOSE-BOOK QA')
        prompts = [get_prompt(question=q, 
                              tokenizer=tokenizer) 
                    for q in questions]

    answers = batch_generate(prompts=prompts,
                             model=model,
                             tokenizer=tokenizer)
    """ for row in df.to_dict('records'): 
        if use_context:
            if use_gold_context:
                print('USE CONTEXT: gold')
                answer = generate(row[DF_COL_NAMES.questions.value], row[DF_COL_NAMES.contexts.value], model, tokenizer)
            else:
                print('USE CONTEXT: retrieved docs')
                df[DF_COL_NAMES.retrieved_docs.value] = df[DF_COL_NAMES.retrieved_docs.value].apply(ast.literal_eval)
                answer = generate(row[DF_COL_NAMES.questions.value], merge_strings(row[DF_COL_NAMES.retrieved_docs.value]), model, tokenizer)
        else:
            print('CLOSE-BOOK QA')
            answer = generate(row[DF_COL_NAMES.questions.value], None, model, tokenizer)
            
        answers.append(answer) """

    df[DF_COL_NAMES.generated_answers.value] = answers

    metrics_df = get_metrics_df(df)
    df = pd.concat([df, metrics_df], axis=1)

    scores = metrics_df.mean(axis=0).to_dict()
    
    new_id = answer_database.new_entry(**config.model_dump(), **scores)

    result_path = os.path.join('RAG','results',f'{new_id}.xlsx')
    df.to_excel(result_path, index=False)
    print(f"{len(answers)} questions answered")

if __name__ == '__main__':
    main()

    