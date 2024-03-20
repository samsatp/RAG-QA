journal

18-19 march
- running to see some results and they are terrible. Then, inspect the data -> see lots of questions that expect us to read to context beforehand. E.g. what is the purpose of this study? how many patients involve in the study? what is the conclusion basedd on Figure 2?
    - so, I filter those questions out based on keyword matching (e.g. 'study','survey','figure') ~400 questions were removed
    - why remove? because for RAG, the use case is : given a question, it tries to find evidence, then answer based on that evidence. Not reading comprehension (i.e. reading the context, and answer based on the context).


20 march
- after inspecting the data and the predictions. the quality of the generated answers oftern rely on the retrieved docs. I.e. if the retrieved docs contain answers, flan-t5 usually answers correctly. So we need to improve retrieval!

some possible ideas:
- use metadata to scope the search space
    - chroma has `where_document` param in query function for that. Extract keyword from question and use that in where_document to filter doc search
    - Or, let say we have a keyword extractor. we run it with each chunk -> keep as metadata. Then during query, run the same extractor to get keyword, use that keyword as meta un search. There is a SciSpacy model that can extract scientific entity, we can use that as a meta extractor 