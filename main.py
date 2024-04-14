



        
import sys, argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--command", type=str, required=True)
    parser.add_argument("-frac", "--frac", type=float, default=1.0, required=False)
    
    # indexing config
    parser.add_argument("-embedding_model", "--embedding_model", type=str, required=False)
    parser.add_argument("-distance_fn","--distance_fn", type=str, required=False)
    parser.add_argument("-chunk_size","--chunk_size", type=int, required=False)
    parser.add_argument("-chunk_overlap", "--chunk_overlap", type=float, required=False)
    parser.add_argument("-data_dir", "--data_dir", required=False, default="data/covid.pkl")
    
    # retrieval config
    parser.add_argument("-question_file","--question_file", type=str, required=False)
    parser.add_argument("-collection_name","--collection_name", type=str, required=False)
    parser.add_argument("-k","--k", type=int, required=False)

    # generation config
    parser.add_argument("-question_with_context_file","--question_with_context_file", type=str, required=False)
    parser.add_argument("-generating_model","--generating_model", type=str, required=False, default="google/flan-t5-base")
    parser.add_argument("-use_context", "--use_context", type=str, required=False, choices=['y','n'])

    args = parser.parse_args()
    print(f"{args.command=}")
    print(f"{args.frac=}")
    if args.use_context:
        use_context = args.use_context == 'y'
    else:
        use_context = True
    print(f"use context to generate (if at all): {use_context}")

    command = args.command
    if command == 'indexing':   
        from RAG.indexing import main as indexing_main     
        indexing_main(data_dir=args.data_dir,
                      distance_fn=args.distance_fn,
                      embedding_model=args.embedding_model,
                      chunk_size=args.chunk_size,
                      chunk_overlap=args.chunk_overlap)
    elif command == 'retrieval':
        from RAG.retrieval import main as retrieval_main
        retrieval_main(frac=args.frac,
                       question_file=args.question_file,
                       collection_name=args.collection_name,
                       k=args.k)
    elif command == 'generating':
        from RAG.generating import main as generating_main
        generating_main(question_with_context_file=args.question_with_context_file,
                        generating_model=args.generating_model,
                        use_context=use_context)
    elif command == 'evaluate':
        ...
    else:
        print('command not found')
