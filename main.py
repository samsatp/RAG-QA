



        
import sys, argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--command", type=str, required=True)
    parser.add_argument("-frac", "--frac", type=float, default=1.0, required=False)
    parser.add_argument("-context", "--context", type=str, required=False, choices=['y','n'])
    args = parser.parse_args()
    print(f"{args.command=}")
    print(f"{args.frac=}")
    if args.context:
        use_context = args.context == 'y'
    else:
        use_context = True
    print(f"use context to generate (if at all): {use_context}")

    command = args.command
    if command == 'indexing':   
        from RAG.indexing import main as indexing_main     
        indexing_main()
    elif command == 'retrieval':
        from RAG.retrieval import main as retrieval_main
        retrieval_main(frac=args.frac)
    elif command == 'generating':
        from RAG.generating import main as generating_main
        generating_main(use_context)
    elif command == 'evaluate':
        from RAG.evaluating import main as evaluate_main
        answer_file = sys.argv[2]
        evaluate_main(answer_file)
    else:
        print('command not found')
