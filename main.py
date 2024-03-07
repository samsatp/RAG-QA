import sys


if __name__=='__main__':
    command = sys.argv[1]
    if command == 'indexing':
        from RAG.indexing import main as indexing_main
        indexing_main()
    elif command == 'retrieval':
        from RAG.retrieval import main as retrieval_main
        n_queries = int(sys.argv[2])
        retrieval_main(n_queries=n_queries)
    elif command == 'generating':
        from RAG.generating import main as generating_main
        generating_main()
    else:
        ...
