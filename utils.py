import sys
import chromadb
if __name__=='__main__':
    command = sys.argv[1]
    if command == 'truncate':
        # reset chromadb (wipe everything out)
        collection_name = sys.argv[2]
        client = chromadb.PersistentClient()
        client.delete_collection(collection_name)
        client.get_or_create_collection(collection_name)