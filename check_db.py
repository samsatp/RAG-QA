__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from rich import print
import chromadb
import pandas as pd

client = chromadb.PersistentClient('chroma')

rows = []
for i, col in enumerate(client.list_collections(), start = 1):
    rows.append({
        'n': i,
        'collection':col.name,
        'count':col.count(),
        'meta':col.metadata
    })

df = pd.DataFrame(rows).set_index('n')
print(df)