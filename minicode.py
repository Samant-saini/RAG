from rank_bm25 import BM25Okapi
import numpy as np

documents = [
    "AI helps in automation and decision making",
    "Machine learning is a subset of AI",
    "Python is used for programming"
]

tokenized_docs = [doc.lower().split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)

query = "AI automation"

tokenized_query = query.lower().split()

scores = bm25.get_scores(tokenized_query)

best_doc_index = np.argmax(scores)
if max(scores) == 0:
    print("No relevant document found")
else:
    best_doc_index = np.argmax(scores)
    print("Answer:", documents[best_doc_index])
