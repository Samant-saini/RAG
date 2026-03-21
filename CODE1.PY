from rank_bm25 import BM25Okapi

documents = [
    "AI helps in automation and decision making",
    "Machine learning is part of AI",
    "Python is used for programming"
]

# Tokenize
tokenized_docs = [doc.lower().split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)

query = "AI automation"
tokenized_query = query.lower().split()

scores = bm25.get_scores(tokenized_query)

print(scores)