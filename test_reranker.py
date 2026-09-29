from langchain.schema import Document
from reranker import rerank_crossencoder

# 1. Siapkan contoh query
query = "Apa ibu kota Indonesia?"

# 2. Siapkan beberapa contoh dokumen
docs = [
    Document(page_content="Tokyo adalah ibu kota Jepang.", metadata={"id": 1}),
    Document(page_content="Jakarta adalah ibu kota dari negara Indonesia.", metadata={"id": 2}),
    Document(page_content="Kuala Lumpur merupakan ibu kota Malaysia.", metadata={"id": 3})
]

# 3. Jalankan reranker
ordered_docs, scores = rerank_crossencoder(query=query, docs=docs, top_k=2)

# 4. Tampilkan hasil
print(f"Query: {query}\n")
print("Hasil Reranking:")
for i, doc in enumerate(ordered_docs):
    print(f"{i+1}. Skor: {scores[id(doc)]:.4f} | Teks: {doc.page_content}")
