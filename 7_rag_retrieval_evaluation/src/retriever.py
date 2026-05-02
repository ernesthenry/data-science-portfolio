import numpy as np

class VectorStore:
    """Mock Vector Store implementation (FAISS-like)."""
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = []
        self.documents = []

    def add_documents(self, docs):
        # In real FAISS, we would encode these documents using a transformer
        print(f"Indexing {len(docs)} documents...")
        self.documents.extend(docs)
        # Mock embeddings
        self.index.append(np.random.randn(len(docs), self.dimension))

    def search(self, query, k=3):
        print(f"Searching for: {query}")
        # Return top k mock documents
        return self.documents[:k]

class RAGSystem:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def query(self, user_query):
        context = self.vector_store.search(user_query)
        context_str = "\n".join(context)
        prompt = f"Context:\n{context_str}\n\nQuestion: {user_query}\nAnswer:"
        
        # In real RAG, we would call an LLM here
        print("Feeding context to LLM...")
        return "This is a retrieved answer based on the provided context."

if __name__ == "__main__":
    store = VectorStore()
    store.add_documents([
        "The capital of France is Paris.",
        "The Eifel Tower was built in 1889.",
        "Wine production is a major industry in France."
    ])
    
    rag = RAGSystem(store)
    print(rag.query("Tell me about France's capital."))
