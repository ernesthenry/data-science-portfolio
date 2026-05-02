# Project 7: RAG Retrieval & Evaluation

## Overview
An enterprise-grade Retrieval-Augmented Generation (RAG) system. This project goes beyond basic RAG by implementing advanced retrieval strategies and rigorous evaluation metrics.

## Features
- **Hybrid Search**: Combining vector similarity with keyword search (BM25).
- **Reranking**: Using Cross-Encoders to improve precision.
- **Evaluation**: Implementing RAGAS (Retrieval-Augmented Generation Assessment) metrics:
    - Faithfulness
    - Answer Relevance
    - Context Precision
- **Graph-based RAG**: Integration with Neo4j for semantic relationship retrieval.

## Tech Stack
- Python
- LangChain / LlamaIndex
- Weaviate / Pinecone
- Sentence-Transformers
- OpenAI / Mistral

## How to Run
1. Navigate to `7_rag_retrieval_evaluation`
2. Pipeline & Evaluation Walkthrough: `notebooks/01_rag_evaluation.ipynb`
3. Run evaluation script: `python src/evaluator.py`
