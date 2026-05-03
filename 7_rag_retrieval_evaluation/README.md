# Project 7: RAG Retrieval & Evaluation

## Overview
An enterprise-grade Retrieval-Augmented Generation (RAG) system. This project goes beyond basic RAG by implementing advanced retrieval strategies and rigorous evaluation metrics.

## Scope in this repo
The Python modules here provide a **mock vector store**, a **simple RAG query loop**, and **retrieval metrics** (e.g. MRR, Hit@K). Notebook walkthroughs cover document and evaluation flows. A full stack with **Weaviate/Pinecone**, **RAGAS**, **cross-encoders**, **Neo4j**, and hosted LLMs is left as a production extension—aligned with the feature list below.

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

## Recommended dataset
**Primary (QA + retrieval):** **SQuAD v2** — passage-level grounding and answer spans for retrieval evaluation.

- [SQuAD explorer](https://rajpurkar.github.io/SQuAD-explorer/)

**Alternative (scientific PDFs / long documents):** **arXiv metadata** — titles/abstracts for indexing and chunking at scale.

- [Kaggle: arXiv metadata](https://www.kaggle.com/datasets/Cornell-University/arxiv)

## How to Run
1. Navigate to `7_rag_retrieval_evaluation`
2. Document pipeline: `notebooks/01_document_analysis.ipynb`
3. Retrieval metrics: `notebooks/02_retrieval_evaluation.ipynb`
4. Metric helpers: `python src/evaluator.py`
