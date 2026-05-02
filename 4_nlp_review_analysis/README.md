# Project 4: NLP Sentiment & Review Analytics

## Overview
A large-scale NLP pipeline for analyzing customer feedback. It uses state-of-the-art transformer models to classify sentiment and extract latent topics.

## Features
- **Sentiment Classification**: Fine-tuned BERT model for 5-star rating prediction.
- **Topic Modeling**: BERTopic for identifying common themes in reviews.
- **NER**: Named Entity Recognition to find specific product features mentioned.

## Tech Stack
- Python
- HuggingFace Transformers
- PyTorch
- BERTopic
- Spacy

## How to Run
1. Navigate to `4_nlp_review_analysis`
2. Install dependencies: `pip install -r requirements.txt`
3. Pipeline Walkthrough: `notebooks/01_nlp_analytics.ipynb`
4. Batch Processing: `python src/inference.py`
