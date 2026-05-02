from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        print(f"Loading model: {model_name}...")
        # Note: In a production environment, you might pre-download the model
        try:
            self.classifier = pipeline("sentiment-analysis", model=model_name)
        except Exception as e:
            print(f"Error loading transformer: {e}")
            self.classifier = None

    def analyze(self, texts):
        if not self.classifier:
            # Fallback for demo purposes if transformers isn't installed or model fails
            return [{"label": "POSITIVE", "score": 0.5}] * len(texts)
        
        results = self.classifier(texts)
        return results

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    reviews = [
        "This is the best purchase I've made all year!",
        "Poor quality and arrived late.",
        "It's okay, does the job but nothing special."
    ]
    scores = analyzer.analyze(reviews)
    for text, score in zip(reviews, scores):
        print(f"Review: {text}")
        print(f"Result: {score['label']} ({score['score']:.2f})\n")
