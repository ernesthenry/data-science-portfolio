import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def clean_text(text):
    """Basic NLP cleaning pipeline."""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    return " ".join(tokens)

if __name__ == "__main__":
    sample = "I love this product! It's better than 100 other things I've tried. <br> 5 stars!"
    cleaned = clean_text(sample)
    print(f"Original: {sample}")
    print(f"Cleaned: {cleaned}")
