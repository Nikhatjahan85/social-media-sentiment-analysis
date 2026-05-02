import re
import nltk
from nltk.corpus import stopwords
import string

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # lowercase
    text = text.lower()
    # remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # remove @mentions
    text = re.sub(r'@\w+', '', text)
    # remove hashtags
    text = re.sub(r'#\w+', '', text)
    # remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
    
    tokens = text.split()
    filtered_tokens = [w for w in tokens if not w in stop_words]
    return ' '.join(filtered_tokens)

def preprocess_pipeline(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    return text