import nltk
import pandas as pd  # <-- Import pandas
import re

# Download required NLTK resources safely
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer

# Define stopwords
stop_words = set(stopwords.words('english'))
new_stopwords = ["mario", "la", "blah", "saturday", "monday", "sunday", "morning", "evening", "friday", "would", "shall", "could", "might"]
stop_words.update(new_stopwords)

# Ensure "not" is NOT removed
if "not" in stop_words:
    stop_words.remove("not")

# Removing special characters
def remove_special_character(content):
    return re.sub(r'\W+', ' ', content)

# Removing URLs
def remove_url(content):
    return re.sub(r'http\S+', '', content)

# Removing stopwords from text
def remove_stopwords(content):
    clean_data = [word.strip().lower() for word in content.split() if word.strip().lower() not in stop_words and word.strip().isalpha()]
    return " ".join(clean_data)

# Expanding English contractions
def contraction_expansion(content):
    contractions = {
        r"won\'t": "would not",
        r"can\'t": "can not",
        r"don\'t": "do not",
        r"shouldn\'t": "should not",
        r"needn\'t": "need not",
        r"hasn\'t": "has not",
        r"haven\'t": "have not",
        r"weren\'t": "were not",
        r"mightn\'t": "might not",
        r"didn\'t": "did not",
        r"n\'t": " not"
    }
    for pattern, replacement in contractions.items():
        content = re.sub(pattern, replacement, content)
    return content

# Data preprocessing function
def data_cleaning(content):
    content = contraction_expansion(content)
    content = remove_special_character(content)
    content = remove_url(content)
    content = remove_stopwords(content)    
    return content

# Custom Transformer for Scikit-Learn Pipelines
class DataCleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('calling--init--')

    def fit(self, X, y=None):
        print('calling fit')
        return self

    def transform(self, X, y=None):
        if isinstance(X, list):
            X = pd.Series(X) 
        X = X.astype(str).apply(data_cleaning)  
        return X

# Lemmatization tokenizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()

    def __call__(self, reviews):
        if isinstance(reviews, str):  # Ensure input is a string
            return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]
        return []
