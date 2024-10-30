import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')


def clean_text(text):
    """
    Clean the input text by removing special characters and extra spaces.
    
    Parameters:
    text (str): The input text.
    
    Returns:
    str: The cleaned text.
    """
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def tokenize_text(text):
    """
    Tokenize the input text into words.
    
    Parameters:
    text (str): The input text.
    
    Returns:
    list: A list of tokenized words.
    """
    tokens = word_tokenize(text)
    return tokens


def remove_stopwords(tokens):
    """
    Remove stopwords from the tokenized words.
    
    Parameters:
    tokens (list): A list of tokenized words.
    
    Returns:
    list: A list of words with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens


def embed_text(tokens):
    """
    Embed the tokenized words using Word2Vec.
    
    Parameters:
    tokens (list): A list of tokenized words.
    
    Returns:
    list: A list of word embeddings.
    """
    model = Word2Vec([tokens], min_count=1)
    embeddings = [model.wv[word] for word in tokens]
    return embeddings


def process_text(text):
    """
    Process the input text by cleaning, tokenizing, removing stopwords, and embedding.
    
    Parameters:
    text (str): The input text.
    
    Returns:
    list: A list of word embeddings.
    """
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    filtered_tokens = remove_stopwords(tokens)
    embeddings = embed_text(filtered_tokens)
    return embeddings
