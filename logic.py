# logic.py
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

def count_words(text):
    return len(text.split())

def preprocess_text(text):
    # Convert to lowercase and tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove non-alphanumeric tokens and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    return tokens

def compute_tf(doc):
    # Calculate term frequency
    tf_dict = {}
    total_words = len(doc)
    
    if total_words == 0:
        return tf_dict
        
    for word in doc:
        tf_dict[word] = tf_dict.get(word, 0) + 1
    
    # Normalize by total words
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / total_words
        
    return tf_dict

def compute_idf(docs):
    # Calculate inverse document frequency
    idf_dict = {}
    total_docs = len(docs)
    
    # Get all unique words across all documents
    all_words = set(word for doc in docs for word in doc)
    
    for word in all_words:
        # Count documents containing the word
        doc_count = sum(1 for doc in docs if word in doc)
        # Calculate IDF with smoothing to avoid division by zero
        idf_dict[word] = math.log((total_docs + 1) / (doc_count + 1)) + 1
        
    return idf_dict

def compute_tfidf(tf, idf):
    # Calculate TF-IDF scores
    return {word: tf[word] * idf.get(word, 0) for word in tf}

def cosine_similarity(vec1, vec2):
    # Handle empty vectors
    if not vec1 or not vec2:
        return 0.0
    
    # Calculate dot product
    dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in set(vec1) & set(vec2))
    
    # Calculate magnitudes
    mag1 = math.sqrt(sum(value * value for value in vec1.values()))
    mag2 = math.sqrt(sum(value * value for value in vec2.values()))
    
    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0
        
    return dot_product / (mag1 * mag2)

def find_common_words(tfidf1, tfidf2):
    # Find words present in both documents
    common = set(tfidf1.keys()) & set(tfidf2.keys())
    # Sort by combined TF-IDF scores
    return sorted(common, key=lambda x: tfidf1[x] + tfidf2[x], reverse=True)

def word_frequencies(doc):
    # Calculate raw word frequencies
    freq = {}
    for word in doc:
        freq[word] = freq.get(word, 0) + 1
    return freq