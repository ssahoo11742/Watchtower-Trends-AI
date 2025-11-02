import nltk
from nltk.corpus import stopwords
import spacy
from datetime import datetime, timedelta

TO_DATE   = datetime.now().date()              # today
FROM_DATE = TO_DATE - timedelta(days=30) + timedelta(days=2)
"""
Configuration settings for the stock analysis system
"""
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# API Configuration
NEWSAPI_KEY = '0c6458185614471e85f31fd67f473e69'
FROM_DATE = FROM_DATE.strftime('%Y-%m-%d')
TO_DATE   = TO_DATE.strftime('%Y-%m-%d')

# HTTP Headers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Text Processing Settings
BOILERPLATE_TERMS = {
    'email', 'digest', 'homepage', 'feed', 'newsletter', 'subscribe', 'subscription',
    'menu', 'navigation', 'sidebar', 'footer', 'header', 'cookie', 'privacy',
    'policy', 'terms', 'service', 'copyright', 'reserved', 'rights', 'contact',
    'facebook', 'twitter', 'instagram', 'linkedin', 'youtube', 'social', 'share',
    'comment', 'comments', 'reply', 'login', 'signup', 'register', 'search',
    'advertisement', 'sponsored', 'promo', 'promotion'
}

STOPWORDS.update(BOILERPLATE_TERMS)

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

GENERIC_NOUNS = {
    "business", "company", "market", "economy", "government", 
    "state", "people", "industry"
}

COMMON_WORD_BLACKLIST = set([
    # Single letters (all of them)
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    
    # Common two-letter words
    'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'hi', 'if',
    'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to',
    'up', 'us', 'we',
    
    # Common three-letter words that are also tickers
    'all', 'and', 'are', 'but', 'can', 'car', 'cat', 'day', 'did', 'dog',
    'eat', 'far', 'few', 'for', 'fun', 'get', 'got', 'had', 'has', 'her',
    'him', 'his', 'how', 'its', 'let', 'man', 'may', 'new', 'not', 'now',
    'old', 'one', 'our', 'out', 'own', 'put', 'ran', 'red', 'run', 'said',
    'saw', 'say', 'see', 'set', 'she', 'sit', 'six', 'ten', 'the', 'too',
    'top', 'two', 'use', 'was', 'way', 'who', 'why', 'win', 'yes', 'yet',
    'you',
    
    # Business/tech words that are tickers
    'app', 'box', 'car', 'data', 'file', 'key', 'link', 'live', 'main',
    'net', 'open', 'post', 'real', 'site', 'tech', 'text', 'true', 'type',
    'uber', 'user', 'web', 'work', 'zoom'
])

# Analysis Settings
SAMPLE_SIZE = 10229
MIN_ARTICLE_LENGTH = 300
MAX_ARTICLE_LENGTH = 100000
MAX_TEXT_LENGTH = 50000
MIN_ARTICLES_FOR_ANALYSIS = 10
MAX_ARTICLES = 500

# Topic Modeling Settings
MIN_TOPIC_SIZE = 8
TOP_N_COMPANIES = 50
SIMILARITY_THRESHOLD = 0.15

# Multithreading
MAX_WORKERS_ARTICLES = 15
MAX_WORKERS_STOCKS = 10

# Search Queries
TOPIC_GROUPS = [
        # "defense OR military OR weapons",
        # "space OR aerospace OR satellite",
        # "technology OR innovation OR 5g OR telecom",
        # "ai OR artificial intelligence OR robotics",
        # "energy OR renewable OR climate",
        # "cybersecurity OR drone",
        "Quantum OR Post-Quantum OR Quantum Cryptography OR semiconductor OR batteries",
        "Post Quantum OR quantum protection OR quantum cryptography OR PCQ OR quantum semiconductor"
]