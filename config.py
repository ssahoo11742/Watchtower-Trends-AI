import nltk
from nltk.corpus import stopwords
import spacy
"""
Configuration settings for the stock analysis system
"""
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# API Configuration
NEWSAPI_KEY = '0c6458185614471e85f31fd67f473e69'
FROM_DATE = '2025-09-26'
TO_DATE = '2025-10-26'

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

COMMON_WORD_BLACKLIST = {
    'in', 'as', 'ai', 'is', 'it', 'at', 'an', 'or', 'on', 'by', 'to', 'of', 'for',
    'the', 'and', 'but', 'not', 'are', 'was', 'has', 'had', 'can', 'may', 'will',
    'be', 'am', 'us', 'we', 'he', 'she', 'so', 'do', 'go', 'no', 'up', 'if', 'me',
    'my', 'oh', 'hi', 'ok', 'ok', 'vs', 'via', 'per', 'etc', 'eg', 'ie', 'ab', 'de',
    'el', 'la', 'le', 'et', 'se', 'ca', 'co', 'ma', 'pa', 'da', 'di', 'ti', 'si',
    'pi', 'ic', 'pc', 'vc', 'cc', 'ii', 'ad', 'all', 'any', 'our', 'out', 'own',
    'well', 'just', 'open', 'top', 'care', 'bill', 'fast', 'rare', 'heat', 'pro',
    'cars', 'pool', 'caps', 'keys', 'wise', 'lake', 'drug', 'am'
}

# Analysis Settings
SAMPLE_SIZE = 9000
MIN_ARTICLE_LENGTH = 300
MAX_ARTICLE_LENGTH = 100000
MAX_TEXT_LENGTH = 50000
MIN_ARTICLES_FOR_ANALYSIS = 10
MAX_ARTICLES = 50

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