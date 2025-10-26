import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime
import re
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
import spacy
from collections import Counter
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from newsapi import NewsApiClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import pandas as pd

start_time = time.time()

# ------------------ Setup ------------------ #
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

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

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

GENERIC_NOUNS = {"business", "company", "market", "economy", "government", "state", "people", "industry"}
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

# ------------------ STOCK SCORING FUNCTIONS ------------------ #
def fetch_stock_data(ticker, period='3mo'):
    """Fetch stock data and calculate metrics"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty or len(hist) < 5:
            return None
        
        info = stock.info
        current_price = hist['Close'].iloc[-1]
        
        # Price changes
        price_5d_ago = hist['Close'].iloc[-5] if len(hist) >= 5 else hist['Close'].iloc[0]
        price_30d_ago = hist['Close'].iloc[-30] if len(hist) >= 30 else hist['Close'].iloc[0]
        price_60d_ago = hist['Close'].iloc[-60] if len(hist) >= 60 else hist['Close'].iloc[0]
        
        change_5d = ((current_price - price_5d_ago) / price_5d_ago) * 100
        change_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100
        change_60d = ((current_price - price_60d_ago) / price_60d_ago) * 100
        
        # Volume metrics
        avg_volume_30d = hist['Volume'].tail(30).mean()
        current_volume = hist['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume_30d if avg_volume_30d > 0 else 1.0
        
        # Volatility
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * 100
        
        # Moving averages
        ma_50 = hist['Close'].tail(50).mean() if len(hist) >= 50 else hist['Close'].mean()
        ma_200 = hist['Close'].tail(200).mean() if len(hist) >= 200 else hist['Close'].mean()
        
        # 52-week range
        high_52w = hist['High'].max()
        low_52w = hist['Low'].min()
        price_position = ((current_price - low_52w) / (high_52w - low_52w)) * 100 if high_52w != low_52w else 50
        
        # RSI
        rsi = calculate_rsi(hist['Close'], period=14)
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'change_5d': change_5d,
            'change_30d': change_30d,
            'change_60d': change_60d,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'ma_50': ma_50,
            'ma_200': ma_200,
            'price_vs_ma50': ((current_price - ma_50) / ma_50) * 100,
            'price_vs_ma200': ((current_price - ma_200) / ma_200) * 100,
            'price_position': price_position,
            'rsi': rsi
        }
    except Exception as e:
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50.0

def normalize_score(value, min_val, max_val):
    """Normalize value to 0-1"""
    if max_val == min_val:
        return 0.5
    return max(0, min(1, (value - min_val) / (max_val - min_val)))

def calculate_momentum_score(stock_data):
    """Calculate momentum score"""
    momentum_raw = (
        stock_data['change_5d'] * 0.4 +
        stock_data['change_30d'] * 0.4 +
        stock_data['change_60d'] * 0.2
    )
    momentum_score = normalize_score(momentum_raw, -20, 20)
    
    ma_bonus = 0
    if stock_data['price_vs_ma50'] > 0:
        ma_bonus += 0.1
    if stock_data['price_vs_ma200'] > 0:
        ma_bonus += 0.1
    if stock_data['ma_50'] > stock_data['ma_200']:
        ma_bonus += 0.1
    
    return min(1.0, momentum_score + ma_bonus)

def calculate_volume_score(stock_data):
    """Calculate volume score"""
    return normalize_score(stock_data['volume_ratio'], 0.5, 2.5)

def calculate_volatility_score(stock_data):
    """Calculate volatility score (lower is better)"""
    volatility_penalty = normalize_score(stock_data['volatility'], 0, 10)
    return 1 - volatility_penalty

def calculate_position_score(stock_data):
    """Calculate price position score"""
    position = stock_data['price_position']
    if 30 <= position <= 70:
        return 1.0
    elif position < 30:
        return normalize_score(position, 0, 30) * 0.7
    else:
        return normalize_score(100 - position, 0, 30) * 0.6

def calculate_rsi_score(stock_data):
    """Calculate RSI score"""
    rsi = stock_data['rsi']
    if 40 <= rsi <= 60:
        return 1.0
    elif rsi < 40:
        return normalize_score(rsi, 20, 40) * 0.8
    else:
        return normalize_score(100 - rsi, 20, 40) * 0.6

def calculate_composite_stock_score(stock_data, weights=None):
    """Calculate composite stock score"""
    if weights is None:
        weights = {
            'momentum': 0.35,
            'volume': 0.25,
            'volatility': 0.15,
            'position': 0.15,
            'rsi': 0.10
        }
    
    scores = {
        'momentum': calculate_momentum_score(stock_data),
        'volume': calculate_volume_score(stock_data),
        'volatility': calculate_volatility_score(stock_data),
        'position': calculate_position_score(stock_data),
        'rsi': calculate_rsi_score(stock_data)
    }
    
    composite = sum(scores[k] * weights[k] for k in scores.keys())
    
    return {
        'individual_scores': scores,
        'composite_score': composite,
        'weights': weights
    }

# ------------------ CSV & TEXT PROCESSING ------------------ #
def load_companies_from_csv(csv_path='companies.csv', sample_size=1000, force_include_tickers=None):
    """Load companies from CSV"""
    companies = []
    forced_companies = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            all_companies = []
            
            for row in reader:
                row = {k.strip(): v for k, v in row.items()}
                ticker = row.get('Ticker', '').strip()
                name = row.get('Name', '').strip()
                description = row.get('Description', '').strip()
                keywords = row.get('Keywords', '').strip()
                
                if ticker and name:
                    combined_text = f"{name} {description} {keywords}"
                    company_data = {
                        'ticker': ticker,
                        'name': name,
                        'name_lower': name.lower(),
                        'ticker_lower': ticker.lower(),
                        'description': description,
                        'keywords': keywords,
                        'combined_text': combined_text.lower()
                    }
                    
                    if force_include_tickers and ticker in force_include_tickers:
                        forced_companies.append(company_data)
                        print(f"  ✓ Force including: {ticker} - {name}")
                    else:
                        all_companies.append(company_data)
            
            companies.extend(forced_companies)
            remaining_slots = sample_size - len(forced_companies)
            if remaining_slots > 0 and len(all_companies) > remaining_slots:
                companies.extend(random.sample(all_companies, remaining_slots))
                print(f"✅ Loaded {len(forced_companies)} priority + {remaining_slots} sampled = {len(companies)} total")
            else:
                companies.extend(all_companies)
                print(f"✅ Loaded all {len(companies)} companies")
                
        return companies
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return []

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower()
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)

def extract_keywords(text):
    if len(text) > 50000:
        text = text[:50000]
    doc = nlp(text)
    return " ".join([
        token.lemma_.lower() 
        for token in doc 
        if token.pos_ in ["NOUN", "PROPN"] and token.lemma_.lower() not in GENERIC_NOUNS
    ])

def extract_entities(text):
    if len(text) > 50000:
        text = text[:50000]
    doc = nlp(text)
    return " ".join([
        ent.text.lower() 
        for ent in doc.ents 
        if ent.label_ in ["ORG", "GPE", "PRODUCT"]
    ])

def generate_company_variants(ticker, name):
    """Generate company name variants"""
    variants = set()
    ticker_lower = ticker.lower()
    
    if ticker_lower in COMMON_WORD_BLACKLIST:
        variants.add(name.lower())
        return variants
    
    variants.add(ticker_lower)
    variants.add(f"${ticker_lower}")
    variants.add(name.lower())
    
    name_clean = re.sub(r'\b(inc\.?|corp\.?|corporation|company|ltd\.?|limited|llc|plc)\b', '', name.lower(), flags=re.IGNORECASE).strip()
    if name_clean and len(name_clean) >= 3:
        variants.add(name_clean)
    
    words = name.split()
    if len(words) >= 2:
        acronym = ''.join([w[0] for w in words if w and (w[0].isupper() or len(w) > 3)])
        if len(acronym) >= 3 and acronym.lower() not in COMMON_WORD_BLACKLIST:
            variants.add(acronym.lower())
    
    if len(ticker_lower) >= 4:
        for i in range(1, len(ticker_lower)):
            variant = ticker_lower[:i] + ' ' + ticker_lower[i:]
            parts = variant.split()
            if all(len(p) >= 2 for p in parts):
                variants.add(variant)
        
    return variants

def count_explicit_mentions(text_lower, company):
    """Count explicit mentions"""
    variants = generate_company_variants(company['ticker'], company['name'])
    mention_count = 0
    mentioned_variants = []
    
    for variant in variants:
        if len(variant) < 3 and variant in COMMON_WORD_BLACKLIST:
            continue
            
        if ' ' in variant:
            pattern = r'\b' + re.escape(variant) + r'\b'
        else:
            pattern = r'\b' + re.escape(variant) + r'\b'
        
        matches = re.findall(pattern, text_lower)
        if matches:
            if variant in COMMON_WORD_BLACKLIST:
                continue
            mention_count += len(matches)
            mentioned_variants.append(f"{variant} ({len(matches)}x)")
    
    return mention_count, mentioned_variants

def fetch_article_body(url):
    """Fetch full article content"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 
                                       'aside', 'form', 'button', 'iframe']):
            element.decompose()
        
        for element in soup.find_all(class_=re.compile(r'(nav|menu|sidebar|footer|header|cookie|newsletter|subscribe|comment|social|share)', re.I)):
            element.decompose()
        
        for element in soup.find_all(id=re.compile(r'(nav|menu|sidebar|footer|header|cookie|newsletter|subscribe|comment|social|share)', re.I)):
            element.decompose()
        
        article_content = None
        article_selectors = [
            'article', '[class*="article"]', '[class*="content"]',
            '[class*="post"]', '[id*="article"]', '[id*="content"]', 'main'
        ]
        
        for selector in article_selectors:
            article_content = soup.select_one(selector)
            if article_content:
                paragraphs = article_content.find_all("p")
                break
        
        if not article_content:
            paragraphs = soup.find_all("p")
        
        text_parts = []
        for p in paragraphs:
            p_text = p.get_text(strip=True)
            if len(p_text) > 40:
                text_parts.append(p_text)
        
        text = " ".join(text_parts)
        if len(text) > 100000:
            return None
            
        return text if len(text) > 300 else None
        
    except:
        return None

def collect_articles_from_newsapi(api_key, query, from_date, to_date, max_articles=100, max_workers=15):
    """Collect articles from NewsAPI"""
    newsapi = NewsApiClient(api_key=api_key)
    print(f"🔍 Fetching articles from NewsAPI for: {query}")
    
    try:
        response = newsapi.get_everything(
            q=query, from_param=from_date, to=to_date,
            language='en', sort_by='relevancy',
            page_size=min(max_articles, 100)
        )
        
        articles_data = response.get('articles', [])
        print(f"  📰 Found {len(articles_data)} articles")
        
        enhanced_articles = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_article = {
                executor.submit(fetch_article_body, article.get('url')): article
                for article in articles_data
            }
            
            completed = 0
            for future in as_completed(future_to_article):
                completed += 1
                article = future_to_article[future]
                
                try:
                    full_text = future.result()
                    if full_text and len(full_text) > 300:
                        enhanced_articles.append({
                            'title': article.get('title', 'No title'),
                            'link': article.get('url'),
                            'snippet': article.get('description', ''),
                            'date': article.get('publishedAt', ''),
                            'fulltext': full_text
                        })
                        print(f"  [{completed}/{len(articles_data)}] ✅")
                    else:
                        print(f"  [{completed}/{len(articles_data)}] ⚠️ Skipped")
                except:
                    print(f"  [{completed}/{len(articles_data)}] ❌")
        
        print(f"  ✅ Fetched {len(enhanced_articles)} articles")
        return enhanced_articles
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return []

# ------------------ COMPANY MATCHING WITH STOCK SCORING ------------------ #
def match_companies_with_stock_scores(articles, topic_model, companies_list, top_n=10, similarity_threshold=0.15):
    """Match companies and calculate stock scores"""
    topic_companies = {}
    
    print("\n🔄 Processing articles...")
    for article in articles:
        if 'processed' not in article:
            article['keywords'] = extract_keywords(article["fulltext"])
            article['entities'] = extract_entities(article["fulltext"])
            article['cleaned'] = clean_text(
                article["fulltext"] + " " + (article['keywords']*3) + " " + (article['entities']*2)
            )
            article['processed'] = True
    
    docs = [a['cleaned'] for a in articles]
    topics, _ = topic_model.transform(docs)
    
    topic_articles = {}
    for article, topic_id in zip(articles, topics):
        if topic_id == -1:
            continue
        if topic_id not in topic_articles:
            topic_articles[topic_id] = []
        topic_articles[topic_id].append(article["fulltext"])
    
    print(f"🔄 Matching {len(companies_list)} companies to {len(topic_articles)} topics...")
    
    company_texts = [c['combined_text'] for c in companies_list]
    company_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
    company_vectors = company_vectorizer.fit_transform(company_texts)
    
    for topic_id, article_texts in topic_articles.items():
        print(f"  Processing Topic {topic_id}...", end=' ')
        
        topic_text = " ".join(article_texts)
        topic_text_cleaned = clean_text(topic_text)
        topic_text_lower = topic_text.lower()
        topic_keywords = " ".join([word for word, _ in topic_model.get_topic(topic_id)])
        combined_topic_text = topic_text_cleaned + " " + (topic_keywords * 5)
        
        try:
            topic_vector = company_vectorizer.transform([combined_topic_text])
            similarities = cosine_similarity(topic_vector, company_vectors)[0]
            
            company_scores = []
            for idx, company in enumerate(companies_list):
                similarity = similarities[idx]
                mention_count, mentioned_variants = count_explicit_mentions(topic_text_lower, company)
                
                mention_boost = 0
                if mention_count > 0:
                    mention_boost = 0.4 + (0.1 * np.log1p(mention_count))
                
                relevance_score = similarity + mention_boost
                
                if relevance_score > similarity_threshold or mention_count > 0:
                    company_scores.append({
                        'ticker': company['ticker'],
                        'name': company['name'],
                        'relevance_score': relevance_score,
                        'mentions': mention_count,
                        'mentioned_as': mentioned_variants if mention_count > 0 else None,
                        'stock_data': None,
                        'stock_score': None
                    })
            
            company_scores.sort(key=lambda x: x['relevance_score'], reverse=True)
            topic_companies[topic_id] = company_scores[:top_n]
            print(f"✓ Found {len(company_scores[:top_n])} companies")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            topic_companies[topic_id] = []
    
    # Fetch stock data for all matched companies
    print(f"\n📈 Fetching stock data for matched companies...")
    all_tickers = set()
    for companies in topic_companies.values():
        for comp in companies:
            all_tickers.add(comp['ticker'])
    
    print(f"  Total unique tickers: {len(all_tickers)}")
    
    stock_cache = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(fetch_stock_data, ticker): ticker
            for ticker in all_tickers
        }
        
        completed = 0
        for future in as_completed(future_to_ticker):
            completed += 1
            ticker = future_to_ticker[future]
            try:
                stock_data = future.result()
                if stock_data:
                    stock_scores = calculate_composite_stock_score(stock_data)
                    stock_cache[ticker] = {
                        'data': stock_data,
                        'score': stock_scores['composite_score'],
                        'scores': stock_scores
                    }
                    print(f"  [{completed}/{len(all_tickers)}] ✅ {ticker}: {stock_scores['composite_score']:.3f}")
                else:
                    print(f"  [{completed}/{len(all_tickers)}] ⚠️ {ticker}: No data")
            except Exception as e:
                print(f"  [{completed}/{len(all_tickers)}] ❌ {ticker}: Error")
    
    # Update companies with stock scores
    for topic_id, companies in topic_companies.items():
        for comp in companies:
            if comp['ticker'] in stock_cache:
                comp['stock_data'] = stock_cache[comp['ticker']]['data']
                comp['stock_score'] = stock_cache[comp['ticker']]['score']
                comp['stock_scores_detail'] = stock_cache[comp['ticker']]['scores']
        
        # Sort by stock score (companies without scores go to bottom)
        companies.sort(key=lambda x: x['stock_score'] if x['stock_score'] is not None else -1, reverse=True)
    
    return topic_companies

# ------------------ BERTOPIC ANALYSIS ------------------ #
def run_bertopic_with_stock_scoring(articles, companies_list):
    print(f"\n🔄 Pre-processing {len(articles)} articles...")
    for article in articles:
        article['keywords'] = extract_keywords(article["fulltext"])
        article['entities'] = extract_entities(article["fulltext"])
        article['cleaned'] = clean_text(
            article["fulltext"] + " " + (article['keywords']*3) + " " + (article['entities']*2)
        )
        article['processed'] = True
    
    docs = [a['cleaned'] for a in articles]
    print(f"\n🤖 Running BERTopic on {len(docs)} documents...")
    
    if len(docs) < 30:
        min_topic_size = max(2, len(docs) // 10)
        nr_topics = None
        print(f"⚠️ Small dataset. Using min_topic_size={min_topic_size}")
    else:
        min_topic_size = 8
        nr_topics = "auto"
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="english",
        min_topic_size=min_topic_size,
        n_gram_range=(1,4),
        top_n_words=10,
        nr_topics=nr_topics,
        calculate_probabilities=False
    )

    topics, _ = topic_model.fit_transform(docs)

    info = topic_model.get_topic_info()
    print("\n" + "="*80)
    print("=== TOP TOPICS ===")
    print("="*80)
    for _, row in info.head(15).iterrows():
        if row['Topic'] != -1:
            keywords = ", ".join([word for word, _ in topic_model.get_topic(row['Topic'])])
            print(f"Topic {row['Topic']}: {keywords}")

    topic_companies = match_companies_with_stock_scores(articles, topic_model, companies_list, top_n=50)
    
    print("\n" + "="*80)
    print("=== COMPANIES PER TOPIC (Ranked by Stock Score) ===")
    print("="*80)
    for topic_id in sorted(topic_companies.keys()):
        keywords = ", ".join([word for word, _ in topic_model.get_topic(topic_id)])
        print(f"\n📊 Topic {topic_id}: {keywords}")
        
        companies = topic_companies[topic_id]
        if companies:
            print(f"{'Rank':<5} {'Ticker':<8} {'Stock Score':<12} {'Relevance':<12} {'Name':<40} {'Mentions'}")
            print("-" * 100)
            for rank, comp in enumerate(companies, 1):
                stock_score_str = f"{comp['stock_score']:.3f}" if comp['stock_score'] is not None else "N/A"
                relevance_str = f"{comp['relevance_score']:.3f}"
                mention_info = ""
                if comp['mentions'] > 0:
                    mention_info = f"🎯 {comp['mentions']} ({', '.join(comp['mentioned_as'][:2])})"
                
                print(f"{rank:<5} {comp['ticker']:<8} {stock_score_str:<12} {relevance_str:<12} {comp['name'][:40]:<40} {mention_info}")
        else:
            print("  No matching companies")

    return topic_model, topics, topic_companies

# ------------------ MAIN ------------------ #
if __name__ == "__main__":
    NEWSAPI_KEY = '0c6458185614471e85f31fd67f473e69'
    FROM_DATE = '2025-09-19'
    TO_DATE = '2025-10-19'
    
    companies_list = load_companies_from_csv('companies.csv', sample_size=9872)
    
    if not companies_list:
        print("⚠️ No companies loaded. Exiting...")
        exit()
    
    topic_groups = [
        "defense OR military OR weapons",
        "space OR aerospace OR satellite",
        "technology OR innovation OR 5g OR telecom",
        "ai OR artificial intelligence OR robotics",
        "energy OR renewable OR climate",
        "cybersecurity OR drone",
        "Quantum OR Post-Quantum OR Quantum Cryptography OR semiconductor OR batteries",
        "Post Quantum OR quantum protection OR quantum cryptography OR PCQ OR quantum semiconductor"
    ]

    all_news_items = []

    for query in topic_groups:
        articles = collect_articles_from_newsapi(
            api_key=NEWSAPI_KEY,
            query=query,
            from_date=FROM_DATE,
            to_date=TO_DATE,
            max_articles=100,
            max_workers=15
        )
        all_news_items.extend(articles)
        time.sleep(1)

    print(f"\n{'='*80}")
    print(f"✅ Total collected: {len(all_news_items)} articles")
    print(f"{'='*80}")

    if len(all_news_items) >= 10:
        model, topics, topic_companies = run_bertopic_with_stock_scoring(all_news_items, companies_list)
        
        # Export detailed results to CSV
        print(f"\n📊 Exporting detailed results to CSV...")
        with open('topic_companies_ranked.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Topic_ID', 'Topic_Keywords', 'Rank', 'Ticker', 'Company_Name',
                'Stock_Score', 'Relevance_Score', 'Mentions', 'Mentioned_As',
                'Momentum', 'Volume', 'Volatility', 'Position', 'RSI',
                'Price', 'Change_5d', 'Change_30d', 'Change_60d'
            ])
            
            for topic_id in sorted(topic_companies.keys()):
                keywords = ", ".join([word for word, _ in model.get_topic(topic_id)])
                
                for rank, comp in enumerate(topic_companies[topic_id], 1):
                    stock_score = comp['stock_score'] if comp['stock_score'] is not None else ''
                    mentioned_as = '; '.join(comp['mentioned_as']) if comp['mentioned_as'] else ''
                    
                    # Extract detailed scores if available
                    momentum = volume = volatility = position = rsi = ''
                    price = change_5d = change_30d = change_60d = ''
                    
                    if comp.get('stock_scores_detail'):
                        scores = comp['stock_scores_detail']['individual_scores']
                        momentum = f"{scores['momentum']:.3f}"
                        volume = f"{scores['volume']:.3f}"
                        volatility = f"{scores['volatility']:.3f}"
                        position = f"{scores['position']:.3f}"
                        rsi = f"{scores['rsi']:.3f}"
                    
                    if comp.get('stock_data'):
                        data = comp['stock_data']
                        price = f"{data['current_price']:.2f}"
                        change_5d = f"{data['change_5d']:.2f}"
                        change_30d = f"{data['change_30d']:.2f}"
                        change_60d = f"{data['change_60d']:.2f}"
                    
                    writer.writerow([
                        topic_id, keywords, rank, comp['ticker'], comp['name'],
                        stock_score, f"{comp['relevance_score']:.3f}",
                        comp['mentions'], mentioned_as,
                        momentum, volume, volatility, position, rsi,
                        price, change_5d, change_30d, change_60d
                    ])
        
        print(f"✅ Results exported to 'topic_companies_ranked.csv'")
        
        # Print top performers across all topics
        print(f"\n{'='*80}")
        print("=== TOP 20 STOCKS BY STOCK SCORE (Across All Topics) ===")
        print(f"{'='*80}")
        
        all_companies = []
        for topic_id, companies in topic_companies.items():
            for comp in companies:
                if comp['stock_score'] is not None:
                    all_companies.append({
                        'topic_id': topic_id,
                        'ticker': comp['ticker'],
                        'name': comp['name'],
                        'stock_score': comp['stock_score'],
                        'relevance_score': comp['relevance_score'],
                        'mentions': comp['mentions']
                    })
        
        all_companies.sort(key=lambda x: x['stock_score'], reverse=True)
        
        print(f"{'Rank':<5} {'Ticker':<8} {'Stock Score':<12} {'Relevance':<12} {'Topic':<8} {'Company Name':<40}")
        print("-" * 100)
        for rank, comp in enumerate(all_companies[:20], 1):
            print(f"{rank:<5} {comp['ticker']:<8} {comp['stock_score']:.3f}        {comp['relevance_score']:.3f}        {comp['topic_id']:<8} {comp['name'][:40]}")
    
    else:
        print("❌ Not enough articles to analyze. Need at least 10 articles.")
        print("   Try expanding date range or adding more topics.")
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'='*80}")
    print(f"⏱️ Script completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*80}")