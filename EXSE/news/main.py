"""
news_factor_engine.py - News Factor Engine with Fixed Company Matching

Pipeline: NER Extraction → Semantic Matching → FinBERT Sentiment → Time Decay → News Factor

Fixes applied:
1. Averaged keyword embeddings (not single string)
2. Name hard-floor if company/ticker mentioned in text
3. Reduced sector weight (90% name, 10% sector)
4. SpaCy NER for first-pass company detection
"""

import csv
import sys
import json
import re
import pickle
import hashlib
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy

# Increase CSV field size limit (Windows-compatible)
csv.field_size_limit(10 * 1024 * 1024)  # 10MB limit

# ==================== CONFIG ====================
EMB_MODEL = "BAAI/bge-large-en"
FINBERT_MODEL = "ProsusAI/finbert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== INIT ====================
semantic_model = SentenceTransformer(EMB_MODEL, device=DEVICE)
finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(DEVICE)
finbert_model.eval()

# Load SpaCy for NER
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("SpaCy model not found. Installing...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

cache = {}

def load_cache(path):
    """Load pickle cache if exists"""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except:
        return {}

def save_cache(path, data):
    """Save pickle cache"""
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# ==================== TEXT PROCESSING ====================
def clean_text(text):
    """Clean and normalize text"""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ==================== NER EXTRACTION ====================
def extract_organizations_ner(text):
    """Extract organization names using SpaCy NER"""
    doc = nlp(text[:10000])  # Limit text length for performance
    orgs = set()
    for ent in doc.ents:
        if ent.label_ == "ORG":
            orgs.add(ent.text.lower())
    return orgs

# ==================== COMPANY MATCHING ====================
def load_companies(csv_path):
    """Load companies from CSV, excluding ETFs"""
    companies = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            # Skip ETFs
            if 'ETF' in row['Name'].upper():
                continue
            companies.append({
                'ticker': row['Ticker'],
                'name': row['Name'],
                'keywords': row.get('Keywords', '')
            })
    return companies

def embed_keywords(keywords, semantic_model):
    """
    Encode keywords individually and average them (FIX #1)
    This prevents generic keyword strings from overwhelming name embeddings
    """
    if not keywords or keywords.strip() == '':
        return None
    
    # Split keywords by comma and clean
    words = [w.strip() for w in keywords.split(',') if w.strip()]
    
    if not words:
        return None
    
    # Encode each keyword separately
    embs = semantic_model.encode(words, normalize_embeddings=True, convert_to_numpy=True)
    
    # Average the embeddings
    avg_emb = np.mean(embs, axis=0)
    
    # Renormalize
    norm = np.linalg.norm(avg_emb)
    if norm > 0:
        avg_emb = avg_emb / norm
    
    return avg_emb

def get_company_embeddings(companies):
    """Get or compute company embeddings (2-vector system with averaged keywords)"""
    cache_key = hashlib.md5(
        (EMB_MODEL + str(len(companies)) + ''.join([c['ticker'] for c in companies])).encode()
    ).hexdigest()
    cache_file = f"company_embs_fixed_{cache_key}.pkl"
    
    cache = load_cache(cache_file)
    if 'embeddings' in cache:
        return cache['embeddings']
    
    # Pure company name embeddings (specific)
    name_texts = [c['name'] for c in companies]
    name_embs = semantic_model.encode(name_texts, normalize_embeddings=True, convert_to_numpy=True)
    
    # Sector keyword embeddings (FIX #1: averaged individual keywords)
    print("Computing averaged keyword embeddings...")
    sector_embs = []
    for c in companies:
        if c['keywords']:
            keyword_emb = embed_keywords(c['keywords'], semantic_model)
            if keyword_emb is not None:
                sector_embs.append(keyword_emb)
            else:
                # Fallback to name if keywords are empty/invalid
                sector_embs.append(semantic_model.encode(c['name'], normalize_embeddings=True, convert_to_numpy=True))
        else:
            # Use name as fallback
            sector_embs.append(semantic_model.encode(c['name'], normalize_embeddings=True, convert_to_numpy=True))
    
    sector_embs = np.array(sector_embs)
    
    embeddings = {
        'name': name_embs,
        'sector': sector_embs
    }
    save_cache(cache_file, {'embeddings': embeddings})
    return embeddings

def match_companies(article_text, companies, company_embs, top_k=5, threshold=0.1):
    """
    Match companies using FIXED 2-vector system with:
    - NER extraction (FIX #4)
    - Name hard-floor (FIX #2)
    - Reduced sector weight (FIX #3: 90% name, 10% sector)
    """
    
    # Step 1: Extract organizations using NER (FIX #4)
    article_lower = article_text.lower()
    ner_orgs = extract_organizations_ner(article_text)
    
    # Step 2: Compute semantic embeddings
    article_emb = semantic_model.encode(article_text[:1000], normalize_embeddings=True, convert_to_numpy=True)
    
    # Company name similarity (specific matches)
    name_sims = np.dot(company_embs['name'], article_emb)
    
    # Sector keyword similarity (broad topic matches)
    sector_sims = np.dot(company_embs['sector'], article_emb)
    
    # Step 3: Apply name hard-floor (FIX #2)
    # If company name or ticker appears in text, force minimum similarity
    for idx, company in enumerate(companies):
        company_name_lower = company['name'].lower()
        ticker_lower = company['ticker'].lower()
        
        # Check direct mention
        if company_name_lower in article_lower or ticker_lower in article_lower:
            name_sims[idx] = max(name_sims[idx], 0.8)
        
        # Check NER matches
        for org in ner_orgs:
            if company_name_lower in org or org in company_name_lower:
                name_sims[idx] = max(name_sims[idx], 0.75)
                break
    
    # Step 4: Detect if article is sector-level or company-specific
    max_name_sim = np.max(name_sims)
    max_sector_sim = np.max(sector_sims)
    
    # If sector signal dominates, it's a general article
    is_sector_article = max_sector_sim > (max_name_sim * 1.2)
    
    # Step 5: FIX #3 - Reduced sector weight (90% name, 10% sector)
    if is_sector_article:
        # Even for sector articles, prioritize name matches more
        combined_sims = name_sims * 0.8 + sector_sims * 0.2
        article_type = "sector"
    else:
        # For specific articles, heavily favor name matches
        combined_sims = name_sims * 0.9 + sector_sims * 0.1
        article_type = "specific"
    
    valid_idx = np.where(combined_sims >= threshold)[0]
    if len(valid_idx) == 0:
        return [], article_type
    
    sorted_idx = valid_idx[np.argsort(-combined_sims[valid_idx])][:top_k]
    total_sim = combined_sims[sorted_idx].sum()
    
    matches = []
    for idx in sorted_idx:
        matches.append({
            'ticker': companies[idx]['ticker'],
            'name': companies[idx]['name'],
            'similarity': float(combined_sims[idx]),
            'relevancy': float(combined_sims[idx] / total_sim),
            'name_score': float(name_sims[idx]),
            'sector_score': float(sector_sims[idx])
        })
    return matches, article_type

# ==================== SENTIMENT ====================
def finbert_sentiment(text, max_length=512):
    """FinBERT sentiment: returns score in [-1, 1]"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not sentences:
        return 0.0
    
    sentiments = []
    with torch.no_grad():
        for sent in sentences[:20]:
            inputs = finbert_tokenizer(
                sent[:max_length], 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            ).to(DEVICE)
            
            outputs = finbert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            positive = probs[0][0].item()
            negative = probs[0][1].item()
            sentiment_score = positive - negative
            sentiments.append(sentiment_score)
    
    return float(np.mean(sentiments))

def calculate_sentiment(title, body, headline_boost=0.4):
    """Calculate article sentiment with FinBERT and headline boost"""
    title_sent = finbert_sentiment(title)
    body_sent = finbert_sentiment(body)
    final = body_sent + (title_sent * headline_boost)
    return {
        'title_sentiment': float(title_sent),
        'body_sentiment': float(body_sent),
        'final_sentiment': float(max(-1.0, min(1.0, final)))
    }

# ==================== TIME DECAY ====================
def time_decay(timestamp, half_life_hours=72):
    """Exponential time decay"""
    age_hours = (datetime.now() - timestamp).total_seconds() / 3600
    return float(np.exp(-age_hours / half_life_hours))

# ==================== MAIN PIPELINE ====================
def compute_news_factors(title, body, timestamp, companies_csv, 
                        top_k=5, threshold=0.1, half_life=72, headline_boost=0.4):
    """
    Main pipeline: compute news factors for article
    
    Args:
        title: Article title
        body: Article body text
        timestamp: datetime object
        companies_csv: Path to companies CSV
        top_k: Top K companies to match
        threshold: Similarity threshold
        half_life: Time decay half-life (hours)
        headline_boost: Headline sentiment multiplier
    
    Returns:
        Dict with 'factors' (list) and 'metadata' (dict)
    """
    
    # 1. Load companies
    companies = load_companies(companies_csv)
    company_embs = get_company_embeddings(companies)
    
    # 2. Preprocess
    title_clean = clean_text(title)
    body_clean = clean_text(body)
    article_text = f"{title_clean} {body_clean}"
    
    # 3. Match companies (FIXED 2-vector system with NER + hard-floor + reduced sector weight)
    matches, article_type = match_companies(article_text, companies, company_embs, top_k, threshold)
    
    if not matches:
        return {
            'factors': [],
            'metadata': {
                'article_sentiment': 0.0,
                'companies_matched': 0,
                'article_type': 'none',
                'timestamp': timestamp.isoformat()
            }
        }
    
    # 4. Sentiment
    sentiment = calculate_sentiment(title_clean, body_clean, headline_boost)
    
    # 5. Time decay
    decay = time_decay(timestamp, half_life)
    decayed_sentiment = sentiment['final_sentiment'] * decay
    
    # 6. Compute news factors with raw relevancy
    factors = []
    for match in matches:
        factors.append({
            'ticker': match['ticker'],
            'name': match['name'],
            'relevancy_raw': float(match['relevancy']),
            'name_score': float(match['name_score']),
            'sector_score': float(match['sector_score']),
            'sentiment': float(sentiment['final_sentiment']),
            'decay': float(decay)
        })
    
    # 7. Normalize relevancy within matched set
    if len(factors) > 1:
        scores = [f['relevancy_raw'] for f in factors]
        min_s = min(scores)
        max_s = max(scores)
        den = (max_s - min_s) + 1e-9
        
        for f in factors:
            rel_norm = (f['relevancy_raw'] - min_s) / den
            f['relevancy_norm'] = float(rel_norm)
            f['relevancy_combined'] = float(np.sqrt(f['relevancy_raw'] * rel_norm))
            f['news_factor'] = float(decayed_sentiment * f['relevancy_combined'])
    else:
        factors[0]['relevancy_norm'] = 1.0
        factors[0]['relevancy_combined'] = float(np.sqrt(factors[0]['relevancy_raw']))
        factors[0]['news_factor'] = float(decayed_sentiment * factors[0]['relevancy_combined'])
    
    return {
        'factors': factors,
        'metadata': {
            'article_sentiment': sentiment['final_sentiment'],
            'title_sentiment': sentiment['title_sentiment'],
            'body_sentiment': sentiment['body_sentiment'],
            'decay': decay,
            'companies_matched': len(factors),
            'article_type': article_type,
            'timestamp': timestamp.isoformat()
        }
    }

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    title = "Zacks.com featured highlights include QuantumScape, Silicon Laboratories and Affiliated Managers"
    body = """
For Immediate Release
Chicago, IL – November 18, 2025 – Stocks in this week's article are QuantumScape Corp. QS, Silicon Laboratories Inc. SLAB and Affiliated Managers Group, Inc. AMG.

QuantumScape Leads 3 Stocks to Buy for Fast Earnings Acceleration

The top brass of a company and analysts value steady earnings growth as a sign of a company's profitability. But an acceleration in earnings tends to have an even stronger impact on boosting stock prices. Research shows that leading stocks typically experience an acceleration in earnings before their share prices move higher.

At this moment, QuantumScape Corp., Silicon Laboratories Inc. and Affiliated Managers Group, Inc. are showing strong earnings acceleration.

What Is Earnings Acceleration?

Earnings acceleration is the incremental growth in a company's earnings per share (EPS). In other words, if a company's quarter-over-quarter earnings growth rate increases within a stipulated time frame, it can be called earnings acceleration.

In the case of earnings growth, you pay for something that is already reflected in the stock price. However, earnings acceleration helps spot stocks that haven't yet caught the attention of investors and, once secured, will invariably lead to a rally in the share price. This is because earnings acceleration considers both the direction and magnitude of growth rates.

An increasing percentage of earnings growth means that the company is fundamentally sound and has been on the right track for a considerable period. Meanwhile, a sideways percentage of earnings growth indicates a period of consolidation or slowdown, while a decelerating percentage of earnings growth may drag prices down.

The above criteria narrowed the universe of around 7,735 stocks to only nine. Here are the top three stocks:

QuantumScape

QuantumScape develops and commercializes solid-state lithium-metal batteries for electric vehicles and other applications in the United States. QuantumScape has a Zacks Rank #2 (Buy). QS's expected earnings growth rate for the current year is 21.3%. You can see the complete list of today's Zacks #1 Rank (Strong Buy) stocks here.

Silicon Laboratories

Silicon Laboratories, a fabless semiconductor company, delivers analog-intensive mixed-signal solutions in the United States and other global markets. Silicon Laboratories has a Zacks Rank #2. SLAB's expected earnings growth rate for the current year is 152.3%.
    """
    timestamp = datetime(2025, 11, 18, 3, 19, 0)
    
    results = compute_news_factors(
        title=title,
        body=body,
        timestamp=timestamp,
        companies_csv="./data/companies.csv",
        top_k=5,
        threshold=0.1,
        half_life=72,
        headline_boost=1.4
    )
    
    print("\n=== FIXED RESULTS ===")
    print(json.dumps(results, indent=2))
    print("\n✅ This should now correctly match: QS, SLAB, AMG")
    print("❌ No more random matches like ACNT, AIVL, OPY")