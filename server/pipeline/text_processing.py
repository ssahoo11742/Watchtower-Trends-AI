import requests
from bs4 import BeautifulSoup
import random
import re
import csv
from newsapi import NewsApiClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import COMMON_WORD_BLACKLIST, TOP_N_COMPANIES, TO_DATE, TOPIC_GROUPS, STOPWORDS, GENERIC_NOUNS, nlp, HEADERS
import sys
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # Windows fix: use a large but reasonable limit (10MB)
    csv.field_size_limit(10485760)


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
    """Generate company name variants with smart filtering"""
    variants = set()
    ticker_lower = ticker.lower()
    
    # CRITICAL: Skip single-letter tickers for variant generation
    if len(ticker_lower) == 1:
        # Only use full company name for single-letter tickers
        variants.add(name.lower())
        name_clean = re.sub(r'\b(inc\.?|corp\.?|corporation|company|ltd\.?|limited|llc|plc)\b', 
                           '', name.lower(), flags=re.IGNORECASE).strip()
        if name_clean and len(name_clean) >= 3:
            variants.add(name_clean)
        return variants
    
    # For 2-letter tickers, be very careful
    if len(ticker_lower) == 2 and ticker_lower in COMMON_WORD_BLACKLIST:
        variants.add(name.lower())
        return variants
    
    # Regular processing for 3+ letter tickers
    variants.add(ticker_lower)
    variants.add(f"${ticker_lower}")  # Stock notation
    variants.add(name.lower())
    
    # Clean name without corporate suffixes
    name_clean = re.sub(r'\b(inc\.?|corp\.?|corporation|company|ltd\.?|limited|llc|plc)\b', 
                       '', name.lower(), flags=re.IGNORECASE).strip()
    if name_clean and len(name_clean) >= 3:
        variants.add(name_clean)
    
    # Acronym from company name (only if 3+ chars)
    words = name.split()
    if len(words) >= 2:
        acronym = ''.join([w[0] for w in words if w and (w[0].isupper() or len(w) > 3)])
        if len(acronym) >= 3 and acronym.lower() not in COMMON_WORD_BLACKLIST:
            variants.add(acronym.lower())
    
    # Split ticker variants (only for 4+ char tickers)
    if len(ticker_lower) >= 4:
        for i in range(1, len(ticker_lower)):
            variant = ticker_lower[:i] + ' ' + ticker_lower[i:]
            parts = variant.split()
            if all(len(p) >= 2 for p in parts):
                variants.add(variant)
    
    return variants

def count_explicit_mentions(text_lower, company):
    """Count explicit mentions with context-aware validation"""
    variants = generate_company_variants(company['ticker'], company['name'])
    mention_count = 0
    mentioned_variants = []
    ticker_lower = company['ticker'].lower()
    
    for variant in variants:
        # Skip very short variants that are likely false positives
        if len(variant) < 2:
            continue
        
        # Extra strict for short tickers/variants
        if len(variant) <= 2:
            # Require $ prefix for 1-2 letter tickers
            if not variant.startswith('$'):
                continue
        
        # Build pattern based on variant type
        if ' ' in variant:
            # Multi-word phrases need exact matching
            pattern = r'\b' + re.escape(variant) + r'\b'
        elif len(variant) <= 3:
            # Short variants need very strict word boundaries
            # Check if it's preceded/followed by alphanumeric (which would make it part of another word)
            pattern = r'(?<![a-z])' + re.escape(variant) + r'(?![a-z])'
        else:
            # Regular word boundary for 4+ char variants
            pattern = r'\b' + re.escape(variant) + r'\b'
        
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        
        if matches:
            # Additional validation for short tickers (1-3 chars)
            if len(variant) <= 3 and variant in COMMON_WORD_BLACKLIST:
                continue
            
            # Context validation for short tickers
            if len(variant) <= 3 and not variant.startswith('$'):
                validated_matches = validate_short_ticker_context(text_lower, variant, matches)
                if not validated_matches:
                    continue
                matches = validated_matches
            
            mention_count += len(matches)
            mentioned_variants.append(f"{variant} ({len(matches)}x)")
    
    return mention_count, mentioned_variants


def validate_short_ticker_context(text_lower, ticker, matches):
    """Validate short ticker mentions using surrounding context"""
    validated = []
    
    # Keywords that suggest it's actually a stock ticker mention
    stock_context_keywords = [
        'stock', 'shares', 'ticker', 'nasdaq', 'nyse', 'trading', 'market',
        'corporation', 'inc', 'company', 'corp', 'limited', 'ltd',
        'price', 'earnings', 'revenue', 'valuation', 'ipo', 'traded'
    ]
    
    # Find all occurrences with context window
    pattern = r'(?<![a-z])' + re.escape(ticker) + r'(?![a-z])'
    
    for match in re.finditer(pattern, text_lower, re.IGNORECASE):
        start = max(0, match.start() - 50)  # 50 chars before
        end = min(len(text_lower), match.end() + 50)  # 50 chars after
        context = text_lower[start:end]
        
        # Check if any stock-related keywords appear in context
        if any(keyword in context for keyword in stock_context_keywords):
            validated.append(match.group())
            continue
        
        # Check for $ prefix nearby (within 5 chars before)
        prefix_start = max(0, match.start() - 5)
        prefix = text_lower[prefix_start:match.start()]
        if '$' in prefix:
            validated.append(match.group())
            continue
        
        # Check for common stock notation patterns like "ticker: A" or "(A)"
        notation_start = max(0, match.start() - 10)
        notation_context = text_lower[notation_start:match.end() + 2]
        if re.search(r'(ticker|symbol|code)[:\s]' + re.escape(ticker), notation_context):
            validated.append(match.group())
            continue
        
        if re.search(r'\(' + re.escape(ticker) + r'\)', notation_context):
            validated.append(match.group())
            continue
    
    return validated

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
        if len(text) > 50000:
            text = text[:50000]
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
