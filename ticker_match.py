from text_processing import extract_entities, extract_keywords, clean_text, count_explicit_mentions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from scoring import fetch_comprehensive_stock_data, calculate_day_trader_score, calculate_longterm_investor_score,calculate_position_trader_score,calculate_swing_trader_score
import numpy as np

def match_companies_with_multitimeframe_scores(articles, topic_model, companies_list, top_n=50, similarity_threshold=0.15):
    """Match companies and calculate multi-timeframe stock scores"""
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
                        'day_trader_score': None,
                        'swing_trader_score': None,
                        'position_trader_score': None,
                        'longterm_investor_score': None
                    })
            
            company_scores.sort(key=lambda x: x['relevance_score'], reverse=True)
            topic_companies[topic_id] = company_scores[:top_n]
            print(f"✓ Found {len(company_scores[:top_n])} companies")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            topic_companies[topic_id] = []
    
    # Fetch stock data for all matched companies
    print(f"\n📈 Fetching multi-timeframe stock data for matched companies...")
    all_tickers = set()
    for companies in topic_companies.values():
        for comp in companies:
            all_tickers.add(comp['ticker'])
    
    print(f"  Total unique tickers: {len(all_tickers)}")
    
    stock_cache = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(fetch_comprehensive_stock_data, ticker): ticker
            for ticker in all_tickers
        }
        
        completed = 0
        for future in as_completed(future_to_ticker):
            completed += 1
            ticker = future_to_ticker[future]
            try:
                stock_data = future.result()
                if stock_data:
                    # Calculate all timeframe scores
                    day_score = calculate_day_trader_score(stock_data)
                    swing_score = calculate_swing_trader_score(stock_data)
                    position_score = calculate_position_trader_score(stock_data)
                    longterm_score = calculate_longterm_investor_score(stock_data)
                    
                    stock_cache[ticker] = {
                        'data': stock_data,
                        'day_trader': day_score,
                        'swing_trader': swing_score,
                        'position_trader': position_score,
                        'longterm_investor': longterm_score
                    }
                    print(f"  [{completed}/{len(all_tickers)}] ✅ {ticker}: Day:{day_score['score']:.2f} Swing:{swing_score['score']:.2f} Pos:{position_score['score']:.2f} LT:{longterm_score['score']:.2f}")
                else:
                    print(f"  [{completed}/{len(all_tickers)}] ⚠️ {ticker}: No data")
            except Exception as e:
                print(f"  [{completed}/{len(all_tickers)}] ❌ {ticker}: Error")
    
    # Update companies with all timeframe scores
    for topic_id, companies in topic_companies.items():
        for comp in companies:
            if comp['ticker'] in stock_cache:
                cache = stock_cache[comp['ticker']]
                comp['stock_data'] = cache['data']
                comp['day_trader_score'] = cache['day_trader']['score'] if cache['day_trader'] else None
                comp['swing_trader_score'] = cache['swing_trader']['score'] if cache['swing_trader'] else None
                comp['position_trader_score'] = cache['position_trader']['score'] if cache['position_trader'] else None
                comp['longterm_investor_score'] = cache['longterm_investor']['score'] if cache['longterm_investor'] else None
                comp['day_trader_category'] = cache['day_trader']['category'] if cache['day_trader'] else None
                comp['swing_trader_category'] = cache['swing_trader']['category'] if cache['swing_trader'] else None
                comp['position_trader_category'] = cache['position_trader']['category'] if cache['position_trader'] else None
                comp['longterm_investor_category'] = cache['longterm_investor']['category'] if cache['longterm_investor'] else None
    
    return topic_companies