import time

import spacy
from output import run_bertopic_with_multitimeframe_scoring
from text_processing import load_companies_from_csv, collect_articles_from_newsapi
from config import NEWSAPI_KEY, FROM_DATE, TO_DATE, TOPIC_GROUPS, MAX_WORKERS_ARTICLES, SAMPLE_SIZE, MAX_ARTICLES
start_time = time.time()

if __name__ == "__main__":
    
    companies_list = load_companies_from_csv('data/companies.csv', sample_size=SAMPLE_SIZE)
    
    if not companies_list:
        print("⚠️ No companies loaded. Exiting...")
        exit()
    

    all_news_items = []

    for query in TOPIC_GROUPS:
        articles = collect_articles_from_newsapi(
            api_key=NEWSAPI_KEY,
            query=query,
            from_date=FROM_DATE,
            to_date=TO_DATE,
            max_articles=MAX_ARTICLES,
            max_workers=MAX_WORKERS_ARTICLES
        )
        all_news_items.extend(articles)
        time.sleep(1)

    print(f"\n{'='*80}")
    print(f"✅ Total collected: {len(all_news_items)} articles")
    print(f"{'='*80}")

    if len(all_news_items) >= 10:
        model, topics, topic_companies = run_bertopic_with_multitimeframe_scoring(all_news_items, companies_list)
    else:
        print("❌ Not enough articles to analyze. Need at least 10 articles.")
        print("   Try expanding date range or adding more topics.")
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'='*80}")
    print(f"⏱️ Script completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*80}")