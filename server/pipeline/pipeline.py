import time
import argparse
import json

# Parse arguments FIRST (before any imports that use ticker_match)
parser = argparse.ArgumentParser(description="Run the pipeline with custom parameters.")
parser.add_argument("-d", "--depth", type=int, default=1, help="Depth of topic modeling.")
parser.add_argument("--custom-config", type=str, help="Path to custom config JSON file")
parser.add_argument("--from-date", type=str, help="Start date (YYYY-MM-DD)")
parser.add_argument("--to-date", type=str, help="End date (YYYY-MM-DD)")
parser.add_argument("--queries", type=str, help="Comma-separated queries")
parser.add_argument("--min-topic-size", type=int, help="Minimum topic size")
parser.add_argument("--max-articles", type=int, help="Maximum articles per query")
parser.add_argument("--user-id", type=str, help="User ID for file naming")
args = parser.parse_args()

start_time = time.time()

# NOW set depth mode before other imports
from ticker_match import set_depth_mode
set_depth_mode(args.depth)

# Continue with other imports
import spacy
from output import run_bertopic_with_multitimeframe_scoring
from text_processing import load_companies_from_csv, collect_articles_from_newsapi
from config import NEWSAPI_KEY, FROM_DATE, TO_DATE, TOPIC_GROUPS, MAX_WORKERS_ARTICLES, SAMPLE_SIZE, MAX_ARTICLES, MIN_TOPIC_SIZE

if __name__ == "__main__":
    
    # Override config values if custom arguments provided
    if args.custom_config:
        with open(args.custom_config, 'r') as f:
            custom_config = json.load(f)
            from_date = custom_config.get('from_date', FROM_DATE)
            to_date = custom_config.get('to_date', TO_DATE)
            topic_groups = custom_config.get('queries', TOPIC_GROUPS)
            max_articles = custom_config.get('max_articles', MAX_ARTICLES)
            min_topic_size = custom_config.get('min_topic_size', MIN_TOPIC_SIZE)
    else:
        # Use CLI arguments or fall back to config defaults
        from_date = args.from_date if args.from_date else FROM_DATE
        to_date = args.to_date if args.to_date else TO_DATE
        topic_groups = args.queries.split(',') if args.queries else TOPIC_GROUPS
        max_articles = args.max_articles if args.max_articles else MAX_ARTICLES
        min_topic_size = args.min_topic_size if args.min_topic_size else MIN_TOPIC_SIZE
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting pipeline with custom configuration:")
    print(f"   From: {from_date}")
    print(f"   To: {to_date}")
    print(f"   Queries: {len(topic_groups)} topics")
    print(f"   Max articles per query: {max_articles}")
    print(f"   Min topic size: {min_topic_size}")
    print(f"{'='*80}\n")
    
    companies_list = load_companies_from_csv('../data/companies.csv', sample_size=SAMPLE_SIZE,     
                           force_include_tickers=['IONQ', 'QUBT', 'RGTI', 'ARQQ', 'IBM', 'GOOGL', 
                           'MSFT', 'NVDA', 'INTC', 'AMD', 'QMCO'],)
    
    if not companies_list:
        print("⚠️ No companies loaded. Exiting...")
        exit()
    
    all_news_items = []

    for query in topic_groups:
        articles = collect_articles_from_newsapi(
            api_key=NEWSAPI_KEY,
            query=query,
            from_date=from_date,
            to_date=to_date,
            max_articles=max_articles,
            max_workers=MAX_WORKERS_ARTICLES
        )
        all_news_items.extend(articles)
        time.sleep(1)

    print(f"\n{'='*80}")
    print(f"✅ Total collected: {len(all_news_items)} articles")
    print(f"{'='*80}")

    if len(all_news_items) >= 10:
        model, topics, topic_companies = run_bertopic_with_multitimeframe_scoring(
            all_news_items, companies_list, depth=args.depth, user_id=args.user_id,
        )
    else:
        print("❌ Not enough articles to analyze. Need at least 10 articles.")
        print("   Try expanding date range or adding more topics.")
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'='*80}")
    print(f"⏱️ Script completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*80}")