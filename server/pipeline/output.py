import csv
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from ticker_match import match_companies_with_multitimeframe_scores
from text_processing import extract_entities, extract_keywords, clean_text
from config import TOP_N_COMPANIES
from topic_cleanse import filter_bertopic_keywords
import csv
from supabase import create_client, Client
import gc
from datetime import datetime
import torch
from ticker_match import set_depth_mode
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# Only create client if we have a service key
if SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    print("⚠️ Warning: SUPABASE_SERVICE_KEY not found in environment. Job status updates will be skipped.")

def update_job_status(job_id: str, status: str, error_message: str = None, result_file_path: str = None):
    """Update job status in Supabase"""
    if not job_id:
        print("⚠️ No job_id provided, skipping status update")
        return
    
    if not supabase:
        print("⚠️ Supabase client not initialized, skipping status update")
        return
        
    try:
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if status == "completed":
            update_data["completed_at"] = datetime.utcnow().isoformat()
            if result_file_path:
                update_data["result_file_path"] = result_file_path
        
        if error_message:
            update_data["error_message"] = error_message
        
        supabase.table("custom_jobs").update(update_data).eq("id", job_id).execute()
        print(f"✅ Updated job {job_id} status to {status}")
    except Exception as e:
        print(f"❌ Failed to update job status: {e}")

# ============================================================================
# DISPLAY RESULTS BY TRADING STYLE
# ============================================================================

def display_results_by_trading_style(topic_companies, topic_model, trading_style='swing'):
    """Display companies ranked by relevance score"""
    
    style_map = {
        'day': ('day_trader_score', 'Day Trading (1d-1w)', 'day_trader_category'),
        'swing': ('swing_trader_score', 'Swing Trading (1w-3m)', 'swing_trader_category'),
        'position': ('position_trader_score', 'Position Trading (3m-1y)', 'position_trader_category'),
        'longterm': ('longterm_investor_score', 'Long-term Investing (1y+)', 'longterm_investor_category')
    }
    
    if trading_style not in style_map:
        print(f"❌ Invalid trading style. Choose: {list(style_map.keys())}")
        return
    
    score_field, style_name, category_field = style_map[trading_style]
    
    print(f"\n{'='*120}")
    print(f"=== COMPANIES RANKED BY RELEVANCE - {style_name.upper()} ===")
    print(f"{'='*120}")
    
    for topic_id in sorted(topic_companies.keys()):
        keywords = ", ".join([word for word, _ in topic_model.get_topic(topic_id)])
        print(f"\n📊 Topic {topic_id}: {keywords}")
        
        companies = topic_companies[topic_id]
        
        # Separate companies with and without stock data
        companies_with_scores = [c for c in companies if c.get(score_field) is not None]
        companies_without_scores = [c for c in companies if c.get(score_field) is None]
        
        # Sort by relevance score (primary) and trading style score (secondary)
        companies_with_scores.sort(key=lambda x: (x['relevance_score'], x[score_field]), reverse=True)
        
        if companies_with_scores:
            print(f"{'Rank':<5} {'Ticker':<8} {'Relevance':<10} {'Score':<8} {'Name':<30} {'Rating':<25} {'Mentions'}")
            print("-" * 120)
            
            for rank, comp in enumerate(companies_with_scores[:20], 1):
                relevance_str = f"{comp['relevance_score']:.3f}"
                score_str = f"{comp[score_field]:.3f}"
                category = comp.get(category_field, "N/A")
                
                mention_info = ""
                if comp['mentions'] > 0:
                    mention_info = f"🎯 {comp['mentions']}"
                
                print(f"{rank:<5} {comp['ticker']:<8} {relevance_str:<10} {score_str:<8} {comp['name'][:30]:<30} {category:<25} {mention_info}")
            
            if companies_without_scores:
                print(f"\n  ⚠️ {len(companies_without_scores)} companies without stock data")
        else:
            print("  No companies with stock data found")

# ============================================================================
# EXPORT TO CSV WITH ALL TIMEFRAMES AND SUPABASE UPLOAD
# ============================================================================

def export_multitimeframe_results(topic_companies, topic_model, depth, filename='topic_companies_multitimeframe.csv', user_id=None, job_id=None):
    """Export results with all timeframe scores, sorted by relevance, and upload to Supabase"""

    timestamp = datetime.now().strftime("%m-%d-%Y_%H")
    
    if user_id:
        filename = f"{user_id}_topic_companies_multitimeframe_depth-{depth}_{timestamp}.csv"
    else:
        filename = f"topic_companies_multitimeframe_depth-{depth}_{timestamp}.csv"
    
    print(f"\n📊 Exporting results to {filename}...")
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Topic_ID', 'Topic_Keywords', 'Ticker', 'Company_Name',
                'Relevance_Score', 'Mentions', 'Mentioned_As',
                'Day_Trade_Score', 'Day_Trade_Rating',
                'Swing_Trade_Score', 'Swing_Trade_Rating',
                'Position_Trade_Score', 'Position_Trade_Rating',
                'LongTerm_Score', 'LongTerm_Rating',
                'Current_Price', 'Change_1D', 'Change_1W', 'Change_1M', 'Change_3M', 'Change_1Y',
                'Volume_Spike_Ratio', 'RSI_14', 'Price_vs_MA50', 'Price_vs_MA200',
                'PE_Ratio', 'PEG_Ratio', 'Dividend_Yield', 'Profit_Margin', 'ROE'
            ])
            
            for topic_id in sorted(topic_companies.keys()):
                keywords = ", ".join([word for word, _ in topic_model.get_topic(topic_id)])
                
                # Sort companies by relevance score before writing
                sorted_companies = sorted(topic_companies[topic_id], 
                                         key=lambda x: x['relevance_score'], 
                                         reverse=True)
                
                for comp in sorted_companies:
                    mentioned_as = '; '.join(comp['mentioned_as']) if comp['mentioned_as'] else ''
                    
                    # Stock scores
                    day_score = comp.get('day_trader_score', '')
                    day_rating = comp.get('day_trader_category', '')
                    swing_score = comp.get('swing_trader_score', '')
                    swing_rating = comp.get('swing_trader_category', '')
                    position_score = comp.get('position_trader_score', '')
                    position_rating = comp.get('position_trader_category', '')
                    longterm_score = comp.get('longterm_investor_score', '')
                    longterm_rating = comp.get('longterm_investor_category', '')
                    
                    # Extract data if available
                    price = change_1d = change_1w = change_1m = change_3m = change_1y = ''
                    volume_spike = rsi = ma50_diff = ma200_diff = ''
                    pe = peg = div_yield = profit_margin = roe = ''
                    
                    if comp.get('stock_data'):
                        data = comp['stock_data']
                        price = f"{data['current_price']:.2f}"
                        change_1d = f"{data['change_1d']:.2f}"
                        change_1w = f"{data['change_1w']:.2f}"
                        change_1m = f"{data['change_1m']:.2f}"
                        change_3m = f"{data['change_3m']:.2f}"
                        change_1y = f"{data['change_1y']:.2f}"
                        volume_spike = f"{data['volume_spike_ratio']:.2f}"
                        rsi = f"{data['rsi_14']:.1f}"
                        ma50_diff = f"{data['price_vs_ma50']:.2f}"
                        ma200_diff = f"{data['price_vs_ma200']:.2f}"
                        
                        if data['pe_ratio']:
                            pe = f"{data['pe_ratio']:.2f}"
                        if data['peg_ratio']:
                            peg = f"{data['peg_ratio']:.2f}"
                        if data['dividend_yield']:
                            div_yield = f"{data['dividend_yield']:.2f}"
                        if data['profit_margin']:
                            profit_margin = f"{data['profit_margin']*100:.2f}"
                        if data['roe']:
                            roe = f"{data['roe']*100:.2f}"
                    
                    writer.writerow([
                        topic_id, keywords, comp['ticker'], comp['name'],
                        f"{comp['relevance_score']:.3f}", comp['mentions'], mentioned_as,
                        day_score, day_rating,
                        swing_score, swing_rating,
                        position_score, position_rating,
                        longterm_score, longterm_rating,
                        price, change_1d, change_1w, change_1m, change_3m, change_1y,
                        volume_spike, rsi, ma50_diff, ma200_diff,
                        pe, peg, div_yield, profit_margin, roe
                    ])
        
        print(f"✅ Exported to {filename}")
        
        # Upload to Supabase Storage (only if supabase client is available)
        if supabase:
            print("☁️ Uploading to Supabase Storage (bucket: daily-reports)...")
            storage_path = f"reports/{filename}"
            
            try:
                with open(filename, "rb") as f:
                    upload_response = supabase.storage.from_("daily-reports").upload(
                        storage_path,
                        f,
                        {
                            "contentType": "text/csv",
                            "upsert": "true"
                        }
                    )
                print(f"✅ Successfully uploaded to Supabase Storage at {storage_path}")
                
                os.remove(filename)
                print(f"🗑️ Deleted local file: {filename}")
                
                # Update job status to completed with the file path
                if job_id:
                    update_job_status(job_id, "completed", result_file_path=storage_path)
                
            except Exception as upload_error:
                print(f"❌ Upload failed: {upload_error}")
                if job_id:
                    update_job_status(job_id, "failed", error_message=f"Upload failed: {str(upload_error)}")
                raise
        else:
            print("⚠️ Skipping Supabase upload (no client initialized)")
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
        if job_id:
            update_job_status(job_id, "failed", error_message=f"Export failed: {str(e)}")
        raise

# ============================================================================
# MAIN BERTOPIC ANALYSIS WITH MULTI-TIMEFRAME SCORING
# ============================================================================

def run_bertopic_with_multitimeframe_scoring(articles, companies_list, depth, user_id=None, job_id=None):
    """Run BERTopic with multi-timeframe stock scoring"""
    
    try:
        # Trim long articles upfront
        for article in articles:
            if article.get('fulltext') and len(article['fulltext']) > 50000:
                article['fulltext'] = article['fulltext'][:50000]
        
        print(f"\n🔄 Pre-processing {len(articles)} articles...")
        for article in articles:
            article['keywords'] = extract_keywords(article["fulltext"])
            article['entities'] = extract_entities(article["fulltext"])
            article['cleaned'] = clean_text(
                article["fulltext"] + " " + (article['keywords']*3) + " " + (article['entities']*2)
            )
            article['processed'] = True
        
        docs = [a['cleaned'] for a in articles]
        
        # Clean up memory before BERTopic
        print("🧹 Cleaning up memory before topic modeling...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
            n_gram_range=(1,3),
            top_n_words=8,
            nr_topics=nr_topics,
            calculate_probabilities=False,
            low_memory=True
        )

        topics, _ = topic_model.fit_transform(docs)

        # Filter keywords using word embeddings
        print(f"\n{'='*80}")
        print("🔍 FILTERING TOPIC KEYWORDS WITH SEMANTIC ANALYSIS")
        print(f"{'='*80}\n")
        
        filtered_topics, topic_model = filter_bertopic_keywords(
            topic_model=topic_model,
            model_path='GoogleNews-vectors-negative300.bin',
            min_keep=3,
            max_keep=5,
            verbose=True
        )

        # Display filtered topics
        info = topic_model.get_topic_info()
        print("\n" + "="*80)
        print("=== FILTERED TOPICS ===")
        print("="*80)
        for _, row in info.head(15).iterrows():
            if row['Topic'] != -1:
                keywords = ", ".join([word for word, _ in topic_model.get_topic(row['Topic'])])
                print(f"Topic {row['Topic']}: {keywords}")

        # Match companies with multi-timeframe scores
        topic_companies = match_companies_with_multitimeframe_scores(
            articles, topic_model, companies_list, top_n=TOP_N_COMPANIES, similarity_threshold=0.08
        )
        
        # Display results for each trading style
        trading_styles = ['day', 'swing', 'position', 'longterm']
        
        for style in trading_styles:
            display_results_by_trading_style(topic_companies, topic_model, style)
        
        # Export to CSV and upload to Supabase
        export_multitimeframe_results(topic_companies, topic_model, depth=depth, user_id=user_id, job_id=job_id)
        
        # Show top performers
        print(f"\n{'='*120}")
        print("=== TOP 10 STOCKS BY RELEVANCE (Across All Topics) ===")
        print(f"{'='*120}")
        
        for style in trading_styles:
            score_field, style_name, category_field = {
                'day': ('day_trader_score', 'Day Trading', 'day_trader_category'),
                'swing': ('swing_trader_score', 'Swing Trading', 'swing_trader_category'),
                'position': ('position_trader_score', 'Position Trading', 'position_trader_category'),
                'longterm': ('longterm_investor_score', 'Long-term Investing', 'longterm_investor_category')
            }[style]
            
            all_companies = []
            for topic_id, companies in topic_companies.items():
                for comp in companies:
                    if comp.get(score_field) is not None:
                        all_companies.append({
                            'topic_id': topic_id,
                            'ticker': comp['ticker'],
                            'name': comp['name'],
                            'score': comp[score_field],
                            'relevance': comp['relevance_score'],
                            'category': comp.get(category_field, "")
                        })
            
            all_companies.sort(key=lambda x: (x['relevance'], x['score']), reverse=True)
            
            print(f"\n🎯 {style_name.upper()}:")
            print(f"{'Rank':<5} {'Ticker':<8} {'Relevance':<10} {'Score':<8} {'Topic':<8} {'Company':<30} {'Rating'}")
            print("-" * 120)
            for rank, comp in enumerate(all_companies[:10], 1):
                print(f"{rank:<5} {comp['ticker']:<8} {comp['relevance']:.3f}      {comp['score']:.3f}    {comp['topic_id']:<8} {comp['name'][:30]:<30} {comp['category']}")
        
        return topic_model, topics, topic_companies
        
    except Exception as e:
        print(f"❌ Error in run_bertopic_with_multitimeframe_scoring: {e}")
        if job_id:
            update_job_status(job_id, "failed", error_message=str(e))
        raise