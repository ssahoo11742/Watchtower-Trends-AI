from fastapi import FastAPI, HTTPException
from pipeline.scoring import (
    fetch_comprehensive_stock_data,
    calculate_day_trader_score,
    calculate_swing_trader_score,
    calculate_position_trader_score,
    calculate_longterm_investor_score
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import paramiko
import os
import logging
import json
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_stock_response(data, topic_id=None, topic_keywords=None, company_name=None, 
                         relevance_score=None, mentions=0, mentioned_as=None,
                         day_score=None, swing_score=None, position_score=None, longterm_score=None):
    """
    Transform stock data into API response format
    """
    
    # Helper to safely convert numpy types and handle NaN/Infinity
    def safe_float(value, decimals=2):
        if value is None:
            return None
        try:
            # Handle numpy types
            if hasattr(value, 'item'):
                value = value.item()
            
            # Convert to float
            value = float(value)
            
            # Check for NaN or Infinity
            if math.isnan(value) or math.isinf(value):
                return None
            
            # Round to specified decimals
            return round(value, decimals)
        except (ValueError, TypeError):
            return None
    
    def safe_int(value):
        if value is None:
            return None
        try:
            if hasattr(value, 'item'):
                value = value.item()
            
            value = int(value)
            
            # Check for unreasonable values
            if abs(value) > 1e15:  # Arbitrary large number check
                return None
                
            return value
        except (ValueError, TypeError):
            return None
    
    # Build the response
    response = {
        "Topic_ID": topic_id,
        "Topic_Keywords": topic_keywords,
        "Ticker": data.get('ticker'),
        "Company_Name": company_name,
        "Relevance_Score": safe_float(relevance_score, 3) if relevance_score else None,
        "Mentions": safe_int(mentions),
        "Mentioned_As": mentioned_as,
        
        # Trading scores
        "Day_Trade_Score": safe_float(day_score['score'], 3) if day_score else None,
        "Day_Trade_Rating": day_score['category'] if day_score else None,
        "Swing_Trade_Score": safe_float(swing_score['score'], 3) if swing_score else None,
        "Swing_Trade_Rating": swing_score['category'] if swing_score else None,
        "Position_Trade_Score": safe_float(position_score['score'], 3) if position_score else None,
        "Position_Trade_Rating": position_score['category'] if position_score else None,
        "LongTerm_Score": safe_float(longterm_score['score'], 2) if longterm_score else None,
        "LongTerm_Rating": longterm_score['category'] if longterm_score else None,
        
        # Price data
        "Current_Price": safe_float(data.get('current_price'), 2),
        "Change_1D": safe_float(data.get('change_1d'), 2),
        "Change_1W": safe_float(data.get('change_1w'), 2),
        "Change_1M": safe_float(data.get('change_1m'), 2),
        "Change_3M": safe_float(data.get('change_3m'), 2),
        "Change_1Y": safe_float(data.get('change_1y'), 2),
        
        # Volume & Technical
        "Volume_Spike_Ratio": safe_float(data.get('volume_spike_ratio'), 2),
        "RSI_14": safe_float(data.get('rsi_14'), 1),
        "Price_vs_MA50": safe_float(data.get('price_vs_ma50'), 2),
        "Price_vs_MA200": safe_float(data.get('price_vs_ma200'), 2),
        
        # Fundamentals
        "PE_Ratio": safe_float(data.get('pe_ratio'), 2),
        "PEG_Ratio": safe_float(data.get('peg_ratio'), 2),
        "Dividend_Yield": safe_float(data.get('dividend_yield'), 2),
        "Profit_Margin": safe_float(data.get('profit_margin'), 4),
        "ROE": safe_float(data.get('roe'), 4)
    }
    
    return response



class JobData(BaseModel):
    job_id: str
    job_name: str
    from_date: str
    to_date: str
    query_type: str
    topics: List[str]
    custom_queries: List[str]
    queries: List[str]
    min_topic_size: int
    top_n_companies: int
    min_articles: int
    max_articles: int

def create_custom_config_content(job_data: JobData) -> str:
    """Generate custom config.py content based on job data"""
    
    # Format queries list for Python
    queries_formatted = ',\n    '.join([f'"{q}"' for q in job_data.queries])
    
    config_content = f'''import nltk
from nltk.corpus import stopwords
import spacy
from datetime import datetime

"""
Custom Configuration - Generated for job: {job_data.job_name}
Job ID: {job_data.job_id}
"""
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# API Configuration
NEWSAPI_KEY = '0c6458185614471e85f31fd67f473e69'
FROM_DATE = '{job_data.from_date}'
TO_DATE   = '{job_data.to_date}'

# HTTP Headers
HEADERS = {{
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}}

# Text Processing Settings
BOILERPLATE_TERMS = {{
    'email', 'digest', 'homepage', 'feed', 'newsletter', 'subscribe', 'subscription',
    'menu', 'navigation', 'sidebar', 'footer', 'header', 'cookie', 'privacy',
    'policy', 'terms', 'service', 'copyright', 'reserved', 'rights', 'contact',
    'facebook', 'twitter', 'instagram', 'linkedin', 'youtube', 'social', 'share',
    'comment', 'comments', 'reply', 'login', 'signup', 'register', 'search',
    'advertisement', 'sponsored', 'promo', 'promotion'
}}

STOPWORDS.update(BOILERPLATE_TERMS)

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 100000

GENERIC_NOUNS = {{
    "business", "company", "market", "economy", "government", 
    "state", "people", "industry"
}}

COMMON_WORD_BLACKLIST = set([
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'hi', 'if',
    'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to',
    'up', 'us', 'we',
    'all', 'and', 'are', 'but', 'can', 'car', 'cat', 'day', 'did', 'dog',
    'eat', 'far', 'few', 'for', 'fun', 'get', 'got', 'had', 'has', 'her',
    'him', 'his', 'how', 'its', 'let', 'man', 'may', 'new', 'not', 'now',
    'old', 'one', 'our', 'out', 'own', 'put', 'ran', 'red', 'run', 'said',
    'saw', 'say', 'see', 'set', 'she', 'sit', 'six', 'ten', 'the', 'too',
    'top', 'two', 'use', 'was', 'way', 'who', 'why', 'win', 'yes', 'yet',
    'you',
    'app', 'box', 'car', 'data', 'file', 'key', 'link', 'live', 'main',
    'net', 'open', 'post', 'real', 'site', 'tech', 'text', 'true', 'type',
    'uber', 'user', 'web', 'work', 'zoom'
])

# Analysis Settings (Custom from Job)
SAMPLE_SIZE = 10229
MIN_ARTICLE_LENGTH = 300
MAX_ARTICLE_LENGTH = 100000
MAX_TEXT_LENGTH = 50000
MIN_ARTICLES_FOR_ANALYSIS = {job_data.min_articles}
MAX_ARTICLES = {job_data.max_articles}

# Topic Modeling Settings (Custom from Job)
MIN_TOPIC_SIZE = {job_data.min_topic_size}
TOP_N_COMPANIES = {job_data.top_n_companies}
SIMILARITY_THRESHOLD = 0.08

# Multithreading
MAX_WORKERS_ARTICLES = 15
MAX_WORKERS_STOCKS = 10

# Search Queries (Custom from Job)
TOPIC_GROUPS = [
    {queries_formatted}
]
'''
    return config_content

@app.post("/custom_job")
async def custom_job(job_data: JobData):
    logger.info(f"Received job request: {job_data.job_id} - {job_data.job_name}")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ssh_key_path = os.path.join(script_dir, "ssh-key-watchtower.key")
    
    logger.info(f"Looking for SSH key at: {ssh_key_path}")
    
    # SSH Configuration
    host = os.getenv("SSH_HOST", "129.213.118.220")
    username = os.getenv("SSH_USERNAME", "ubuntu")
    
    logger.info(f"SSH Host configured as: {host}")
    
    ssh_success = False
    ssh_message = ""
    pipeline_output = ""
    pipeline_error = ""
    
    try:
        # Check if SSH key exists
        if not os.path.exists(ssh_key_path):
            logger.error(f"SSH key file not found: {ssh_key_path}")
            raise HTTPException(status_code=500, detail=f"SSH key file not found: {ssh_key_path}")
        
        # Create SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Load private key
        private_key = paramiko.RSAKey.from_private_key_file(ssh_key_path)
        
        # Connect to remote server
        logger.info(f"Attempting SSH connection to {username}@{host}")
        ssh.connect(
            hostname=host,
            username=username,
            pkey=private_key,
            timeout=10
        )
        
        ssh_success = True
        ssh_message = f"SSH connection successful to {username}@{host}"
        logger.info(ssh_message)
        
        # Create custom config as JSON
        custom_config_json = {
            "from_date": job_data.from_date,
            "to_date": job_data.to_date,
            "queries": job_data.queries,
            "max_articles": job_data.max_articles,
            "min_topic_size": job_data.min_topic_size
        }
        
        config_json_str = json.dumps(custom_config_json)
        custom_config_path = f"~/Watchtower-Trends-AI/server/pipeline/custom_config_{job_data.job_id}.json"
        
        # Write custom config JSON to remote server
        logger.info("Writing custom config JSON...")
        stdin, stdout, stderr = ssh.exec_command(f"cat > {custom_config_path}")
        stdin.write(config_json_str)
        stdin.channel.shutdown_write()
        stdout.channel.recv_exit_status()
        
        # Run the pipeline with custom config
        logger.info("Running pipeline.py with custom config...")
        pipeline_cmd = f"cd ~/Watchtower-Trends-AI/server/pipeline && python3 pipeline.py -d 1 --custom-config {custom_config_path} --user-id {user_id}"
        stdin, stdout, stderr = ssh.exec_command(pipeline_cmd, get_pty=True)
        
        pipeline_output = stdout.read().decode('utf-8')
        pipeline_error = stderr.read().decode('utf-8')
        exit_status = stdout.channel.recv_exit_status()
        
        logger.info(f"Pipeline exit status: {exit_status}")
        
        # Clean up custom config file
        logger.info("Cleaning up custom config...")
        stdin, stdout, stderr = ssh.exec_command(f"rm {custom_config_path}")
        stdout.channel.recv_exit_status()
        
        # Close connection
        ssh.close()
        
        response = {
            "status": "completed",
            "job_id": job_data.job_id,
            "job_name": job_data.job_name,
            "ssh_status": "success",
            "ssh_message": ssh_message,
            "pipeline_exit_status": exit_status,
            "pipeline_output": pipeline_output[-5000:] if len(pipeline_output) > 5000 else pipeline_output,  # Last 5000 chars
            "pipeline_error": pipeline_error[-2000:] if len(pipeline_error) > 2000 else pipeline_error,
            "received_data": job_data.dict()
        }
        
        logger.info(f"Job {job_data.job_id} completed successfully")
        return response
        
    except paramiko.AuthenticationException as e:
        ssh_message = f"SSH Authentication failed: {str(e)}"
        logger.error(ssh_message)
        raise HTTPException(status_code=500, detail=ssh_message)
    except paramiko.SSHException as e:
        ssh_message = f"SSH connection error: {str(e)}"
        logger.error(ssh_message)
        raise HTTPException(status_code=500, detail=ssh_message)
    except Exception as e:
        logger.error(f"Unexpected error processing job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing job: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/api/ticker/{symbol}")
def get_ticker(symbol: str):
    """Get complete stock analysis"""
    
    # Fetch data
    data = fetch_comprehensive_stock_data(symbol.upper())
    if not data:
        return {"error": f"Could not fetch data for {symbol}"}
    
    # Calculate scores
    day_score = calculate_day_trader_score(data)
    swing_score = calculate_swing_trader_score(data)
    position_score = calculate_position_trader_score(data)
    longterm_score = calculate_longterm_investor_score(data)
    
    # Format and return
    return format_stock_response(
        data=data,
        day_score=day_score,
        swing_score=swing_score,
        position_score=position_score,
        longterm_score=longterm_score
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # <-- Render injects this
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)