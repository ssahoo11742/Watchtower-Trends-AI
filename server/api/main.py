from fastapi import FastAPI, HTTPException
from api.scoring import (
    fetch_comprehensive_stock_data,
    calculate_day_trader_score,
    calculate_swing_trader_score,
    calculate_position_trader_score,
    calculate_longterm_investor_score
)
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import paramiko
import os
import logging
import json
from datetime import datetime
from supabase import create_client, Client
import uvicorn
import math
from dotenv import load_dotenv
load_dotenv()

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

# Initialize Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # Use service key for backend
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
    user_id: str
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
    depth: int

def update_job_status(job_id: str, status: str, error_message: str = None):
    """Update job status in Supabase - only for initial status"""
    try:
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if error_message:
            update_data["error_message"] = error_message
        
        supabase.table("custom_jobs").update(update_data).eq("id", job_id).execute()
        logger.info(f"Updated job {job_id} status to {status}")
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")

def process_job_background(job_data: JobData):
    """Process the job in the background - SSH and trigger remote script"""
    job_id = job_data.job_id
    user_id = job_data.user_id
    depth = job_data.depth
    
    logger.info(f"Starting background processing for job {job_id}")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ssh_key_path = os.path.join(script_dir, "ssh-key-watchtower.key")
    
    # SSH Configuration
    host = os.getenv("SSH_HOST", "129.213.118.220")
    username = os.getenv("SSH_USERNAME", "ubuntu")
    
    try:
        # Update status to running
        update_job_status(job_id, "running")
        
        # Check if SSH key exists
        if not os.path.exists(ssh_key_path):
            raise Exception(f"SSH key file not found: {ssh_key_path}")
        
        # Create SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Load private key
        private_key = paramiko.RSAKey.from_private_key_file(ssh_key_path)
        
        # Connect to remote server
        logger.info(f"Connecting to {username}@{host}")
        ssh.connect(
            hostname=host,
            username=username,
            pkey=private_key,
            timeout=10
        )
        
        # Create custom config as JSON
        custom_config_json = {
            "from_date": job_data.from_date,
            "to_date": job_data.to_date,
            "queries": job_data.queries,
            "max_articles": job_data.max_articles,
            "min_topic_size": job_data.min_topic_size
        }
        
        config_json_str = json.dumps(custom_config_json)
        custom_config_path = f"~/Watchtower-Trends-AI/server/pipeline/custom_config_{job_id}.json"
        
        # Write custom config JSON to remote server
        logger.info("Writing custom config JSON...")
        stdin, stdout, stderr = ssh.exec_command(f"cat > {custom_config_path}")
        stdin.write(config_json_str)
        stdin.channel.shutdown_write()
        stdout.channel.recv_exit_status()
        
        # Run the pipeline with custom config in background using nohup
        # This allows the SSH connection to close while the pipeline continues running
        logger.info("Starting pipeline.py in background on remote server...")
        pipeline_cmd = (
            f"nohup bash -c 'cd ~/Watchtower-Trends-AI/server/pipeline && "
            f"python3 pipeline.py -d {depth} --custom-config {custom_config_path} --user-id {user_id} --job-id {job_id}' "
            f"> ~/pipeline_{job_id}.log 2>&1 &"
        )
        
        ssh.exec_command(pipeline_cmd)
        
        # Give it a moment to start
        import time
        time.sleep(2)
        
        logger.info(f"Pipeline started in background for job {job_id}")
        
        # Close SSH connection - the remote process will continue
        ssh.close()
        
        logger.info(f"SSH connection closed. Job {job_id} is running on remote server.")
        
    except Exception as e:
        logger.error(f"Error initiating job {job_id}: {e}", exc_info=True)
        update_job_status(job_id, "failed", error_message=str(e))

@app.post("/custom_job")
async def custom_job(job_data: JobData, background_tasks: BackgroundTasks):
    logger.info(f"Received job request: {job_data.job_id} - {job_data.job_name}")
    
    # Add the job processing to background tasks
    background_tasks.add_task(process_job_background, job_data)
    
    # Return immediately
    return {
        "status": "accepted",
        "job_id": job_data.job_id,
        "message": "Job has been queued for processing"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a specific job"""
    try:
        result = supabase.table("custom_jobs").select("*").eq("id", job_id).single().execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {str(e)}")

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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
