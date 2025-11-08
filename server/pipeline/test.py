
import csv
from supabase import create_client, Client

from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# --- Configure Supabase ---
SUPABASE_URL = "https://uxrdywchpcwljsteomtn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV4cmR5d2NocGN3bGpzdGVvbXRuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjIxMjA1MzMsImV4cCI6MjA3NzY5NjUzM30.Ayt6lmN-ZRM7bH1GhNw7Cx1RcDw1uaGY0-oLqsY2jhs"


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

update_job_status("c0c7911b-8f7a-42ed-9d3a-12bfcf123af1", "completed", result_file_path="reports/e2de0813-3edf-4f26-a25e-5171a12ccac4_topic_companies_multitimeframe_depth-2_11-08-2025_02.csv")