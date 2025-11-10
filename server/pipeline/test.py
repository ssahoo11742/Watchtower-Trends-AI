from supabase import create_client, Client
SUPABASE_URL = "https://uxrdywchpcwljsteomtn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV4cmR5d2NocGN3bGpzdGVvbXRuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjIxMjA1MzMsImV4cCI6MjA3NzY5NjUzM30.Ayt6lmN-ZRM7bH1GhNw7Cx1RcDw1uaGY0-oLqsY2jhs"


# Only create client if we have a service key
if SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
try:
    with open("topic_companies_multitimeframe_depth-1_11-10-2025_16.csv", "rb") as f:
        upload_response = supabase.storage.from_("daily-reports").upload(
            path="reports/topic_companies_multitimeframe_depth-1_11-10-2025_16.csv",
            file=f,
            file_options={"content-type": "text/csv", "upsert": "true"}
        )
    print(f"✅ Upload successful: {upload_response}")
except Exception as e:
    print(f"❌ Upload failed: {e}")
    raise
