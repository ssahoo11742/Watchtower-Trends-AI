from supabase import create_client, Client

SUPABASE_URL = "https://uxrdywchpcwljsteomtn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV4cmR5d2NocGN3bGpzdGVvbXRuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjIxMjA1MzMsImV4cCI6MjA3NzY5NjUzM30.Ayt6lmN-ZRM7bH1GhNw7Cx1RcDw1uaGY0-oLqsY2jhs"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

filename = "topic_companies_multitimeframe.csv"

# ✅ Put file inside a folder in the bucket (recommended)
storage_path = f"reports/{filename}"

try:
    with open(filename, "rb") as f:
        upload_response = supabase.storage.from_("daily-reports").upload(
            storage_path,
            f,
            {
                "contentType": "text/csv",
                "upsert": "true"   # must be *string*
            }
        )
    print("✅ Successfully uploaded!", upload_response)
except Exception as e:
    print("❌ Upload failed:", e)
