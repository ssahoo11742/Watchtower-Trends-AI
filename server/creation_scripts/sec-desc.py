import requests
from bs4 import BeautifulSoup
import re
import time
import csv
import json
import pathlib

# Configuration
# IMPORTANT: Replace with your actual name and email!
HEADERS = {
    'User-Agent': 'Your Name your.email@example.com'
}
CIK_JSON = pathlib.Path("ticker_cik.json")
CSV_FILE = pathlib.Path("../data/companies.csv")

# ---------- Helper Functions ----------

def load_cik_map() -> dict:
    """Load ticker to CIK mapping from JSON file"""
    with CIK_JSON.open() as f:
        return {k.upper(): str(v).zfill(10) for k, v in json.load(f).items()}


def remove_consecutive_duplicates(paragraphs):
    """Remove consecutive duplicate paragraphs"""
    if not paragraphs:
        return paragraphs
    
    cleaned = [paragraphs[0]]
    for para in paragraphs[1:]:
        if para != cleaned[-1]:
            cleaned.append(para)
    return cleaned


def scrape_sec_business_section(url):
    """
    Extract Item 1. Business section from SEC filing URL.
    Returns cleaned business section text.
    """
    print(f"  Fetching: {url}")
    time.sleep(0.2)  # Be nice to SEC servers
    
    # Use simpler headers for HTML requests
    html_headers = {'User-Agent': HEADERS['User-Agent']}
    
    try:
        response = requests.get(url, headers=html_headers, timeout=20)
        response.raise_for_status()
    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove scripts and styles
    for element in soup(['script', 'style', 'head']):
        element.decompose()
    
    body = soup.find('body') or soup
    all_elements = body.find_all(['p', 'div', 'span', 'td', 'font'])
    
    business_start_idx = None
    business_end_idx = None
    
    # Find the start - look for Item 1 followed by substantial content
    for i, elem in enumerate(all_elements):
        text = elem.get_text(strip=True)
        
        if re.search(r'Item\s+1\.?\s*Business', text, re.IGNORECASE):
            # Look ahead for real content
            content_length = 0
            for j in range(i, min(i+20, len(all_elements))):
                content_length += len(all_elements[j].get_text(strip=True))
            
            if content_length > 500:
                business_start_idx = i
                break
    
    if business_start_idx is None:
        print("  ✗ Could not find Item 1. Business section")
        return None
    
    # Find the end - look for Item 1A or Item 2
    for i in range(business_start_idx + 1, len(all_elements)):
        text = all_elements[i].get_text(strip=True)
        
        if re.match(r'Item\s+1A\.?\s*(Risk|Unresolved)', text, re.IGNORECASE):
            business_end_idx = i
            break
        elif re.match(r'Item\s+2\.?\s*Propert', text, re.IGNORECASE):
            business_end_idx = i
            break
    
    if business_end_idx is None:
        business_end_idx = min(business_start_idx + 1000, len(all_elements))
    
    # Extract content
    business_elements = all_elements[business_start_idx:business_end_idx]
    
    paragraphs = []
    seen_texts = set()
    
    for elem in business_elements:
        text = elem.get_text(strip=True)
        
        # Skip short fragments, page numbers, and duplicates
        if len(text) > 10 and not re.match(r'^\d+$', text):
            if text not in seen_texts:
                paragraphs.append(text)
                seen_texts.add(text)
    
    paragraphs = remove_consecutive_duplicates(paragraphs)
    business_text = '\n\n'.join(paragraphs)
    
    # Final cleanup
    business_text = re.sub(r'\n{3,}', '\n\n', business_text)
    business_text = re.sub(r'[ \t]+', ' ', business_text)
    
    # Remove ALL newlines before returning
    business_text = business_text.replace('\n', ' ')
    business_text = re.sub(r'\s+', ' ', business_text).strip()
    
    print(f"  ✓ Extracted {len(business_text)} characters ({len(paragraphs)} paragraphs)")
    return business_text


def get_10k_business(ticker: str, cik: str, max_chars: int = 500000) -> str:
    """
    Fetch the most recent 10-K filing and extract business section.
    Returns the business description text (truncated to max_chars).
    """
    try:
        # Get filing list
        sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        sub = requests.get(sub_url, headers=HEADERS, timeout=20).json()
        
        forms = sub["filings"]["recent"]["form"]
        accnums = sub["filings"]["recent"]["accessionNumber"]
        docs = sub["filings"]["recent"]["primaryDocument"]
        
        # Find first 10-K
        for form, acc, doc in zip(forms, accnums, docs):
            if form == "10-K":
                acc_stripped = acc.replace("-", "")
                html_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_stripped}/{doc}"
                
                # Use the new scraping method
                business_text = scrape_sec_business_section(html_url)
                
                if business_text:
                    return business_text[:max_chars]
                break  # Only try first 10-K
                
    except Exception as e:
        print(f"  ⚠️  {ticker}: {e}")
    
    return ""


def main():
    """Main function to update companies.csv with 10-K business descriptions"""
    
    # Load CIK mapping
    cik_map = load_cik_map()
    print(f"Loaded CIK map with {len(cik_map)} tickers")
    
    # Load CSV
    rows = []
    with CSV_FILE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    print(f"Loaded {len(rows)} rows from {CSV_FILE.name}\n")
    
    # Process each row
    count = 0
    for row in rows:
        count += 1
        
        # REMOVE THIS LIMIT WHEN READY TO PROCESS ALL ROWS
        # if count > 5:
        #     print(f"\n⚠️  Stopping after 5 rows (remove limit in code to process all)")
        #     break
        
        ticker = row["Ticker"].upper()
        
        # Skip if ticker not in CIK map
        if ticker not in cik_map:
            print(f"[{count}] {ticker}: Not in CIK map, skipping")
            continue
        
        cik = cik_map[ticker]
        print(f"\n[{count}] Processing {ticker} (CIK: {cik})...")
        
        # Fetch business description
        business = get_10k_business(ticker, cik)
        
        if business:
            # APPEND to existing description
            existing = row.get("Description", "")
            row["Description"] = existing + " " + business
            print(f"  ✓ Updated description (total length: {len(row['Description'])} chars)")
        else:
            print(f"  ✗ No business section found")
        
        time.sleep(0.3)  # Be nice to SEC
    
    # Write back to CSV
    print(f"\n{'='*60}")
    print("Writing updated data back to CSV...")
    with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Done! Updated {CSV_FILE.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()