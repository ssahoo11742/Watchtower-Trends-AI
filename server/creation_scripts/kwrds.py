import pandas as pd
import spacy
import re
from collections import Counter

# ---------- spaCy ----------
nlp = spacy.load("en_core_web_sm")

# ---------- SETTINGS ----------
INPUT_FILE    = "../data/companies.csv"
OUTPUT_FILE   = "../data/companies.csv"  # overwrites original
MAX_KEYWORDS  = 12

# ---------- BLACKLISTS ----------
BLACKLIST_EXACT = {
    'united states', 'north america', 'south america', 'europe', 'asia pacific',
    'middle east', 'latin america', 'formerly known', 'headquartered', 'based in',
    'founded in', 'inc .', 'llc .', 'plc .', 'company inc', 'corporation inc',
    'including ads', 'include cloud', 'including enterprise', 'amazon web services',
    'azure quantum', 'amazon braket'
}

BLACKLIST_CONTAINS = [
    'beverage', 'formerly known', 'headquartered', 'based in', 'founded in',
    'inc .', 'llc .', 'plc .', 'middle east', 'latin america', 'segment provides',
    'segment focuses', 'segment engages', 'segment includes', 'segment develops',
    'amazon web services', 'amazon braket', 'azure quantum', 'beverage group holdings'
]

# ---------- STOP-WORDS ----------
STOP_WORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by could did do does doing down during each
few for from further had has have having he her here hers herself him himself
his how i if in into is it its itself let me more most my myself no nor not of
off on once only or other ought our ours ourselves out over own same she should
so some such than that the their theirs them themselves then there these they
this those through to too under until up very was we were what when where which
while who whom why with would you your yours yourself yourselves
""".split())

# ---------- CLEAN ----------
def clean_text(text, company_name):
    # remove company name (longer words only)
    for w in company_name.split():
        if len(w) > 4:
            text = re.sub(rf'\b{re.escape(w)}\b', '', text, flags=re.I)

    # boilerplate sentences
    patterns = [
        r'The company was formerly known as[^.]*\.',
        r'formerly known as[^.]*\.',
        r'headquartered in[^.]*\.',
        r'based in \w+, \w+\.',
        r'founded in \d{4}[^.]*\.'
    ]
    for p in patterns:
        text = re.sub(p, '', text, flags=re.I)
    return text

# ---------- EXTRACT ----------
def extract_keywords(text, company_name):
    text = clean_text(text, company_name)
    doc = nlp(text)

    phrases = []
    for chunk in doc.noun_chunks:
        tokens = [t.lemma_.lower() for t in chunk if t.lemma_.isalpha()]
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
        if not tokens or len(tokens) < 2:
            continue
        phrase = " ".join(tokens)

        # blacklist checks
        if phrase in BLACKLIST_EXACT:
            continue
        if any(bl in phrase for bl in BLACKLIST_CONTAINS):
            continue
        if any(t in company_name.lower() for t in tokens):
            continue
        if any(c in phrase for c in ".,;()"):
            continue
        if phrase.split()[0] in {"also", "provides", "offers", "includes", "company", "various", "such", "primarily", "facilitates"}:
            continue
        if phrase.split()[-1] in {"company", "inc", "corporation", "segment", "segments", "across", "related", "worldwide"}:
            continue
        phrases.append(phrase)

    # score by frequency + uniqueness (simple tf-idf proxy)
    counts = Counter(phrases)
    total = sum(counts.values()) or 1
    scored = {p: (cnt / total) * len(p.split()) for p, cnt in counts.items()}
    top = sorted(scored, key=scored.get, reverse=True)[:MAX_KEYWORDS]
    return top

# ---------- MAIN ----------
print(f"Loading {INPUT_FILE}...\n")
df = pd.read_csv(INPUT_FILE)

print(f"Processing {len(df)} companies...\n")
print("=" * 100)

for idx, row in df.iterrows():
    ticker = row['Ticker']
    name = row['Name'] if pd.notna(row['Name']) else ""
    desc = row['Description'] if pd.notna(row['Description']) else ""
    
    if not desc:
        print(f"[{idx+1}/{len(df)}] {ticker} - ⚠️  No description, skipping")
        continue
    
    try:
        keywords = extract_keywords(desc, name)
        df.at[idx, 'Keywords'] = ", ".join(keywords)
        
        print(f"[{idx+1}/{len(df)}] {ticker} - ✓ {len(keywords)} keywords")
        print(f"    → {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
        
    except Exception as e:
        print(f"[{idx+1}/{len(df)}] {ticker} - ✗ Error: {e}")

# ---------- SAVE ----------
print("\n" + "=" * 100)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Updated keywords saved to {OUTPUT_FILE}")
print(f"   Total records: {len(df)}")
print("=" * 100)