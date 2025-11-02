"""
ticker_match.py  --  FAST semantic matcher
Speed-ups:
 1. Tiny NLI model (50 MB) instead of BART-large (440 MB)
 2. Batched NLI calls (CPU parallel)
 3. Skip NLI if cosine < 0.10
 4. Cache company embeddings
 5. Configurable thresholds
"""

import os
import pickle
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

from text_processing import (extract_entities, extract_keywords, clean_text,
                             count_explicit_mentions)
from scoring import (fetch_comprehensive_stock_data, calculate_day_trader_score,
                     calculate_swing_trader_score, calculate_position_trader_score,
                     calculate_longterm_investor_score)

# ------------------ CONFIG ------------------
EMB_CACHE_FILE      = "company_embs_384.pkl"
SIMILARITY_GATE     = 0.10          # skip NLI below this
ENTAILMENT_CUTOFF   = 0.08         # NLI acceptance
EMB_MODEL_NAME      = "all-MiniLM-L6-v2"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
# ------------------------------------------

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------- models ----------
semantic_model = SentenceTransformer(EMB_MODEL_NAME, device=device)
nli = pipeline("zero-shot-classification",
               model="cross-encoder/nli-deberta-v3-small",
               device=device)


# ---------- helpers ----------
def load_or_build_company_embeddings(companies_list):
    if os.path.exists(EMB_CACHE_FILE):
        return pickle.load(open(EMB_CACHE_FILE, "rb"))
    embs = semantic_model.encode([c["combined_text"][:400] for c in companies_list],
                                 normalize_embeddings=True, convert_to_numpy=True)
    pickle.dump(embs, open(EMB_CACHE_FILE, "wb"))
    return embs

def weighted_topic_vector(articles):
    """Fast centroid – no date window for small corpus."""
    texts = [clean_text(a["fulltext"]) for a in articles if a.get("fulltext")]
    if not texts:
        return np.zeros(384, dtype=np.float32)
    embs = semantic_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return (embs.mean(axis=0) / np.linalg.norm(embs.mean(axis=0))).astype(np.float32)

def entailment_filter_batch(company_texts, topic_keywords, cut_off=ENTAILMENT_CUTOFF):
    if len(topic_keywords) < 4:
        return [True] * len(company_texts)
    labels = [f"This company works with {kw}." for kw in topic_keywords[:6]]
    results = nli(company_texts, candidate_labels=labels, multi_label=True)
    return [any(s > cut_off for s in r["scores"]) for r in results]

# ---------- main matcher ----------
def match_companies_with_multitimeframe_scores(
        articles, topic_model, companies_list, top_n=50, similarity_threshold=0.12):
    """
    Same I/O as before – just faster.
    Returns  dict  topic_id -> list[company-dict]
    """

    # 1. prep articles
    for art in articles:
        if "processed" not in art:
            art["keywords"] = extract_keywords(art["fulltext"])
            art["entities"] = extract_entities(art["fulltext"])
            art["processed"] = True

    docs = [clean_text(a["fulltext"]) for a in articles]
    topics, _ = topic_model.transform(docs)
    topic_articles = {}
    for art, tid in zip(articles, topics):
        if tid == -1:
            continue
        topic_articles.setdefault(tid, []).append(art)

    # 2. company embeddings (cached)
    company_embs = load_or_build_company_embeddings(companies_list)

    topic_companies = {}
    for tid, arts in topic_articles.items():
        print(f"  Processing Topic {tid}...", end=" ")
        topic_vec = weighted_topic_vector(arts)
        topic_keywords = [w for w, _ in topic_model.get_topic(tid)]

        sims = np.dot(company_embs, topic_vec)  # cosine
        qualified = []

        # ---- batch NLI prep ----
        texts, idxs = [], []
        for idx, comp in enumerate(companies_list):
            if sims[idx] < similarity_threshold:
                continue
            texts.append(comp["combined_text"][:400])
            idxs.append(idx)

        if texts:  # run NLI once per topic
            flags = entailment_filter_batch(texts, topic_keywords)
            for idx, flag in zip(idxs, flags):
                if not flag:
                    continue
                comp = companies_list[idx]
                mention_count, variants = count_explicit_mentions(
                    " ".join(a["fulltext"] for a in arts).lower(), comp)
                bonus = min(0.15, 0.05 * mention_count)
                score = float(sims[idx]) + bonus

                qualified.append({
                    "ticker": comp["ticker"],
                    "name": comp["name"],
                    "relevance_score": score,
                    "mentions": mention_count,
                    "mentioned_as": variants if mention_count else None,
                    "stock_data": None,
                    "day_trader_score": None,
                    "swing_trader_score": None,
                    "position_trader_score": None,
                    "longterm_investor_score": None,
                })

        qualified.sort(key=lambda x: x["relevance_score"], reverse=True)
        topic_companies[tid] = qualified[:top_n]
        print(f"✓ {len(qualified[:top_n])} companies")

    # 3. stock-score threading (unchanged)
    all_tickers = {c["ticker"] for lst in topic_companies.values() for c in lst}
    stock_cache = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        fut = {ex.submit(fetch_comprehensive_stock_data, tk): tk for tk in all_tickers}
        for f in as_completed(fut):
            tk = fut[f]
            try:
                data = f.result()
                if data:
                    stock_cache[tk] = {
                        "data": data,
                        "day_trader": calculate_day_trader_score(data),
                        "swing_trader": calculate_swing_trader_score(data),
                        "position_trader": calculate_position_trader_score(data),
                        "longterm_investor": calculate_longterm_investor_score(data),
                    }
            except Exception:
                pass

    for tid, comps in topic_companies.items():
        for c in comps:
            if c["ticker"] in stock_cache:
                cache = stock_cache[c["ticker"]]
                c["stock_data"] = cache["data"]
                for tf in ("day_trader", "swing_trader", "position_trader", "longterm_investor"):
                    c[f"{tf}_score"] = cache[tf]["score"] if cache[tf] else None
                    c[f"{tf}_category"] = cache[tf]["category"] if cache[tf] else None

    return topic_companies



def is_relevant_company(company, topic_keywords, topic_text_lower):
    """Check if company is semantically relevant to topic"""
    
    # Extract key topic terms
    topic_domains = {
        'quantum': ['quantum', 'qubit', 'superconducting', 'photonic'],
        'semiconductor': ['semiconductor', 'chip', 'wafer', 'fabrication', 'foundry'],
        'crypto': ['cryptography', 'encryption', 'security', 'cyber']
    }
    
    # Identify topic domain
    active_domains = []
    for domain, terms in topic_domains.items():
        if any(term in topic_keywords for term in terms):
            active_domains.append(domain)
    
    if not active_domains:
        return True  # No specific domain, allow all
    
    # Check if company description matches domain
    company_text = company['combined_text'].lower()
    for domain in active_domains:
        domain_terms = topic_domains[domain]
        if any(term in company_text for term in domain_terms):
            return True
    
    # Check for explicit mentions in articles
    mention_count, _ = count_explicit_mentions(topic_text_lower, company)
    if mention_count > 0:
        return True
    
    return False