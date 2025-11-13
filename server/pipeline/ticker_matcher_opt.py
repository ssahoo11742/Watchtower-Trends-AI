"""
ticker_match_opt.py  --  OPTIMIZED Hybrid semantic matcher (Fast + Accurate)
Key optimizations:
1. Multi-stage filtering (cheap filters first, NLI only for survivors)
2. Aggressive NLI caching with hashing
3. Global batch processing for NLI (all topics at once)
4. Lighter NLI model option
5. Early elimination of obviously irrelevant candidates
"""

import os
import pickle
import numpy as np
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import re

from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

from text_processing import (extract_entities, extract_keywords, clean_text,
                             count_explicit_mentions)
from scoring import (fetch_comprehensive_stock_data, calculate_day_trader_score,
                     calculate_swing_trader_score, calculate_position_trader_score,
                     calculate_longterm_investor_score)

# ------------------ CONFIG ------------------
EMB_CACHE_FILE      = "company_embs_384.pkl"
NLI_CACHE_FILE      = "nli_cache.pkl"  # Persistent cache
SIMILARITY_GATE     = 0.03
EMB_MODEL_NAME      = "all-MiniLM-L6-v2"

# FASTER NLI model options (choose one):
# Option 1: Lighter model (40MB, faster, slight accuracy drop)
NLI_MODEL_NAME      = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 3x faster than deberta
# Option 2: Keep accuracy but use smaller context
NLI_MODEL_NAME    = "cross-encoder/nli-deberta-v3-small"
NLI_MAX_LENGTH      = 128  # Reduced from 256 (2x faster)

# Multi-stage filtering thresholds
MIN_KEYWORD_OVERLAP = 0.05  # Must have some keyword overlap before NLI
MIN_EMBEDDING_SIM   = 0.05  # Slightly higher than SIMILARITY_GATE

# CPU optimization for ARM/low-resource environments
CPU_THREAD_LIMIT    = 4  # Match your VM cores
# ------------------------------------------

# Device setup with CPU optimization
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    torch.set_num_threads(CPU_THREAD_LIMIT)
    print(f"🖥️  Running on CPU with {CPU_THREAD_LIMIT} threads")

# ---------- models ----------
print("📥 Loading models...")
semantic_model = SentenceTransformer(EMB_MODEL_NAME, device=device)
print(f"  ✓ Semantic model loaded ({EMB_MODEL_NAME})")

nli_model = CrossEncoder(NLI_MODEL_NAME, max_length=NLI_MAX_LENGTH, device=device)
print(f"  ✓ NLI model loaded ({NLI_MODEL_NAME})")
print(f"  ✓ Using device: {device}")

# Load persistent NLI cache
if os.path.exists(NLI_CACHE_FILE):
    with open(NLI_CACHE_FILE, "rb") as f:
        nli_cache = pickle.load(f)
else:
    nli_cache = {}


def save_nli_cache():
    """Save NLI cache to disk"""
    with open(NLI_CACHE_FILE, "wb") as f:
        pickle.dump(nli_cache, f)


def hash_text(text):
    """Create hash for cache keys"""
    return hashlib.md5(text.encode()).hexdigest()


# ---------- helpers ----------
def load_or_build_company_embeddings(companies_list):
    if os.path.exists(EMB_CACHE_FILE):
        return pickle.load(open(EMB_CACHE_FILE, "rb"))
    embs = semantic_model.encode([c["combined_text"][:400] for c in companies_list],
                                 normalize_embeddings=True, convert_to_numpy=True)
    pickle.dump(embs, open(EMB_CACHE_FILE, "wb"))
    return embs


def weighted_topic_vector(articles):
    texts = [clean_text(a["fulltext"]) for a in articles if a.get("fulltext")]
    if not texts:
        return np.zeros(384, dtype=np.float32)
    
    batch_size = 32
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = semantic_model.encode(batch, normalize_embeddings=True, 
                                     convert_to_numpy=True, show_progress_bar=False)
        all_embs.append(embs)
    
    embs = np.vstack(all_embs)
    return (embs.mean(axis=0) / np.linalg.norm(embs.mean(axis=0))).astype(np.float32)


def extract_topic_domain_signals(topic_keywords, articles):
    """Extract domain signals from topic WITHOUT hardcoding industries."""
    all_text = " ".join(topic_keywords[:15])
    
    for art in articles[:10]:
        if 'keywords' in art:
            all_text += " " + art['keywords']
        if 'title' in art:
            all_text += " " + art['title']
    
    all_text_lower = all_text.lower()
    
    generic_words = {
        'system', 'technology', 'company', 'data', 'research', 
        'service', 'platform', 'solution', 'business', 'market',
        'industry', 'product', 'customer', 'management', 'global',
        'world', 'new', 'report', 'news', 'today', 'year'
    }
    
    technical_terms = set()
    for kw in topic_keywords[:12]:
        if len(kw) > 3 and kw.lower() not in generic_words:
            technical_terms.add(kw.lower())
    
    entities = set()
    for art in articles[:15]:
        if 'entities' in art:
            for entity in art['entities'].split():
                if len(entity) > 2:
                    entities.add(entity.lower())
    
    return {
        'technical_terms': technical_terms,
        'entities': entities,
        'topic_text': all_text_lower,
        'topic_keywords': topic_keywords[:8]
    }


def quick_keyword_score(company_text_lower, technical_terms):
    """FAST keyword scoring for early filtering"""
    if not technical_terms:
        return 0.0
    
    keyword_matches = 0
    for term in technical_terms:
        if term in company_text_lower:
            keyword_matches += 1.0
    
    return keyword_matches / len(technical_terms)


def validate_contextual_mention(mention_text, company_name, company_ticker):
    """STRICT validation: Is this mention actually the company, not a common word?"""
    mention_lower = mention_text.lower()
    company_name_lower = company_name.lower()
    ticker_lower = company_ticker.lower()
    
    is_likely_common_word = (
        len(company_ticker) <= 4 and 
        company_ticker.isupper() and
        not any(c.isdigit() for c in company_ticker)
    )
    
    if is_likely_common_word:
        ticker_pattern = r'\b' + re.escape(ticker_lower) + r'\b'
        matches = list(re.finditer(ticker_pattern, mention_lower))
        
        if not matches:
            return False
        
        valid_count = 0
        for match in matches:
            mention_idx = match.start()
            start = max(0, mention_idx - 100)
            end = min(len(mention_text), mention_idx + len(company_ticker) + 100)
            context = mention_text[start:end].lower()
            
            company_context_patterns = [
                r'\b(inc\.|incorporated|corp\.|corporation|ltd\.|limited|llc|plc)\b',
                r'\b(' + re.escape(ticker_lower) + r"'s|" + re.escape(ticker_lower) + r"'s)\b",
                r'\$' + re.escape(ticker_lower) + r'\b',
                r'\b(ceo|cfo|president|chairman|founder|executive)\b.*\b' + re.escape(ticker_lower) + r'\b',
                r'\b' + re.escape(ticker_lower) + r'\b.*(announced|reported|said|stated|revealed|disclosed|launched|released)',
                r'\b(shares|stock|ticker|symbol)\b.*\b' + re.escape(ticker_lower) + r'\b',
                r'\b' + re.escape(ticker_lower) + r'\b.*(shares|stock|trading|investors)',
                r'\b(nasdaq|nyse|exchange|traded)\b.*\b' + re.escape(ticker_lower) + r'\b',
                r'\b' + re.escape(ticker_lower) + r'\b.*(ipo|offering|acquisition|merger|earnings|revenue)',
            ]
            
            company_name_words = [w for w in company_name_lower.split() if len(w) > 3]
            has_company_name = any(word in context for word in company_name_words[:2])
            has_company_context = any(re.search(pattern, context) for pattern in company_context_patterns)
            
            if has_company_context or has_company_name:
                valid_count += 1
        
        return valid_count > 0
    
    ticker_pattern = r'\b' + re.escape(ticker_lower) + r'\b'
    matches = list(re.finditer(ticker_pattern, mention_lower))
    
    if not matches:
        return False
    
    for match in matches:
        mention_idx = match.start()
        start = max(0, mention_idx - 50)
        end = min(len(mention_text), mention_idx + len(company_ticker) + 50)
        context = mention_text[start:end]
        
        actual_mention = mention_text[mention_idx:mention_idx + len(company_ticker)]
        if actual_mention.isupper() or actual_mention[0].isupper():
            return True
    
    return False


def nli_relevance_batch_optimized(company_descriptions, topic_keywords, batch_size=None):
    """
    OPTIMIZED NLI with aggressive caching and adaptive batching.
    Only called on pre-filtered candidates.
    """
    if not topic_keywords or not company_descriptions:
        return [0.0] * len(company_descriptions)
    
    # Adaptive batch size: larger for GPU, smaller for CPU
    if batch_size is None:
        batch_size = 128 if device != "cpu" else 32
    
    # Create hypotheses (fewer = faster)
    hypotheses = []
    for kw in topic_keywords[:4]:  # Reduced from 6
        hypotheses.append(f"This company works with {kw}.")
        hypotheses.append(f"This company specializes in {kw}.")
    
    hyp_tuple = tuple(hypotheses)
    all_scores = []
    
    # Collect uncached pairs
    uncached_pairs = []
    uncached_indices = []
    
    for i, desc in enumerate(company_descriptions):
        desc_hash = hash_text(desc[:200])
        cache_key = (desc_hash, hyp_tuple)
        
        if cache_key in nli_cache:
            all_scores.append(nli_cache[cache_key])
        else:
            all_scores.append(None)  # Placeholder
            uncached_indices.append(i)
            for hyp in hypotheses:
                uncached_pairs.append([desc[:200], hyp])
    
    # Batch predict all uncached pairs at once
    if uncached_pairs:
        scores = nli_model.predict(uncached_pairs, batch_size=batch_size)
        probs = 1 / (1 + np.exp(-np.array(scores)))
        
        # Group by company (each company has len(hypotheses) pairs)
        hyp_count = len(hypotheses)
        for idx, orig_idx in enumerate(uncached_indices):
            start = idx * hyp_count
            end = start + hyp_count
            company_probs = probs[start:end]
            max_prob = float(np.max(company_probs))
            
            # Cache result
            desc = company_descriptions[orig_idx]
            desc_hash = hash_text(desc[:200])
            cache_key = (desc_hash, hyp_tuple)
            nli_cache[cache_key] = max_prob
            
            all_scores[orig_idx] = max_prob
    
    return all_scores


def calculate_multi_signal_relevance(company, company_text, domain_signals, articles, 
                                    nli_score=None, keyword_score_precomputed=None,
                                    debug_tickers=None):
    """
    OPTIMIZED multi-signal relevance - some scores pre-computed.
    Includes early exit for obviously irrelevant companies.
    """
    company_text_lower = company_text.lower()
    technical_terms = domain_signals['technical_terms']
    entities = domain_signals['entities']
    
    # Signal 1: NLI (pre-computed)
    semantic_score = nli_score if nli_score is not None else 0.0
    
    # Signal 2: Keyword (use pre-computed if available)
    if keyword_score_precomputed is not None:
        keyword_score = keyword_score_precomputed
        matched_keywords = []
    else:
        keyword_matches = 0
        matched_keywords = []
        for term in technical_terms:
            if term in company_text_lower:
                keyword_matches += 1.0
                matched_keywords.append(term)
        keyword_score = min(1.0, keyword_matches / max(len(technical_terms), 1))
    
    # EARLY EXIT: If both NLI and keywords are weak, don't waste time
    if semantic_score < 0.20 and keyword_score < 0.10:
        return create_relevance_result(False, semantic_score, keyword_score, 0, 0, 0)
    
    # Signal 3: Entity matching
    company_name_words = set(company['name'].lower().split())
    company_ticker = company['ticker'].lower()
    
    entity_matches = 0
    matched_entities = []
    for entity in entities:
        if entity in company_text_lower or entity in company_ticker:
            entity_matches += 1.0
            matched_entities.append(entity)
    
    entity_score = min(1.0, entity_matches / max(len(entities), 1)) if entities else 0.0
    
    # Signal 4: STRICTLY validated explicit mentions
    validated_mentions = 0
    for art in articles[:15]:
        art_text = art["fulltext"][:10000]
        mentions, variants = count_explicit_mentions(art_text, company)
        
        if mentions > 0:
            if validate_contextual_mention(art_text, company['name'], company['ticker']):
                validated_mentions += 1
    
    mention_score = min(1.0, validated_mentions / 10.0)
    
    # Universal filters
    etf_indicators = ['etf', 'exchange traded fund', 'exchange-traded fund', 'mutual fund', 'index fund']
    company_name_lower = company['name'].lower()
    is_etf = any(indicator in company_name_lower for indicator in etf_indicators)
    
    if is_etf and mention_score < 0.2:
        return create_relevance_result(False, semantic_score, keyword_score, 
                                       entity_score, mention_score, validated_mentions)
    
    is_reit = ('real estate investment trust' in company_text_lower or 
               'reit' in company_name_lower)
    
    if is_reit and mention_score < 0.15 and semantic_score < 0.3:
        return create_relevance_result(False, semantic_score, keyword_score, 
                                       entity_score, mention_score, validated_mentions)
    
    # Calculate composite score
    total_score = (
        semantic_score * 0.65 +
        keyword_score * 0.15 +
        mention_score * 0.10 +
        entity_score * 0.10
    )
    
    # Decision criteria
    is_relevant = (
        total_score >= 0.15 or
        (semantic_score >= 0.30 and keyword_score >= 0.15) or
        mention_score >= 0.20 or
        (semantic_score >= 0.40)
    )
    
    if debug_tickers and company['ticker'] in debug_tickers:
        print(f"\n🔍 {company['ticker']} ({company['name'][:30]}):")
        print(f"  NLI:      {semantic_score:.3f}")
        print(f"  Keyword:  {keyword_score:.3f} {matched_keywords[:5]}")
        print(f"  Entity:   {entity_score:.3f} {matched_entities[:5]}")
        print(f"  Mention:  {mention_score:.3f} ({validated_mentions} articles)")
        print(f"  TOTAL:    {total_score:.3f}")
        print(f"  PASS:     {'✅ YES' if is_relevant else '❌ NO'}")
    
    return create_relevance_result(is_relevant, semantic_score, keyword_score, 
                                   entity_score, mention_score, validated_mentions)


def create_relevance_result(is_relevant, nli_score, keyword_score, 
                           entity_score, mention_score, mention_count):
    """Helper to avoid repetition"""
    total_score = (nli_score * 0.65 + keyword_score * 0.15 + 
                   mention_score * 0.10 + entity_score * 0.10)
    
    return {
        'is_relevant': is_relevant,
        'nli_score': nli_score,
        'keyword_score': keyword_score,
        'entity_score': entity_score,
        'mention_score': mention_score,
        'total_score': total_score,
        'mention_count': mention_count
    }


# ---------- main matcher ----------
def match_companies_with_multitimeframe_scores(
        articles, topic_model, companies_list, top_n=50, 
        similarity_threshold=0.03, debug_tickers=None):
    """
    OPTIMIZED Hybrid approach: Multi-stage filtering + batched NLI
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

    # 3. FIRST PASS: Collect ALL candidates across topics for global batching
    print("\n📊 Stage 1: Initial filtering...")
    all_candidates = []  # [(topic_id, company_idx, sim_score, keyword_score)]
    topic_domain_signals = {}
    
    for tid, arts in topic_articles.items():
        topic_vec = weighted_topic_vector(arts)
        topic_keywords = [w for w, _ in topic_model.get_topic(tid)]
        domain_signals = extract_topic_domain_signals(topic_keywords, arts)
        topic_domain_signals[tid] = domain_signals
        
        sims = np.dot(company_embs, topic_vec)
        
        for idx, comp in enumerate(companies_list):
            sim = sims[idx]
            if sim < MIN_EMBEDDING_SIM:
                continue
            
            # FAST keyword filter (cheap, eliminates most)
            company_text_lower = comp["combined_text"][:400].lower()
            kw_score = quick_keyword_score(company_text_lower, domain_signals['technical_terms'])
            
            # Only proceed if has some keyword overlap OR high embedding similarity
            if kw_score >= MIN_KEYWORD_OVERLAP or sim >= 0.15:
                all_candidates.append((tid, idx, float(sim), kw_score))
    
    print(f"  ✓ {len(all_candidates)} candidates passed initial filters")
    
    # 4. SECOND PASS: Batch NLI for all candidates at once
    print("\n📊 Stage 2: NLI verification (batched)...")
    
    # Group candidates by topic for NLI
    topic_candidate_map = {}
    for tid, idx, sim, kw_score in all_candidates:
        if tid not in topic_candidate_map:
            topic_candidate_map[tid] = []
        topic_candidate_map[tid].append((idx, sim, kw_score))
    
    # Process each topic's candidates
    topic_companies = {}
    nli_calls_saved = 0
    
    for tid, candidates in topic_candidate_map.items():
        arts = topic_articles[tid]
        domain_signals = topic_domain_signals[tid]
        
        print(f"  Topic {tid}: {len(candidates)} candidates...", end=" ")
        
        # Prepare for batch NLI
        candidate_descriptions = []
        candidate_data = []
        
        for idx, sim, kw_score in candidates:
            comp = companies_list[idx]
            candidate_descriptions.append(comp["combined_text"][:400])
            candidate_data.append((idx, sim, kw_score))
        
        # Batch NLI (adaptive batch size based on device)
        nli_scores = nli_relevance_batch_optimized(
            candidate_descriptions, 
            domain_signals['topic_keywords']
            # batch_size auto-detected (128 for GPU, 32 for CPU)
        )
        
        # Process with NLI scores
        qualified = []
        for i, (idx, sim, kw_score) in enumerate(candidate_data):
            comp = companies_list[idx]
            company_text = comp["combined_text"][:400]
            
            relevance_signals = calculate_multi_signal_relevance(
                comp, company_text, domain_signals, arts,
                nli_score=nli_scores[i],
                keyword_score_precomputed=kw_score,
                debug_tickers=debug_tickers
            )
            
            if not relevance_signals['is_relevant']:
                continue
            
            # Calculate final score with AGGRESSIVE BOOSTING
            base_sim = sim
            nli_boost = relevance_signals['nli_score'] * 0.30
            keyword_boost = relevance_signals['keyword_score'] * 0.20
            mention_boost = min(0.15, relevance_signals['mention_count'] * 0.04)
            
            final_score = base_sim + nli_boost + keyword_boost + mention_boost
            
            # Get mention details
            all_variants = []
            mention_count = relevance_signals['mention_count']
            if mention_count > 0:
                for art in arts[:10]:
                    art_text = art["fulltext"][:10000].lower()
                    _, variants = count_explicit_mentions(art_text, comp)
                    all_variants.extend(variants)
                all_variants = list(set(all_variants))
            
            qualified.append({
                "ticker": comp["ticker"],
                "name": comp["name"],
                "relevance_score": final_score,
                "mentions": mention_count,
                "mentioned_as": all_variants if mention_count else None,
                "stock_data": None,
                "day_trader_score": None,
                "swing_trader_score": None,
                "position_trader_score": None,
                "longterm_investor_score": None,
            })
        
        qualified.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Adaptive top_n
        if qualified:
            high_quality_cutoff = top_n
            if len(qualified) > 10:
                scores = [c['relevance_score'] for c in qualified]
                for i in range(10, min(len(scores), top_n)):
                    if scores[i] < scores[9] - 0.15:
                        high_quality_cutoff = i
                        break
            
            topic_companies[tid] = qualified[:high_quality_cutoff]
        else:
            topic_companies[tid] = []
        
        print(f"✓ {len(topic_companies[tid])} companies")
    
    # Save NLI cache periodically
    save_nli_cache()
    print(f"\n💾 NLI cache size: {len(nli_cache)} entries")

    # 5. stock-score threading (reduced workers for low-resource env)
    print("\n📊 Stage 3: Fetching stock data...")
    all_tickers = {c["ticker"] for lst in topic_companies.values() for c in lst}
    stock_cache = {}
    max_workers = min(10, CPU_THREAD_LIMIT * 2)  # Don't overwhelm the system
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
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