"""
ticker_match.py  --  Unified Hybrid semantic matcher with configurable depth
Depth modes:
1: FAST (ms-marco, optimized pipeline, early exits)
2: BALANCED (ms-marco, standard pipeline)
3: ACCURATE (deberta, optimized pipeline)
4: MAXIMUM (deberta, standard pipeline, full validation)
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

# ==================== CONFIGURATION ====================
# Default depth mode (will be set before model loading)
DEPTH_MODE = 1

# Depth mode configurations
DEPTH_CONFIGS = {
    1: {  # FAST
        'name': '⚡ FAST',
        'nli_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        'nli_max_length': 128,
        'min_keyword_overlap': 0.05,
        'min_embedding_sim': 0.05,
        'use_early_exit': True,
        'batch_size_gpu': 128,
        'batch_size_cpu': 32,
        'nli_hypotheses': 4,
        'description': 'Fastest mode - good for large datasets'
    },
    2: {  # BALANCED
        'name': '⚖️ BALANCED',
        'nli_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        'nli_max_length': 128,
        'min_keyword_overlap': 0.03,
        'min_embedding_sim': 0.03,
        'use_early_exit': False,
        'batch_size_gpu': 64,
        'batch_size_cpu': 32,
        'nli_hypotheses': 6,
        'description': 'Balanced speed/accuracy'
    },
    3: {  # ACCURATE
        'name': '🎯 ACCURATE',
        'nli_model': 'cross-encoder/nli-deberta-v3-small',
        'nli_max_length': 128,
        'min_keyword_overlap': 0.05,
        'min_embedding_sim': 0.05,
        'use_early_exit': True,
        'batch_size_gpu': 128,
        'batch_size_cpu': 32,
        'nli_hypotheses': 4,
        'description': 'High accuracy with optimizations'
    },
    4: {  # MAXIMUM
        'name': '💎 MAXIMUM',
        'nli_model': 'cross-encoder/nli-deberta-v3-small',
        'nli_max_length': 256,
        'min_keyword_overlap': 0.03,
        'min_embedding_sim': 0.03,
        'use_early_exit': False,
        'batch_size_gpu': 64,
        'batch_size_cpu': 32,
        'nli_hypotheses': 6,
        'description': 'Maximum accuracy - slower but best results'
    }
}

# Global variables that will be set by set_depth_mode()
CONFIG = None
EMB_CACHE_FILE = "company_embs_384.pkl"
NLI_CACHE_FILE = None
SIMILARITY_GATE = 0.03
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
NLI_MODEL_NAME = None
NLI_MAX_LENGTH = None
MIN_KEYWORD_OVERLAP = None
MIN_EMBEDDING_SIM = None
USE_EARLY_EXIT = None
NLI_HYPOTHESES = None
CPU_THREAD_LIMIT = 4

# Models (will be loaded lazily)
semantic_model = None
nli_model = None
device = None
nli_cache = {}

def set_depth_mode(depth):
    """Set the depth mode and initialize models"""
    global DEPTH_MODE, CONFIG, NLI_MODEL_NAME, NLI_MAX_LENGTH
    global MIN_KEYWORD_OVERLAP, MIN_EMBEDDING_SIM, USE_EARLY_EXIT, NLI_HYPOTHESES
    global NLI_CACHE_FILE, semantic_model, nli_model, device, nli_cache
    
    if depth not in DEPTH_CONFIGS:
        raise ValueError(f"Invalid depth: {depth}. Must be 1-4.")
    
    DEPTH_MODE = depth
    CONFIG = DEPTH_CONFIGS[DEPTH_MODE]
    
    # Update all dependent variables
    NLI_MODEL_NAME = CONFIG['nli_model']
    NLI_MAX_LENGTH = CONFIG['nli_max_length']
    MIN_KEYWORD_OVERLAP = CONFIG['min_keyword_overlap']
    MIN_EMBEDDING_SIM = CONFIG['min_embedding_sim']
    USE_EARLY_EXIT = CONFIG['use_early_exit']
    NLI_HYPOTHESES = CONFIG['nli_hypotheses']
    NLI_CACHE_FILE = f"nli_cache_depth{DEPTH_MODE}.pkl"
    
    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(CPU_THREAD_LIMIT)
        print(f"🖥️  Running on CPU with {CPU_THREAD_LIMIT} threads")
    
    # Load models
    print(f"\n{'='*80}")
    print(f"🔧 DEPTH MODE: {CONFIG['name']} - {CONFIG['description']}")
    print(f"{'='*80}")
    print("📥 Loading models...")
    
    semantic_model = SentenceTransformer(EMB_MODEL_NAME, device=device)
    print(f"  ✓ Semantic model loaded ({EMB_MODEL_NAME})")
    
    nli_model = CrossEncoder(NLI_MODEL_NAME, max_length=NLI_MAX_LENGTH, device=device)
    print(f"  ✓ NLI model loaded ({NLI_MODEL_NAME})")
    print(f"  ✓ Max context length: {NLI_MAX_LENGTH}")
    print(f"  ✓ Using device: {device}")
    print(f"  ✓ Early exit: {'Enabled' if USE_EARLY_EXIT else 'Disabled'}")
    print(f"{'='*80}\n")
    
    # Load NLI cache
    if os.path.exists(NLI_CACHE_FILE):
        with open(NLI_CACHE_FILE, "rb") as f:
            nli_cache = pickle.load(f)
        print(f"📦 Loaded {len(nli_cache)} cached NLI results")
    else:
        nli_cache = {}

def _ensure_initialized():
    """Ensure models are loaded before use"""
    if semantic_model is None or nli_model is None:
        raise RuntimeError(
            "Models not initialized! Call set_depth_mode(depth) before using ticker_match functions."
        )

def save_nli_cache():
    """Save NLI cache to disk"""
    with open(NLI_CACHE_FILE, "wb") as f:
        pickle.dump(nli_cache, f)

def hash_text(text):
    """Create hash for cache keys"""
    return hashlib.md5(text.encode()).hexdigest()

# ---------- helpers ----------
def load_or_build_company_embeddings(companies_list):
    _ensure_initialized()
    if os.path.exists(EMB_CACHE_FILE):
        return pickle.load(open(EMB_CACHE_FILE, "rb"))
    embs = semantic_model.encode([c["combined_text"][:400] for c in companies_list],
                                 normalize_embeddings=True, convert_to_numpy=True)
    pickle.dump(embs, open(EMB_CACHE_FILE, "wb"))
    return embs

def weighted_topic_vector(articles):
    _ensure_initialized()
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
    _ensure_initialized()
    
    if not topic_keywords or not company_descriptions:
        return [0.0] * len(company_descriptions)
    
    # Adaptive batch size based on device and mode
    if batch_size is None:
        batch_size = CONFIG['batch_size_gpu'] if device != "cpu" else CONFIG['batch_size_cpu']
    
    # Create hypotheses (number based on depth mode)
    hypotheses = []
    for kw in topic_keywords[:NLI_HYPOTHESES]:
        hypotheses.append(f"This company works with {kw}.")
        hypotheses.append(f"This company specializes in {kw}.")
    
    hyp_tuple = tuple(hypotheses)
    all_scores = []
    
    # Collect uncached pairs
    uncached_pairs = []
    uncached_indices = []
    
    for i, desc in enumerate(company_descriptions):
        # Use longer context for maximum mode
        context_length = 200 if DEPTH_MODE <= 2 else 300
        desc_hash = hash_text(desc[:context_length])
        cache_key = (desc_hash, hyp_tuple)
        
        if cache_key in nli_cache:
            all_scores.append(nli_cache[cache_key])
        else:
            all_scores.append(None)  # Placeholder
            uncached_indices.append(i)
            for hyp in hypotheses:
                uncached_pairs.append([desc[:context_length], hyp])
    
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
            context_length = 200 if DEPTH_MODE <= 2 else 300
            desc_hash = hash_text(desc[:context_length])
            cache_key = (desc_hash, hyp_tuple)
            nli_cache[cache_key] = max_prob
            
            all_scores[orig_idx] = max_prob
    
    return all_scores

def calculate_multi_signal_relevance(company, company_text, domain_signals, articles, 
                                    nli_score=None, keyword_score_precomputed=None,
                                    debug_tickers=None):
    """
    Multi-signal relevance with optional early exit based on depth mode.
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
    
    # EARLY EXIT (only in optimized modes)
    if USE_EARLY_EXIT and semantic_score < 0.20 and keyword_score < 0.10:
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
    
    # Decision criteria (slightly stricter in maximum mode)
    threshold_adjustment = 0.02 if DEPTH_MODE == 4 else 0.0
    
    is_relevant = (
        total_score >= (0.15 + threshold_adjustment) or
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
    Unified hybrid approach with configurable depth.
    """
    _ensure_initialized()

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

    # 3. FIRST PASS: Collect ALL candidates across topics
    print("\n📊 Stage 1: Initial filtering...")
    all_candidates = []
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
            
            # FAST keyword filter
            company_text_lower = comp["combined_text"][:400].lower()
            kw_score = quick_keyword_score(company_text_lower, domain_signals['technical_terms'])
            
            # Threshold based on mode
            if kw_score >= MIN_KEYWORD_OVERLAP or sim >= 0.15:
                all_candidates.append((tid, idx, float(sim), kw_score))
    
    print(f"  ✓ {len(all_candidates)} candidates passed initial filters")
    
    # 4. SECOND PASS: Batch NLI
    print(f"\n📊 Stage 2: NLI verification ({CONFIG['name']})...")
    
    # Group candidates by topic
    topic_candidate_map = {}
    for tid, idx, sim, kw_score in all_candidates:
        if tid not in topic_candidate_map:
            topic_candidate_map[tid] = []
        topic_candidate_map[tid].append((idx, sim, kw_score))
    
    # Process each topic's candidates
    topic_companies = {}
    
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
        
        # Batch NLI
        nli_scores = nli_relevance_batch_optimized(
            candidate_descriptions, 
            domain_signals['topic_keywords']
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
            
            # Calculate final score
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
    
    # Save NLI cache
    save_nli_cache()
    print(f"\n💾 NLI cache size: {len(nli_cache)} entries")

    # 5. Stock data fetching
    print("\n📊 Stage 3: Fetching stock data...")
    all_tickers = {c["ticker"] for lst in topic_companies.values() for c in lst}
    stock_cache = {}
    max_workers = min(10, CPU_THREAD_LIMIT * 2) if device == "cpu" else 10
    
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