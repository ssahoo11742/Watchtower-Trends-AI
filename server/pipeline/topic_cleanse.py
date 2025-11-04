from gensim.models import KeyedVectors
from statistics import mean
import numpy as np

class BERTopicKeywordFilter:
    """
    Filters BERTopic keywords to keep only semantically coherent, domain-specific terms.
    Removes generic/broad terms like 'system', 'technology', 'research'.
    """
    
    def __init__(self, model_path='GoogleNews-vectors-negative300.bin'):
        """Initialize with word2vec model"""
        print("🔄 Loading Word2Vec model for keyword filtering...")
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        print("✅ Model loaded successfully!\n")
        
        self.generic_words = set([
            "science", "technology", "research", "system", "data", "information",
            "industry", "development", "company", "time", "general", "field",
            "university", "business", "sector", "application", "world", "service",
            "product", "management", "process", "solution", "network", "platform",
            "operations", "global", "international", "national", "project", "program",
            "engineering", "design", "analysis", "method", "approach", "standard",
            "quality", "performance", "resources", "infrastructure", "digital",
            "advanced", "modern", "new", "innovation", "strategy", "professional",
            "based", "using", "used", "including", "related", "available", "current"
        ])
    
    def filter_topic_keywords(self, keywords, min_keep=3, max_keep=6, verbose=False):
        """
        Filter keywords from a single topic.
        
        Args:
            keywords: List of (word, score) tuples from BERTopic
            min_keep: Minimum keywords to keep
            max_keep: Maximum keywords to keep
            verbose: Print filtering details
            
        Returns:
            List of filtered (word, score) tuples
        """
        # Separate single words from phrases
        single_words = [(w, s) for w, s in keywords if ' ' not in w]
        phrases = [(w, s) for w, s in keywords if ' ' in w]
        
        # Always keep domain-specific phrases
        domain_phrases = [
            p for p in phrases 
            if any(term in p[0] for term in ['quantum', 'semiconductor', 'crypto', 'cryptography'])
        ]
        
        words = [word for word, score in single_words]
        valid_words = [w for w in words if w in self.model]
        
        if len(valid_words) < min_keep:
            # Not enough valid words, return original
            return keywords[:min_keep]
        
        # Compute similarity matrix
        similarity_matrix = {}
        for w1 in valid_words:
            similarity_matrix[w1] = {}
            for w2 in valid_words:
                if w1 == w2:
                    similarity_matrix[w1][w2] = 1.0
                else:
                    try:
                        similarity_matrix[w1][w2] = float(self.model.similarity(w1, w2))
                    except:
                        similarity_matrix[w1][w2] = 0.0
        
        # Calculate metrics for each word
        metrics = {}
        generic_vecs = [self.model[w] for w in self.generic_words if w in self.model]
        generic_center = np.mean(generic_vecs, axis=0) if generic_vecs else None
        
        for word in valid_words:
            # Cohesion: average similarity to other words
            similarities = [similarity_matrix[word][other] for other in valid_words if other != word]
            cohesion = mean(similarities) if similarities else 0.0
            
            # Centrality: ratio of strong connections
            strong_connections = sum(1 for s in similarities if s > 0.4)
            centrality = strong_connections / len(similarities) if similarities else 0.0
            
            # Niche: distance from generic centroid
            if generic_center is not None:
                try:
                    word_vec = self.model[word]
                    niche_distance = np.linalg.norm(word_vec - generic_center)
                except:
                    niche_distance = 0.0
            else:
                niche_distance = 0.0
            
            metrics[word] = {
                'cohesion': cohesion,
                'centrality': centrality,
                'niche_raw': niche_distance
            }
        
        # Normalize niche scores
        max_niche = max(m['niche_raw'] for m in metrics.values())
        for word in metrics:
            metrics[word]['niche_norm'] = metrics[word]['niche_raw'] / max_niche if max_niche > 0 else 0.0
        
        # Calculate final scores
        final_scores = {}
        for word in valid_words:
            m = metrics[word]
            
            # Base score
            score = (
                m['cohesion'] * 0.40 +
                m['niche_norm'] * 0.35 +
                m['centrality'] * 0.25
            )
            
            # Penalize generic words
            if word.lower() in self.generic_words:
                score -= 0.20
            
            # Frequency penalty for very common words
            try:
                vocab_size = len(self.model.key_to_index)
                word_rank = self.model.key_to_index.get(word, vocab_size)
                normalized_rarity = word_rank / vocab_size
                
                if normalized_rarity < 0.05:
                    score -= 0.20
                elif normalized_rarity < 0.15:
                    score -= 0.10
            except:
                pass
            
            final_scores[word] = score
        
        # Sort by score
        sorted_words = sorted(final_scores.items(), key=lambda x: -x[1])
        
        # Find natural cutoff
        scores_only = [s for _, s in sorted_words]
        
        # Calculate score gaps
        gaps = []
        for i in range(min_keep, min(len(sorted_words) - 1, max_keep + 1)):
            gap = scores_only[i-1] - scores_only[i]
            gaps.append((i, gap))
        
        # Find largest gap
        if gaps:
            cutoff_idx = max(gaps, key=lambda x: x[1])[0]
        else:
            cutoff_idx = min_keep
        
        # Ensure within bounds
        cutoff_idx = max(min_keep, min(cutoff_idx, max_keep))
        
        # Also check for negative score boundary
        first_negative = next((i for i, (w, s) in enumerate(sorted_words) if s < 0), len(sorted_words))
        cutoff_idx = min(cutoff_idx, max(min_keep, first_negative))
        
        # Get filtered words
        filtered_words = [w for w, _ in sorted_words[:cutoff_idx]]
        
        if verbose:
            all_words = [w for w, _ in keywords]
            domain_phrase_words = [w for w, _ in domain_phrases]
            print(f"\n  Original keywords: {', '.join(all_words)}")
            print(f"  Domain phrases kept: {', '.join(domain_phrase_words)}")
            print(f"  Filtered keywords: {', '.join(filtered_words)}")
            print(f"  Removed: {', '.join([w for w in words if w not in filtered_words])}")
        
        # **FIX: Return tuples with original scores, maintaining proper format**
        # Get remaining slots after domain phrases
        remaining_slots = max_keep - len(domain_phrases)
        
        # Return filtered keywords WITH SCORES as tuples
        filtered_single_word_tuples = [(word, score) for word, score in single_words if word in filtered_words][:remaining_slots]
        
        return domain_phrases + filtered_single_word_tuples

    def filter_all_topics(self, topic_model, min_keep=3, max_keep=6, verbose=True):
        """
        Filter keywords for all topics in a BERTopic model.
        
        Args:
            topic_model: Fitted BERTopic model
            min_keep: Minimum keywords per topic
            max_keep: Maximum keywords per topic
            verbose: Print filtering details
            
        Returns:
            Dictionary mapping topic_id to filtered keywords
        """
        filtered_topics = {}
        topic_info = topic_model.get_topic_info()
        
        print(f"🔍 Filtering keywords for {len(topic_info)} topics...")
        print(f"{'='*80}\n")
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            
            if topic_id == -1:
                continue
            
            # Get original keywords
            keywords = topic_model.get_topic(topic_id)
            
            if verbose:
                print(f"Topic {topic_id}:")
            
            # Filter keywords
            filtered = self.filter_topic_keywords(
                keywords, 
                min_keep=min_keep, 
                max_keep=max_keep,
                verbose=verbose
            )
            
            filtered_topics[topic_id] = filtered
            
            if verbose:
                print()
        
        return filtered_topics
    
    def update_topic_model_representation(self, topic_model, filtered_topics):
        """
        Update BERTopic model's topic representations with filtered keywords.
        
        Args:
            topic_model: Fitted BERTopic model
            filtered_topics: Dictionary from filter_all_topics()
            
        Returns:
            Updated topic_model
        """
        # Update the internal topic representations
        for topic_id, keywords in filtered_topics.items():
            if topic_id in topic_model.topic_representations_:
                topic_model.topic_representations_[topic_id] = keywords
        
        return topic_model


# ============================================================================
# INTEGRATION FUNCTION FOR YOUR PIPELINE
# ============================================================================

def filter_bertopic_keywords(topic_model, model_path='GoogleNews-vectors-negative300.bin', 
                             min_keep=3, max_keep=6, verbose=True):
    """
    Main function to filter BERTopic keywords.
    
    Args:
        topic_model: Fitted BERTopic model
        model_path: Path to word2vec model
        min_keep: Minimum keywords to keep per topic
        max_keep: Maximum keywords to keep per topic
        verbose: Print filtering details
        
    Returns:
        Tuple of (filtered_topics dict, updated topic_model)
    """
    # Initialize filter
    keyword_filter = BERTopicKeywordFilter(model_path)
    
    # Filter all topics
    filtered_topics = keyword_filter.filter_all_topics(
        topic_model, 
        min_keep=min_keep, 
        max_keep=max_keep,
        verbose=verbose
    )
    
    # Update the model
    updated_model = keyword_filter.update_topic_model_representation(
        topic_model, 
        filtered_topics
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print("📊 FILTERING SUMMARY")
    print(f"{'='*80}\n")
    
    for topic_id in sorted(filtered_topics.keys()):
        original_keywords = ", ".join([word for word, _ in topic_model.get_topic(topic_id)][:10])
        filtered_keywords = ", ".join([word for word, _ in filtered_topics[topic_id]])
        
        print(f"Topic {topic_id}:")
        print(f"  Before: {original_keywords}")
        print(f"  After:  {filtered_keywords}\n")
    
    return filtered_topics, updated_model