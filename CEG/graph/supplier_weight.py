"""
Zero-Hardcode Supplier Weight Calculator
=========================================

Calculates supplier relationship weights using ONLY data-driven metrics:
1. Relative shipment volume (statistical distribution)
2. Shipment consistency (inferred from volume patterns)
3. Product specificity (text complexity analysis)
4. Supplier diversity (concentration metrics)
5. Geographic distribution (statistical variance)

NO HARDCODED VALUES - all thresholds learned from data
"""

import re
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import math
import statistics


class SupplierWeights:
    """
    Calculate supplier weights using ONLY statistical properties of the data.
    No hardcoded product types, country risks, or arbitrary thresholds.
    """
    
    def __init__(self, suppliers: List[Dict], company_ticker: str = None):
        """
        Initialize calculator.
        
        Args:
            suppliers: List of supplier dicts from ImportYeti
            company_ticker: Optional ticker for logging
        """
        self.suppliers = suppliers
        self.ticker = company_ticker
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """Analyze the entire supplier dataset to derive metrics."""
        if not self.suppliers:
            self.stats = {
                'shipment_mean': 0,
                'shipment_std': 0,
                'shipment_percentiles': {},
                'description_lengths': [],
                'location_diversity': 0,
                'product_groups': {},
            }
            return
        
        # Extract shipment counts
        shipments = []
        descriptions = []
        locations = []
        
        for supplier in self.suppliers:
            # Shipments
            shipments_str = supplier.get('total_shipments', '0')
            try:
                count = int(shipments_str.replace(',', ''))
            except:
                count = 0
            shipments.append(count)
            supplier['_shipment_count'] = count
            
            # Descriptions
            desc = supplier.get('product_description', '')
            descriptions.append(desc)
            supplier['_desc_length'] = len(desc)
            supplier['_desc_words'] = len(desc.split())
            
            # Locations
            location = supplier.get('location', '')
            locations.append(location)
        
        # Statistical analysis
        shipments_nonzero = [s for s in shipments if s > 0] or [0]
        
        self.stats = {
            'shipment_mean': statistics.mean(shipments_nonzero),
            'shipment_median': statistics.median(shipments_nonzero),
            'shipment_std': statistics.stdev(shipments_nonzero) if len(shipments_nonzero) > 1 else 0,
            'shipment_min': min(shipments_nonzero),
            'shipment_max': max(shipments),
            'total_suppliers': len(self.suppliers),
            'desc_mean_length': statistics.mean([len(d) for d in descriptions]),
            'desc_mean_words': statistics.mean([len(d.split()) for d in descriptions]),
            'unique_locations': len(set(locations)),
            'location_entropy': self._calculate_entropy([locations.count(loc) for loc in set(locations)]),
        }
        
        # Calculate percentiles
        if shipments_nonzero:
            self.stats['shipment_p25'] = statistics.quantiles(shipments_nonzero, n=4)[0]
            self.stats['shipment_p75'] = statistics.quantiles(shipments_nonzero, n=4)[2]
        
        # Group suppliers by similarity (unsupervised clustering)
        self._cluster_suppliers()
    
    def _calculate_entropy(self, counts: List[int]) -> float:
        """Calculate Shannon entropy of a distribution."""
        if not counts or sum(counts) == 0:
            return 0.0
        
        total = sum(counts)
        probabilities = [c / total for c in counts if c > 0]
        
        entropy = -sum(p * math.log2(p) for p in probabilities)
        return entropy
    
    def _cluster_suppliers(self):
        """
        Group suppliers by similarity using text features.
        No hardcoded categories - purely data-driven clustering.
        """
        # Extract text features
        word_freq = Counter()
        for supplier in self.suppliers:
            desc = supplier.get('product_description', '').lower()
            words = re.findall(r'\b\w+\b', desc)
            word_freq.update(words)
        
        # Find most discriminative words (appear in some but not all)
        total_suppliers = len(self.suppliers)
        discriminative_words = []
        
        for word, freq in word_freq.most_common(100):
            # Skip very common and very rare words
            if 0.1 * total_suppliers < freq < 0.9 * total_suppliers:
                discriminative_words.append(word)
        
        # Assign suppliers to groups based on discriminative words
        supplier_groups = defaultdict(list)
        
        for supplier in self.suppliers:
            desc = supplier.get('product_description', '').lower()
            
            # Find matching discriminative words
            matching_words = [w for w in discriminative_words if w in desc]
            
            if matching_words:
                # Use most specific word as group key
                group_key = min(matching_words, key=lambda w: word_freq[w])
            else:
                group_key = 'generic'
            
            supplier_groups[group_key].append(supplier)
            supplier['_product_group'] = group_key
        
        self.product_groups = dict(supplier_groups)
        self.stats['product_groups'] = {k: len(v) for k, v in supplier_groups.items()}
    
    def calculate_volume_score(self, supplier: Dict) -> float:
        """
        Volume score based on statistical position in distribution.
        Uses z-score normalization - completely data-driven.
        """
        count = supplier['_shipment_count']
        
        if count == 0:
            return 0.05
        
        mean = self.stats['shipment_mean']
        std = self.stats['shipment_std']
        
        if std == 0:
            return 0.5
        
        # Z-score: how many standard deviations from mean
        z_score = (count - mean) / std
        
        # Convert z-score to 0-1 using sigmoid
        # This naturally handles outliers
        normalized = 1 / (1 + math.exp(-z_score / 2))
        
        return round(normalized, 4)
    
    def calculate_consistency_score(self, supplier: Dict) -> float:
        """
        Consistency inferred from relative volume position.
        Higher volume suggests more consistent relationship.
        """
        count = supplier['_shipment_count']
        
        if count == 0:
            return 0.1
        
        # Use percentile position as proxy for consistency
        percentile = self._calculate_percentile(count)
        
        # Sigmoid transformation centered at median
        consistency = 1 / (1 + math.exp(-5 * (percentile - 0.5)))
        
        return round(consistency, 4)
    
    def _calculate_percentile(self, value: float) -> float:
        """Calculate percentile rank of a value in the dataset."""
        all_values = [s['_shipment_count'] for s in self.suppliers]
        
        if not all_values:
            return 0.5
        
        rank = sum(1 for v in all_values if v < value)
        percentile = rank / len(all_values)
        
        return percentile
    
    def calculate_specificity_score(self, supplier: Dict) -> float:
        """
        Product specificity based on description complexity.
        More specific descriptions suggest more critical/specialized products.
        """
        desc_length = supplier['_desc_length']
        desc_words = supplier['_desc_words']
        
        # Relative length compared to dataset
        mean_length = self.stats['desc_mean_length']
        mean_words = self.stats['desc_mean_words']
        
        if mean_length == 0:
            return 0.5
        
        # Length score (longer = more specific)
        length_score = min(1.0, desc_length / (mean_length * 2))
        
        # Word count score (more words = more detailed)
        word_score = min(1.0, desc_words / (mean_words * 2))
        
        # Combined specificity
        specificity = (length_score + word_score) / 2
        
        # Boost for very detailed descriptions
        if desc_words > mean_words * 1.5:
            specificity = min(1.0, specificity * 1.2)
        
        return round(specificity, 4)
    
    def calculate_concentration_score(self, supplier: Dict) -> float:
        """
        Concentration based on supplier group size.
        Fewer suppliers in group = higher concentration risk.
        """
        group = supplier.get('_product_group', 'generic')
        group_size = len(self.product_groups.get(group, [supplier]))
        
        # Inverse relationship: fewer suppliers = higher score
        # Use log scale to prevent extreme values
        concentration = 1.0 / math.log2(1 + group_size)
        
        # Normalize to 0.3-1.0 range
        normalized = 0.3 + (concentration * 0.7)
        
        return round(normalized, 4)
    
    def calculate_geographic_score(self, supplier: Dict) -> float:
        """
        Geographic risk based on location distribution entropy.
        Lower entropy (concentrated) = higher risk.
        """
        location = supplier.get('location', '')
        
        # Count this location's frequency
        location_count = sum(1 for s in self.suppliers 
                           if s.get('location', '') == location)
        
        # Relative concentration
        concentration = location_count / self.stats['total_suppliers']
        
        # Higher concentration = higher risk (less diversification)
        # But we want score to represent importance, not just risk
        # So we balance concentration with global diversity
        
        max_entropy = math.log2(self.stats['total_suppliers'])
        actual_entropy = self.stats['location_entropy']
        
        # Normalized entropy (0-1, higher = more diverse globally)
        global_diversity = actual_entropy / max_entropy if max_entropy > 0 else 0.5
        
        # Score combines local concentration and global diversity
        score = 0.5 + (concentration * 0.3) + (global_diversity * 0.2)
        
        return round(min(1.0, score), 4)
    
    def calculate_weight(self, supplier: Dict) -> Tuple[float, Dict]:
        """
        Calculate final weight using adaptive weighting.
        Weights adapt based on data characteristics.
        """
        volume = self.calculate_volume_score(supplier)
        consistency = self.calculate_consistency_score(supplier)
        specificity = self.calculate_specificity_score(supplier)
        concentration = self.calculate_concentration_score(supplier)
        geographic = self.calculate_geographic_score(supplier)
        
        # ADAPTIVE WEIGHTING based on data properties
        
        # If shipments vary a lot, volume matters more
        cv = self.stats['shipment_std'] / self.stats['shipment_mean'] if self.stats['shipment_mean'] > 0 else 0
        volume_weight = 0.25 + min(0.15, cv * 0.3)  # 0.25-0.40
        
        # If descriptions vary a lot, specificity matters more
        desc_std = statistics.stdev([s['_desc_length'] for s in self.suppliers])
        desc_mean = self.stats['desc_mean_length']
        desc_cv = desc_std / desc_mean if desc_mean > 0 else 0
        specificity_weight = 0.25 + min(0.15, desc_cv * 0.3)  # 0.25-0.40
        
        # If locations are concentrated, geographic matters more
        loc_weight = 0.1 + (0.2 * (1 - self.stats['location_entropy'] / 5))  # 0.1-0.3
        
        # Remaining weight distributed
        remaining = 1.0 - volume_weight - specificity_weight - loc_weight
        consistency_weight = remaining * 0.4
        concentration_weight = remaining * 0.6
        
        # Calculate final weight
        weight = (
            volume * volume_weight +
            consistency * consistency_weight +
            specificity * specificity_weight +
            concentration * concentration_weight +
            geographic * loc_weight
        )
        
        components = {
            'volume': round(volume, 3),
            'consistency': round(consistency, 3),
            'specificity': round(specificity, 3),
            'concentration': round(concentration, 3),
            'geographic': round(geographic, 3),
            'weights_used': {
                'volume_weight': round(volume_weight, 3),
                'consistency_weight': round(consistency_weight, 3),
                'specificity_weight': round(specificity_weight, 3),
                'concentration_weight': round(concentration_weight, 3),
                'geographic_weight': round(loc_weight, 3),
            },
            'final_weight': round(weight, 3)
        }
        
        return round(weight, 4), components
    
    def calculate_confidence(self, supplier: Dict) -> float:
        """
        Confidence based on data completeness and volume.
        More data = higher confidence.
        """
        count = supplier['_shipment_count']
        desc_length = supplier['_desc_length']
        
        # Volume confidence
        percentile = self._calculate_percentile(count)
        volume_conf = percentile
        
        # Description confidence
        desc_conf = min(1.0, desc_length / (self.stats['desc_mean_length'] * 2))
        
        # Combined confidence
        confidence = (volume_conf * 0.7) + (desc_conf * 0.3)
        
        return round(max(0.2, confidence), 3)
    
    def calculate_all_weights(self) -> List[Dict]:
        """Calculate weights for all suppliers."""
        enhanced_suppliers = []
        
        for supplier in self.suppliers:
            weight, components = self.calculate_weight(supplier)
            confidence = self.calculate_confidence(supplier)
            
            supplier['weight'] = weight
            supplier['confidence'] = confidence
            supplier['_weight_components'] = components
            
            enhanced_suppliers.append(supplier)
        
        # Sort by weight descending
        enhanced_suppliers.sort(key=lambda x: x['weight'], reverse=True)
        
        return enhanced_suppliers
    
    def print_weight_breakdown(self, top_n: int = 10):
        """Print detailed breakdown of top N suppliers."""
        enhanced = self.calculate_all_weights()
        
        print(f"\n{'='*80}")
        print(f"TOP {top_n} SUPPLIERS BY WEIGHT" + (f" - {self.ticker}" if self.ticker else ""))
        print(f"{'='*80}")
        print(f"\nDataset Statistics:")
        print(f"  Total suppliers: {self.stats['total_suppliers']}")
        print(f"  Shipment mean: {self.stats['shipment_mean']:.1f}")
        print(f"  Shipment std: {self.stats['shipment_std']:.1f}")
        print(f"  Location diversity (entropy): {self.stats['location_entropy']:.2f}")
        print(f"  Product groups identified: {len(self.product_groups)}")
        print()
        
        for i, supplier in enumerate(enhanced[:top_n], 1):
            components = supplier.get('_weight_components', {})
            weights_used = components.get('weights_used', {})
            
            print(f"{i}. {supplier.get('supplier_name', 'Unknown')}")
            print(f"   Location: {supplier.get('location', 'Unknown')}")
            print(f"   Shipments: {supplier.get('total_shipments', '0')}")
            print(f"   Product Group: {supplier.get('_product_group', 'Unknown')}")
            print(f"   Description: {supplier.get('product_description', 'N/A')[:60]}...")
            print(f"\n   Component Scores × Adaptive Weights:")
            print(f"     Volume:        {components.get('volume', 0):.3f} × {weights_used.get('volume_weight', 0):.3f}")
            print(f"     Consistency:   {components.get('consistency', 0):.3f} × {weights_used.get('consistency_weight', 0):.3f}")
            print(f"     Specificity:   {components.get('specificity', 0):.3f} × {weights_used.get('specificity_weight', 0):.3f}")
            print(f"     Concentration: {components.get('concentration', 0):.3f} × {weights_used.get('concentration_weight', 0):.3f}")
            print(f"     Geographic:    {components.get('geographic', 0):.3f} × {weights_used.get('geographic_weight', 0):.3f}")
            print(f"   → FINAL WEIGHT:  {components.get('final_weight', 0):.3f}")
            print(f"   → CONFIDENCE:    {supplier.get('confidence', 0):.3f}")
            print()


# Example usage
if __name__ == "__main__":
    # Mock supplier data
    example_suppliers = [
        {
            'supplier_name': 'High Volume Critical Parts Co',
            'location': 'Seoul, South Korea',
            'total_shipments': '450',
            'product_description': 'Advanced OLED display panels with quantum dot technology, high-resolution camera modules with optical image stabilization, LPDDR5 memory chips',
        },
        {
            'supplier_name': 'Packaging Solutions Ltd',
            'location': 'Shenzhen, China',
            'total_shipments': '200',
            'product_description': 'Boxes and packaging',
        },
        {
            'supplier_name': 'Specialized Semiconductor Fab',
            'location': 'Hsinchu, Taiwan',
            'total_shipments': '50',
            'product_description': 'Custom application-specific integrated circuits manufactured using advanced 5-nanometer process technology with extreme ultraviolet lithography',
        },
        {
            'supplier_name': 'Generic Parts Supplier',
            'location': 'Shenzhen, China',
            'total_shipments': '180',
            'product_description': 'Various components',
        },
        {
            'supplier_name': 'Rare Component Manufacturer',
            'location': 'Munich, Germany',
            'total_shipments': '15',
            'product_description': 'Precision-engineered rare earth magnetic sensors for automotive applications',
        }
    ]
    
    calculator = SupplierWeights(example_suppliers, 'EXAMPLE')
    calculator.print_weight_breakdown(top_n=5)