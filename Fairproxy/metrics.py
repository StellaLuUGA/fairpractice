"""LAFT metrics + FACTER baselines"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class LAFTMetrics:
    """
    Leakage-Aware Fairness Testing metrics
    Based on internal vector geometry
    """
    
    def vector_distance_ratio(self, v0, v1, v2, metric='l2'):
        """
        YOUR CORE CONTRIBUTION
        
        Compute: ||v0 - v1||_2 / ||v0 - v2||_2
        
        Args:
            v0: Neutral context vector (no gender)
            v1: Vector with predicted gender
            v2: Vector with opposite gender
            metric: 'l2' or 'cosine'
        
        Returns:
            ratio: Leakage indicator
        """
        if metric == 'l2':
            dist_to_predicted = np.linalg.norm(v0 - v1, ord=2)
            dist_to_opposite = np.linalg.norm(v0 - v2, ord=2)
        elif metric == 'cosine':
            dist_to_predicted = 1 - cosine_similarity([v0], [v1])[0, 0]
            dist_to_opposite = 1 - cosine_similarity([v0], [v2])[0, 0]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Avoid division by zero
        epsilon = 1e-8
        ratio = dist_to_predicted / (dist_to_opposite + epsilon)
        
        return ratio
    
    def leakage_score(self, ratio):
        """
        Convert ratio to interpretable leakage score
        
        ratio = 1 → score = 0 (no leakage)
        ratio < 1 → score < 0 (closer to predicted)
        ratio > 1 → score > 0 (closer to opposite)
        """
        return np.log(ratio)
    
    def compute_laft_scores(self, vector_data):
        """Compute LAFT scores for all samples"""
        results = []
        
        for sample in vector_data:
            ratio = self.vector_distance_ratio(
                sample['v0'],
                sample['v1'],
                sample['v2']
            )
            
            results.append({
                'sample_id': sample['sample_id'],
                'movie': sample['movie_title'],
                'genre': sample['genre'],
                'predicted_gender': sample['predicted_gender'],
                'ratio': ratio,
                'leakage_score': self.leakage_score(ratio)
            })
        
        return results


class FACTERBaseline:
    """FACTER metrics for comparison"""
    
    def compute_cfr(self, predictions):
        """
        Counterfactual Fairness Ratio
        Measures prediction consistency across gender flips
        """
        correct_same_gender = sum(1 for p in predictions if p['correct'])
        total = len(predictions)
        
        return correct_same_gender / total
    
    def compute_accuracy_gap(self, predictions):
        """Accuracy gap between Male and Female predictions"""
        male_acc = np.mean([p['correct'] for p in predictions if p['predicted_gender'] == 'Male'])
        female_acc = np.mean([p['correct'] for p in predictions if p['predicted_gender'] == 'Female'])
        
        return abs(male_acc - female_acc)