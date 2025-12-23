"""Main evaluation pipeline"""

import numpy as np
import pandas as pd
from pathlib import Path

class LAFTEvaluator:
    def __init__(self, config):
        self.config = config
        self.results_dir = Path(config.RESULTS_DIR)
        self.results_dir.mkdir(exist_ok=True)
        
    def run_evaluation(self, predictions, vector_data):
        """Run complete LAFT evaluation"""
        
        # 1. Compute LAFT metrics
        laft_metrics = LAFTMetrics()
        laft_results = laft_metrics.compute_laft_scores(vector_data)
        df_laft = pd.DataFrame(laft_results)
        
        # 2. Compute FACTER baselines
        facter = FACTERBaseline()
        facter_results = {
            'CFR': facter.compute_cfr(predictions),
            'Accuracy_Gap': facter.compute_accuracy_gap(predictions)
        }
        
        # 3. Summary statistics
        summary = self._compute_summary(df_laft, facter_results)
        
        # 4. Save results
        self._save_results(df_laft, facter_results, summary)
        
        return {
            'laft': df_laft,
            'facter': facter_results,
            'summary': summary
        }
    
    def _compute_summary(self, df_laft, facter_results):
        """Compute summary statistics"""
        return {
            'n_samples': len(df_laft),
            'mean_ratio': df_laft['ratio'].mean(),
            'median_ratio': df_laft['ratio'].median(),
            'mean_leakage': df_laft['leakage_score'].mean(),
            'std_leakage': df_laft['leakage_score'].std(),
            'high_leakage_pct': (df_laft['leakage_score'].abs() > 0.5).mean(),
            **facter_results
        }
    
    def _save_results(self, df_laft, facter_results, summary):
        """Save results to disk"""
        df_laft.to_csv(self.results_dir / 'laft_scores.csv', index=False)
        
        pd.DataFrame([summary]).to_csv(self.results_dir / 'summary.csv', index=False)
        
        import json
        with open(self.results_dir / 'facter_baseline.json', 'w') as f:
            json.dump(facter_results, f, indent=2)
        
        print(f"\nResults saved to {self.results_dir}/")