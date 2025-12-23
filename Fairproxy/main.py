"""Main experiment runner"""

import sys
from config import Config
from data import load_data
from llm_predictor import LLaMAGenderPredictor
from vector_extractor import VectorExtractor
from proxy import ProxyAnalyzer
from evaluation import LAFTEvaluator

def main():
    print("="*60)
    print("LAFT: Leakage-Aware Fairness Testing")
    print("="*60)
    
    config = Config()
    
    # ========== STEP 1: Load Data ==========
    print("\n[1/5] Loading MovieLens-1M dataset...")
    data = load_data(config)
    print(f"Loaded {len(data)} samples")
    
    # ========== STEP 2: Proxy Analysis ==========
    print("\n[2/5] Analyzing proxy features...")
    proxy_analyzer = ProxyAnalyzer(data)
    proxies = proxy_analyzer.identify_strong_proxies()
    
    # ========== STEP 3: Gender Prediction ==========
    print("\n[3/5] Predicting gender from (movie, genre)...")
    predictor = LLaMAGenderPredictor(config.MODEL_NAME, config.DEVICE)
    predictions = predictor.predict_batch(data, config.CONFIDENCE_THRESHOLD)
    
    # ========== STEP 4: Vector Extraction ==========
    print("\n[4/5] Extracting internal LLaMA vectors...")
    extractor = VectorExtractor(
        predictor.model,
        predictor.tokenizer,
        config.LAYER_IDX,
        config.DEVICE
    )
    vector_data = extractor.extract_batch(predictions[:config.N_SAMPLES])
    print(f"Extracted vectors for {len(vector_data)} samples")
    
    # ========== STEP 5: Evaluation ==========
    print("\n[5/5] Running LAFT evaluation...")
    evaluator = LAFTEvaluator(config)
    results = evaluator.run_evaluation(predictions, vector_data)
    
    # ========== Print Summary ==========
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    summary = results['summary']
    print(f"Samples evaluated: {summary['n_samples']}")
    print(f"\nLAFT Metrics:")
    print(f"  Mean ratio: {summary['mean_ratio']:.4f}")
    print(f"  Median ratio: {summary['median_ratio']:.4f}")
    print(f"  Mean leakage score: {summary['mean_leakage']:.4f} ± {summary['std_leakage']:.4f}")
    print(f"  High leakage (>0.5): {summary['high_leakage_pct']*100:.1f}%")
    
    print(f"\nFACTER Baselines:")
    print(f"  CFR: {summary['CFR']:.4f}")
    print(f"  Accuracy Gap: {summary['Accuracy_Gap']:.4f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()