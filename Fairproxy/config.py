# Hyperparameters and paths
"""Configuration for LAFT experiments"""

class Config:
    # Data
    DATASET = "movielens-1m"
    DATA_PATH = "./data/ml-1m/"
    SAMPLE_SIZE = 2500
    
    # Model
    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    LAYER_IDX = -2  # Which layer to extract vectors from
    DEVICE = "cuda"
    
    # Prediction
    CONFIDENCE_THRESHOLD = 0.8  # For high-confidence filtering
    MIN_ACCURACY = 0.7          # Minimum prediction accuracy
    
    # Proxies
    PROXY_FEATURES = ["movie_title", "genre"]
    
    # Metrics
    DISTANCE_METRIC = "l2"  # or "cosine"
    
    # Evaluation
    N_SAMPLES = 1000        # Number of samples to evaluate
    RANDOM_SEED = 42
    
    # Output
    RESULTS_DIR = "./results/"
    SAVE_VECTORS = True