# config.py
"""Configuration for LAFT experiments (LAFT-only)."""

from pathlib import Path


class Config:
    # --------------------
    # Data
    # --------------------
    # Use 'ml-1m' because your loader checks: if self.dataset_name == 'ml-1m'
    DATASET = "ml-1m"
    DATA_PATH = "/lambda/nfs/Aristella/FACTER/data/ml-1m/"
          # folder containing ratings.dat/users.dat/movies.dat
    SAMPLE_SIZE = 5000                   # None => use all rows (no sampling)
    ##After  load and merge the full MovieLens-1M tables (ratings.dat + users.dat + movies.dat) 
    # into one big dataframe of rating events,
    #randomly pick 5,000 rows from that merged dataframe (each row ≈ one user rated one movie at one time),
    # LAFT pipeline only on those 5,000 rows instead of the entire dataset.

    # --------------------
    # Model
    # --------------------
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
    LAYER_IDX = -2                       # second-to-last transformer layer
    DEVICE = "cuda"                      # will fall back to CPU inside your predictor/extractor if needed

    # --------------------
    # Gender prediction filtering
    # --------------------
    CONFIDENCE_THRESHOLD = 0.7           # keep only high-confidence â predictions

    # (Optional) If you later add an accuracy gate, you can use this, but it's not used now.
    MIN_ACCURACY = None                 # None => do not enforce an accuracy cutoff

    # --------------------
    # Proxy definition
    # --------------------
    # In your pipeline, the proxy is the *pair* (movie_title, genre).
    # These are the column names produced by your preprocessing in data.py.
    PROXY_FEATURES = ["clean_title", "main_genre"]

    # --------------------
    # Evaluation control
    # --------------------
    N_SAMPLES = None                     # None => evaluate all retained predictions
    RANDOM_SEED = 42                     # reproducibility (sampling, shuffling, etc.)

    # Proxy aggregation config (used by main.py + metrics.py)
    MIN_PROXY_SAMPLES = 5                # minimum samples per (movie_title, genre) group
    PROXY_WEIGHT_MODE = "n"              # "n" | "sqrt_n" | "uniform"

    # --------------------
    # Output
    # --------------------
    RESULTS_DIR = "./results/"
    SAVE_VECTORS = False                 # raw vectors are huge; save scores instead
    

    # OST controls (optional)
    OST_TOPK_PROXIES = None      # 例如 50：只测前50个 proxy；None 就全测
    OST_MAX_NEW_TOKENS = 32
    OST_DO_SAMPLE = False
    OST_TEMPERATURE = 0.0
    OST_TOP_P = 0.95
    OST_ONE_LINE_ONLY = True
    OST_SHOW_TOPK = 10
# thresholds for "both internal + output" evidence
    OST_LEAK_THR = 0.05          # |mean_log_vdr| >= 0.05
    OST_OUT_THR_EDIT = 0.30      # delta_MF_edit >= 0.30


# Optional: create results dir early (safe no-op if exists)
Path(Config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
