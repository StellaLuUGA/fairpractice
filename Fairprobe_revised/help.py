
import numpy as np, pandas as pd
X = np.load("/home/ubuntu/Aristella/FACTER/Fairprobe_revised/outputs_reco_v0v1v2/reco_emb_v0.npy")
k = min(50, X.shape[1])  # first 50 dims
df = pd.DataFrame(X[:, :k], columns=[f"dim_{i}" for i in range(k)])
df.to_csv("/home/ubuntu/Aristella/FACTER/Fairprobe_revised/outputs_reco_v0v1v2/reco_emb_v0.csv", index=False)
print("Wrote outputs_reco_v0v1v2/reco_emb_v0.csv with shape", df.shape)

X = np.load("/home/ubuntu/Aristella/FACTER/Fairprobe_revised/outputs_reco_v0v1v2/reco_emb_v1.npy")
k = min(50, X.shape[1])  # first 50 dims
df = pd.DataFrame(X[:, :k], columns=[f"dim_{i}" for i in range(k)])
df.to_csv("/home/ubuntu/Aristella/FACTER/Fairprobe_revised/outputs_reco_v0v1v2/reco_emb_v1.csv", index=False)
print("Wrote outputs_reco_v0v1v2/reco_emb_v1.csv with shape", df.shape)

X = np.load("/home/ubuntu/Aristella/FACTER/Fairprobe_revised/outputs_reco_v0v1v2/reco_emb_v2.npy")
k = min(50, X.shape[1])  # first 50 dims
df = pd.DataFrame(X[:, :k], columns=[f"dim_{i}" for i in range(k)])
df.to_csv("/home/ubuntu/Aristella/FACTER/Fairprobe_revised/outputs_reco_v0v1v2/reco_emb_v2.csv", index=False)
print("Wrote outputs_reco_v0v1v2/reco_emb_v2.csv with shape", df.shape)





