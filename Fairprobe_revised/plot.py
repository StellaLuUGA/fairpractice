


import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

LABEL_ORDER = ["Male", "Female", "Neutral"]

def read_labeled_file(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 1:
                label, title = parts[0].strip(), ""
            else:
                label, title = parts[0].strip(), parts[1].strip()

            if label not in LABEL_ORDER:
                label = "Neutral"
            rows.append({"label": label, "title": title})
    return pd.DataFrame(rows)

def count_labels(df: pd.DataFrame) -> pd.Series:
    c = df["label"].value_counts()
    return pd.Series({lab: int(c.get(lab, 0)) for lab in LABEL_ORDER})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_png", type=str, default="gender_distribution_v0v1v2.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    v0_path = os.path.join(args.in_dir, "reco_titles_v0_labeled.txt")
    v1_path = os.path.join(args.in_dir, "reco_titles_v1_labeled.txt")
    v2_path = os.path.join(args.in_dir, "reco_titles_v2_labeled.txt")

    for p in [v0_path, v1_path, v2_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing labeled file: {p}")

    df0 = read_labeled_file(v0_path)
    df1 = read_labeled_file(v1_path)
    df2 = read_labeled_file(v2_path)

    c0 = count_labels(df0)
    c1 = count_labels(df1)
    c2 = count_labels(df2)

    counts = pd.DataFrame({"v0": c0, "v1": c1, "v2": c2}, index=LABEL_ORDER)
    totals = counts.sum(axis=0)
    props = counts.divide(totals, axis=1).fillna(0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = [0.0] * props.shape[1]

    for lab in LABEL_ORDER:
        vals = props.loc[lab].values
        bars = ax.bar(["v0", "v1", "v2"], vals, bottom=bottoms, label=lab)
        for i, b in enumerate(bars):
            cnt = int(counts.loc[lab].iloc[i])
            if vals[i] >= 0.06:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    bottoms[i] + vals[i] / 2,
                    str(cnt),
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        bottoms = [bottoms[i] + vals[i] for i in range(len(bottoms))]

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion of labeled recommendation titles")
    ax.set_title("Gender Association Distribution of Recommended Titles (v0 vs v1 vs v2)")
    ax.legend(title="Label")

    for i, name in enumerate(["v0", "v1", "v2"]):
        ax.text(i, 1.02, f"n={int(totals[name])}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=args.dpi)
    plt.close()
    print("Saved:", args.out_png)

    out_csv = os.path.splitext(args.out_png)[0] + "_counts.csv"
    counts.to_csv(out_csv)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()

