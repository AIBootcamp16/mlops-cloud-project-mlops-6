"""
embed_fit.py
- Validate → build corpus → fit TF-IDF → save artifacts.
Usage:
  python -m src.pipelines.embed_fit --style reds
"""
from __future__ import annotations
import argparse, json, os
from scipy import sparse
from joblib import dump
from src.validate import load_latest_frame
from src.reco.corpus import build_corpus
from src.reco.embed  import fit_tfidf

def run(style: str = "reds", outdir: str = "artifacts") -> None:
    os.makedirs(outdir, exist_ok=True)
    df = load_latest_frame(style)
    if df.empty:
        raise SystemExit("Empty frame after validation.")
    corpus = build_corpus(df)
    vec, X = fit_tfidf(corpus, ngram=(1,2), min_df=2)

    dump(vec, f"{outdir}/tfidf.pkl")
    sparse.save_npz(f"{outdir}/X_{style}.npz", X)
    with open(f"{outdir}/ids_{style}.json","w",encoding="utf-8") as f:
        json.dump(df["id"].astype(int).tolist(), f)
    with open(f"{outdir}/meta.json","w",encoding="utf-8") as f:
        json.dump({"style": style, "rows": int(X.shape[0]), "dims": int(X.shape[1])}, f, indent=2)
    print(f"[EMBED] saved to {outdir}/ (rows={X.shape[0]}, dims={X.shape[1]})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="reds")
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()
    run(args.style, args.outdir)
