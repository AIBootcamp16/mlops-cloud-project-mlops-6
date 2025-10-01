"""
embed_fit.py
- Validate → build corpus → fit TF-IDF → save artifacts + log to WandB
"""
from __future__ import annotations
import argparse, json, os, time
from scipy import sparse
from joblib import dump
from src.validate import load_latest_frame
from src.reco.corpus import build_corpus
from src.reco.embed  import fit_tfidf
import wandb


def run(style: str = "reds", outdir: str = "artifacts") -> None:
    os.makedirs(outdir, exist_ok=True)

    # 1. 데이터 불러오기
    df = load_latest_frame(style)
    if df.empty:
        raise SystemExit("Empty frame after validation.")

    # 2. 코퍼스 생성 및 TF-IDF 학습
    corpus = build_corpus(df)
    vec, X = fit_tfidf(corpus, ngram=(1, 2), min_df=2)

    # 3. 로컬 저장
    dump(vec, f"{outdir}/tfidf_{style}.pkl")
    sparse.save_npz(f"{outdir}/X_{style}.npz", X)
    with open(f"{outdir}/ids_{style}.json", "w", encoding="utf-8") as f:
        json.dump(df["id"].astype(int).tolist(), f)
    with open(f"{outdir}/meta_{style}.json", "w", encoding="utf-8") as f:
        json.dump(
            {"style": style, "rows": int(X.shape[0]), "dims": int(X.shape[1])},
            f,
            indent=2,
        )

    print(f"[EMBED] saved to {outdir}/ (rows={X.shape[0]}, dims={X.shape[1]})")

    # 4. ✅ WandB 로깅 + Artifact 업로드
    wandb.init(
        project="wine-reco",
        job_type="embed_fit",
        name=f"embed_{style}_{time.strftime('%Y%m%d-%H%M%S')}",
    )
    wandb.config.update({"style": style, "ngram": (1, 2), "min_df": 2})
    wandb.log({"rows": X.shape[0], "dims": X.shape[1]})

    artifact = wandb.Artifact(
        name=f"tfidf-{style}",
        type="model",
        description=f"TF-IDF model for style={style}",
    )
    artifact.add_file(f"{outdir}/tfidf_{style}.pkl")
    artifact.add_file(f"{outdir}/X_{style}.npz")
    artifact.add_file(f"{outdir}/ids_{style}.json")
    artifact.add_file(f"{outdir}/meta_{style}.json")

    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="reds")
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()
    run(args.style, args.outdir)
