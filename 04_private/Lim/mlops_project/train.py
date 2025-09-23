import os
import time
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import wandb

# -------------------------
# 📌 평가 함수
# -------------------------
def precision_at_k(preds, actual, k=5):
    if len(preds) == 0 or len(actual) == 0:
        return 0.0
    return len(set(preds[:k]) & set(actual)) / k

def recall_at_k(preds, actual, k=5):
    if len(preds) == 0 or len(actual) == 0:
        return 0.0
    return len(set(preds[:k]) & set(actual)) / len(actual)


# -------------------------
# 📌 데이터 로드
# -------------------------
DATA_PATH = "./data/user_wine_log.csv"  # ✅ 상대 경로
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

df['wine_id'] = df['wine_id'].astype("category")
df['user_id'] = df['user_id'].astype("category")

df['user_idx'] = df['user_id'].cat.codes
df['wine_idx'] = df['wine_id'].cat.codes

user_item_matrix = csr_matrix(
    (df['rating'], (df['user_idx'], df['wine_idx']))
)


# -------------------------
# 📌 W&B Init
# -------------------------
user_count = df['user_id'].nunique()
item_count = df['wine_id'].nunique()

wandb.init(
    project="wine-recommender",
    name=f"train_{user_count}users(10)"
)

wandb.config["user_count"] = user_count
wandb.config["item_count"] = item_count

# -------------------------
# 📌 모델 학습
# -------------------------
model = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

start = time.time()
model.fit(user_item_matrix)
train_time = time.time() - start

# ✅ user_count, item_count도 log로 기록
wandb.log({
    "train_time": float(train_time),
    "user_count": user_count,
    "item_count": item_count
})



# -------------------------
# 📌 평가 (Precision@5, Recall@5)
# -------------------------
precisions, recalls = [], []

for user_idx in df['user_idx'].unique():
    recs = model.recommend(user_idx, user_item_matrix[user_idx], N=5)

    preds = []
    for r in recs:
        wine_idx = int(r[0]) if isinstance(r, (list, tuple, np.ndarray)) else int(r)
        preds.append(df['wine_id'].cat.categories[wine_idx])

    actual = list(df[df['user_idx'] == user_idx]['wine_id'])

    precisions.append(precision_at_k(preds, actual, k=5))
    recalls.append(recall_at_k(preds, actual, k=5))

wandb.log({
    "precision@5": float(np.mean(precisions)),
    "recall@5": float(np.mean(recalls))
})

print("✅ Training finished")
print(f"user_count={user_count}, item_count={item_count}")
print(f"train_time: {train_time:.4f} sec")
print(f"Precision@5: {np.mean(precisions):.4f}")
print(f"Recall@5: {np.mean(recalls):.4f}")
