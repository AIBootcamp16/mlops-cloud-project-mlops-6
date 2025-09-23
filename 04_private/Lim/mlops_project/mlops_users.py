import requests
import pandas as pd
import random
import os
import argparse

BASE_URL = "https://api.sampleapis.com/wines/reds"

def fetch_wines():
    resp = requests.get(BASE_URL)
    if resp.status_code == 200:
        return resp.json()
    else:
        raise Exception("API 요청 실패")

def generate_user_logs(wines, user_count=100, max_select=10):
    records = []
    for user_id in range(1, user_count + 1):
        select_count = random.randint(1, max_select)
        selected = random.choices(wines, k=select_count)
        for wine in selected:
            try:
                rating = float(wine["rating"]["average"])
            except:
                rating = random.uniform(2.5, 4.5)
            records.append({
                "user_id": str(user_id),
                "wine_id": str(wine.get("id", "")),
                "wine": wine.get("wine", ""),
                "winery": wine.get("winery", ""),
                "location": wine.get("location", ""),
                "rating": rating
            })
    return pd.DataFrame(records)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=2000, help="유저 수")
    parser.add_argument("--out", type=str, default="./data/user_wine_log.csv", help="저장 경로")
    args = parser.parse_args()

    wines = fetch_wines()
    df = generate_user_logs(wines, user_count=args.users)
    os.makedirs("./data", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ 데이터 생성 완료: {args.out}")
