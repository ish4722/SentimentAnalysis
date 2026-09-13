"""Research-grade news/market prediction pipeline.

Implements the 10 requested upgrades:
1) chronological train/validation/test split
2) leakage-free TF-IDF fitting
3) publication timestamps from article pages
4) realistic backtest + transaction costs
5) buy-and-hold baseline
6) Sharpe ratio + maximum drawdown
7) technical indicators
8) FinBERT sentiment
9) walk-forward validation
10) ablation study: technical vs sentiment vs TF-IDF vs combined

Run: python research_pipeline.py --ticker META --pages 100
"""

import argparse
import json
import re
import time
from dataclasses import dataclass
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class Config:
    ticker: str = "META"
    pages: int = 100
    train_frac: float = 0.70
    validation_frac: float = 0.15
    transaction_cost: float = 0.001
    max_tfidf_features: int = 1500
    random_state: int = 42


HEADERS = {"User-Agent": "Mozilla/5.0 (research project; +https://github.com/ish4722/SentimentAnalysis)"}
TECH_FEATURES = ["return_1d", "return_5d", "return_20d", "sma_10_ratio", "sma_20_ratio", "sma_50_ratio", "rsi_14", "macd", "macd_signal", "volatility_20d", "volume_change"]
SENT_FEATURES = ["news_count", "sentiment_mean", "sentiment_max", "sentiment_min", "sentiment_std", "positive_ratio", "negative_ratio", "finbert_mean", "finbert_max", "finbert_min", "finbert_std"]


def article_timestamp(article_url):
    """Read an article's absolute publication timestamp from JSON-LD/meta tags."""
    try:
        html = requests.get(article_url, headers=HEADERS, timeout=12).text
        soup = BeautifulSoup(html, "lxml")
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                obj = json.loads(script.string or script.get_text())
                objects = obj if isinstance(obj, list) else [obj]
                for item in objects:
                    if isinstance(item, dict) and item.get("datePublished"):
                        return pd.to_datetime(item["datePublished"], utc=True, errors="coerce")
            except (json.JSONDecodeError, TypeError):
                pass
        for key in ["article:published_time", "datePublished", "pubdate"]:
            node = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
            if node and node.get("content"):
                return pd.to_datetime(node["content"], utc=True, errors="coerce")
    except requests.RequestException:
        pass
    return pd.NaT


def fetch_news(ticker, pages=100, pause=0.1):
    """Collect headlines and absolute publication times; relative page labels are rejected."""
    rows = []
    for page in range(1, pages + 1):
        url = f"https://markets.businessinsider.com/news/{ticker.lower()}-stock?p={page}&"
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
        except requests.RequestException:
            continue
        soup = BeautifulSoup(response.text, "lxml")
        for story in soup.find_all("div", class_="latest-news__story"):
            link = story.find("a", class_="news-link")
            if not link or not link.get("href"):
                continue
            headline = link.get_text(" ", strip=True)
            article_url = urljoin("https://markets.businessinsider.com", link["href"])
            published_at = article_timestamp(article_url)
            if pd.notna(published_at):
                rows.append({"published_at": published_at, "news": headline, "url": article_url})
        time.sleep(pause)
    news = pd.DataFrame(rows).drop_duplicates(subset=["published_at", "news"])
    if news.empty:
        raise RuntimeError("No absolute publication timestamps were found. Update the source or use a licensed timestamped news API.")
    news["date"] = news["published_at"].dt.tz_convert(None).dt.normalize()
    return news.sort_values("published_at").reset_index(drop=True)


def technical_features(prices):
    df = prices.copy()
    close = df["Close"]
    volume = df["Volume"]
    ret = close.pct_change()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["return_1d"] = ret
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)
    df["sma_10_ratio"] = close / close.rolling(10).mean() - 1
    df["sma_20_ratio"] = close / close.rolling(20).mean() - 1
    df["sma_50_ratio"] = close / close.rolling(50).mean() - 1
    df["rsi_14"] = 100 - 100 / (1 + rs)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["volatility_20d"] = ret.rolling(20).std()
    df["volume_change"] = volume.pct_change()
    return df


def vader_daily(news):
    sia = SentimentIntensityAnalyzer()
    scored = news.copy()
    scores = scored["news"].map(sia.polarity_scores).apply(pd.Series)
    scored["compound"] = scores["compound"].values
    scored["positive"] = scores["pos"].values
    scored["negative"] = scores["neg"].values
    daily = scored.groupby("date").agg(news_count=("news", "size"), sentiment_mean=("compound", "mean"), sentiment_max=("compound", "max"), sentiment_min=("compound", "min"), sentiment_std=("compound", "std"), positive_ratio=("positive", lambda x: (x > 0.05).mean()), negative_ratio=("negative", lambda x: (x > 0.05).mean())).fillna(0)
    return daily


def finbert_daily(news, batch_size=32):
    from transformers import pipeline
    model = pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert", top_k=None)
    rows = []
    texts = news["news"].tolist()
    for start in range(0, len(texts), batch_size):
        outputs = model(texts[start:start + batch_size], truncation=True, max_length=128)
        for scores in outputs:
            score_map = {x["label"].lower(): x["score"] for x in scores}
            rows.append(score_map.get("positive", 0.0) - score_map.get("negative", 0.0))
    scored = news[["date"]].copy()
    scored["finbert_score"] = rows
    return scored.groupby("date")["finbert_score"].agg(["mean", "max", "min", "std"]).fillna(0).add_prefix("finbert_")


def build_dataset(cfg):
    news = fetch_news(cfg.ticker, cfg.pages)
    prices = yf.Ticker(cfg.ticker).history(period="5y", auto_adjust=False)
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = technical_features(prices)
    prices["target_return"] = prices["Close"].shift(-1) / prices["Close"] - 1
    prices["target"] = (prices["target_return"] > 0).astype(int)
    text_daily = news.groupby("date")["news"].apply(lambda x: " ".join(x)).rename("news_text")
    data = prices.join(text_daily, how="left").join(vader_daily(news), how="left").join(finbert_daily(news), how="left")
    data["news_text"] = data["news_text"].fillna("")
    for col in SENT_FEATURES:
        if col in data:
            data[col] = data[col].fillna(0)
    data = data.dropna(subset=TECH_FEATURES + ["target_return"])
    return data.iloc[:-1].copy()


def split_data(data, cfg):
    n = len(data)
    a = int(n * cfg.train_frac)
    b = int(n * (cfg.train_frac + cfg.validation_frac))
    return data.iloc[:a].copy(), data.iloc[a:b].copy(), data.iloc[b:].copy()


def classification_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0)}


def backtest(test, probability, cost=0.001):
    out = test.copy()
    out["signal"] = (probability >= 0.5).astype(int)
    out["strategy_return"] = out["signal"] * out["target_return"]
    turnover = out["signal"].diff().abs().fillna(out["signal"])
    out["strategy_return"] -= turnover * cost
    out["strategy_equity"] = (1 + out["strategy_return"]).cumprod()
    out["buy_hold_equity"] = (1 + out["target_return"]).cumprod()
    excess = out["strategy_return"]
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() else 0.0
    drawdown = out["strategy_equity"] / out["strategy_equity"].cummax() - 1
    active = out.loc[out["signal"] == 1, "target_return"]
    return {"strategy_return": out["strategy_equity"].iloc[-1] - 1, "buy_hold_return": out["buy_hold_equity"].iloc[-1] - 1, "sharpe": sharpe, "max_drawdown": drawdown.min(), "trades": int(turnover.sum()), "win_rate": (active > 0).mean() if len(active) else 0.0}


def numeric_model(train, test, features, random_state=42):
    model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, random_state=random_state)
    model.fit(train[features], train["target"])
    return model.predict_proba(test[features])[:, 1]


def tfidf_model(train, test, max_features=1500, random_state=42):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train = vectorizer.fit_transform(train["news_text"])
    x_test = vectorizer.transform(test["news_text"])
    model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_state)
    model.fit(x_train, train["target"])
    return model.predict_proba(x_test)[:, 1]


def combined_model(train, test, cfg):
    vectorizer = TfidfVectorizer(max_features=cfg.max_tfidf_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_text_train = vectorizer.fit_transform(train["news_text"])
    x_text_test = vectorizer.transform(test["news_text"])
    scaler = StandardScaler(with_mean=False)
    x_num_train = scaler.fit_transform(train[TECH_FEATURES + SENT_FEATURES])
    x_num_test = scaler.transform(test[TECH_FEATURES + SENT_FEATURES])
    x_train = hstack([x_text_train, csr_matrix(x_num_train)]).tocsr()
    x_test = hstack([x_text_test, csr_matrix(x_num_test)]).tocsr()
    model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=cfg.random_state)
    model.fit(x_train, train["target"])
    return model.predict_proba(x_test)[:, 1]


def evaluate(name, train, test, probability, cfg):
    result = {"model": name, **classification_metrics(test["target"], probability), **backtest(test, probability, cfg.transaction_cost)}
    return result


def ablation(data, cfg):
    train, validation, test = split_data(data, cfg)
    results = []
    for name, features in [("technical_only", TECH_FEATURES), ("sentiment_only", SENT_FEATURES), ("technical_plus_sentiment", TECH_FEATURES + SENT_FEATURES)]:
        p_val = numeric_model(train, validation, features, cfg.random_state)
        p_test = numeric_model(pd.concat([train, validation]), test, features, cfg.random_state)
        row = evaluate(name, train, test, p_test, cfg)
        row["validation_accuracy"] = classification_metrics(validation["target"], p_val)["accuracy"]
        results.append(row)
    p_test = tfidf_model(pd.concat([train, validation]), test, cfg.max_tfidf_features, cfg.random_state)
    results.append(evaluate("tfidf_only", train, test, p_test, cfg))
    p_test = combined_model(pd.concat([train, validation]), test, cfg)
    results.append(evaluate("tfidf_plus_market_plus_sentiment", train, test, p_test, cfg))
    return pd.DataFrame(results)


def walk_forward(data, cfg):
    rows = []
    initial = max(250, int(len(data) * 0.50))
    step = max(30, int(len(data) * 0.10))
    for end in range(initial, len(data) - 30, step):
        train = data.iloc[:end].copy()
        test = data.iloc[end:min(end + step, len(data))].copy()
        p = combined_model(train, test, cfg)
        rows.append({"fold_end": str(test.index[-1].date()), **classification_metrics(test["target"], p), **backtest(test, p, cfg.transaction_cost)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="META")
    parser.add_argument("--pages", type=int, default=100)
    args = parser.parse_args()
    cfg = Config(ticker=args.ticker, pages=args.pages)
    data = build_dataset(cfg)
    train, validation, test = split_data(data, cfg)
    print(f"Rows={len(data)} | Train={len(train)} | Validation={len(validation)} | Test={len(test)}")
    print(f"Dates={data.index.min().date()} -> {data.index.max().date()}")
    print("\nABLATION STUDY")
    print(ablation(data, cfg).round(4).to_string(index=False))
    print("\nWALK-FORWARD VALIDATION")
    print(walk_forward(data, cfg).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
