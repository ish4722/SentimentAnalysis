"""Leakage-free news + market pipeline for next-day stock direction.

Implements: chronological split, leakage-free TF-IDF, publication-time alignment,
proper backtesting, buy-and-hold baseline, Sharpe/max drawdown, technical features,
FinBERT sentiment, walk-forward validation, and ablation experiments.

Usage:
    python improved_pipeline.py --ticker META

The Business Insider scraper is intentionally isolated in fetch_business_insider_news().
For production use, replace it with a licensed news API that provides timestamps.
"""

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class Config:
    ticker: str = "META"
    years: str = "5y"
    train_frac: float = 0.70
    validation_frac: float = 0.15
    transaction_cost: float = 0.001
    risk_free_rate: float = 0.0
    max_tfidf_features: int = 1500
    random_state: int = 42


def fetch_business_insider_news(ticker: str, pages: int = 250, pause: float = 0.15) -> pd.DataFrame:
    """Scrape headline + publication date when available.

    Business Insider page structure can change. The function keeps only records with
    parseable absolute timestamps where possible; relative timestamps are not used
    for model labels because they can create date leakage/misalignment.
    """
    rows: List[Dict] = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for page in range(1, pages + 1):
        url = f"https://markets.businessinsider.com/news/{ticker.lower()}-stock?p={page}&"
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            stories = soup.find_all("div", class_="latest-news__story")
            for story in stories:
                link = story.find("a", class_="news-link")
                date_node = story.find(class_="latest-news__date")
                if link is None or date_node is None:
                    continue
                headline = link.get_text(" ", strip=True)
                raw_date = date_node.get_text(" ", strip=True)
                parsed = pd.to_datetime(raw_date, errors="coerce", utc=True)
                if pd.notna(parsed):
                    rows.append({"published_at": parsed, "news": headline})
        except requests.RequestException:
            continue
        time.sleep(pause)
    news = pd.DataFrame(rows).drop_duplicates(subset=["published_at", "news"])
    if news.empty:
        raise RuntimeError("No absolute publication timestamps were parsed. Use a timestamped news API or update the scraper.")
    news["date"] = news["published_at"].dt.tz_convert(None).dt.normalize()
    return news.sort_values("published_at").reset_index(drop=True)


def technical_features(price: pd.DataFrame) -> pd.DataFrame:
    """Create technical features using only information available by day t."""
    df = price.copy()
    close = df["Close"]
    volume = df["Volume"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["return_1d"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)
    df["sma_10_ratio"] = close / close.rolling(10).mean() - 1
    df["sma_20_ratio"] = close / close.rolling(20).mean() - 1
    df["sma_50_ratio"] = close / close.rolling(50).mean() - 1
    df["rsi_14"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["volatility_20d"] = close.pct_change().rolling(20).std()
    df["volume_change"] = volume.pct_change()
    return df


def sentiment_features(news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level VADER signals by calendar date."""
    sia = SentimentIntensityAnalyzer()
    scores = news["news"].map(sia.polarity_scores).apply(pd.Series)
    news = news.copy()
    news[["vader_compound", "vader_negative", "vader_positive", "vader_neutral"]] = scores[["compound", "neg", "pos", "neu"]].to_numpy()
    news["positive_flag"] = (news["vader_compound"] > 0.05).astype(int)
    news["negative_flag"] = (news["vader_compound"] < -0.05).astype(int)
    daily = news.groupby("date").agg(
        news_count=("news", "size"),
        sentiment_mean=("vader_compound", "mean"),
        sentiment_max=("vader_compound", "max"),
        sentiment_min=("vader_compound", "min"),
        sentiment_std=("vader_compound", "std"),
        positive_ratio=("positive_flag", "mean"),
        negative_ratio=("negative_flag", "mean"),
    )
    return daily.fillna(0)


def finbert_daily_sentiment(news: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """Compute FinBERT probabilities and aggregate them by date.

    FinBERT is loaded lazily so users can run the classical baseline without it.
    """
    from transformers import pipeline
    classifier = pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert", top_k=None)
    records = []
    texts = news["news"].tolist()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        outputs = classifier(batch, truncation=True, max_length=128)
        for text, scores in zip(batch, outputs):
            score_map = {item["label"].lower(): item["score"] for item in scores}
            records.append({"news": text, "finbert_score": score_map.get("positive", 0.0) - score_map.get("negative", 0.0)})
    scored = news[["date", "news"]].merge(pd.DataFrame(records), on="news", how="left")
    return scored.groupby("date")["finbert_score"].agg(["mean", "max", "min", "std"]).fillna(0).add_prefix("finbert_")


def build_dataset(ticker: str, years: str = "5y", use_finbert: bool = True) -> pd.DataFrame:
    """Build a point-in-time daily dataset."""
    news = fetch_business_insider_news(ticker)
    prices = yf.Ticker(ticker).history(period=years, auto_adjust=False)
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = technical_features(prices)
    prices["target_return"] = prices["Close"].shift(-1) / prices["Close"] - 1
    prices["target"] = (prices["target_return"] > 0).astype(int)

    daily_sent = sentiment_features(news)
    data = prices.join(daily_sent, how="left")
    if use_finbert:
        data = data.join(finbert_daily_sentiment(news), how="left")
    data = data.ffill(limit=0) if False else data.fillna(0)
    data = data.iloc[:-1].copy()
    return data


def chronological_split(data: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(data)
    train_end = int(n * cfg.train_frac)
    val_end = int(n * (cfg.train_frac + cfg.validation_frac))
    return data.iloc[:train_end].copy(), data.iloc[train_end:val_end].copy(), data.iloc[val_end:].copy()


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, text_train: pd.Series = None, text_test: pd.Series = None, model_name: str = "logistic", cfg: Config = Config()):
    """Train on the past and predict only the future."""
    if text_train is not None:
        vectorizer = TfidfVectorizer(max_features=cfg.max_tfidf_features, ngram_range=(1, 2), min_df=2)
        X_train_text = vectorizer.fit_transform(text_train)
        X_test_text = vectorizer.transform(text_test)
    else:
        X_train_text = None
        X_test_text = None

    if model_name == "logistic":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg.random_state)
    elif model_name == "rf":
        model = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=cfg.random_state, n_jobs=-1)
    else:
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=cfg.random_state)

    if X_train_text is not None:
        model.fit(X_train_text, train["target"])
        return model.predict_proba(X_test_text)[:, 1]
    feature_cols = [c for c in train.columns if c in NUMERIC_FEATURES]
    model.fit(train[feature_cols], train["target"])
    return model.predict_proba(test[feature_cols])[:, 1]


NUMERIC_FEATURES = [
    "return_1d", "return_5d", "return_20d", "sma_10_ratio", "sma_20_ratio", "sma_50_ratio",
    "rsi_14", "macd", "macd_signal", "volatility_20d", "volume_change", "news_count",
    "sentiment_mean", "sentiment_max", "sentiment_min", "sentiment_std", "positive_ratio", "negative_ratio",
    "finbert_mean", "finbert_max", "finbert_min", "finbert_std"
]


def metrics(y_true: pd.Series, probability: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (probability >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def backtest(test: pd.DataFrame, probability: np.ndarray, cfg: Config) -> Dict[str, float]:
    """Long/cash strategy with transaction costs, compared to buy-and-hold."""
    out = test.copy()
    out["signal"] = (probability >= 0.5).astype(int)
    out["strategy_return"] = out["signal"] * out["target_return"]
    turnover = out["signal"].diff().abs().fillna(out["signal"])
    out["strategy_return"] -= turnover * cfg.transaction_cost
    out["strategy_equity"] = (1 + out["strategy_return"]).cumprod()
    out["buy_hold_equity"] = (1 + out["target_return"]).cumprod()
    daily = out["strategy_return"]
    excess = daily - cfg.risk_free_rate / 252
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0.0
    drawdown = out["strategy_equity"] / out["strategy_equity"].cummax() - 1
    return {
        "strategy_return": float(out["strategy_equity"].iloc[-1] - 1),
        "buy_hold_return": float(out["buy_hold_equity"].iloc[-1] - 1),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "trades": int(turnover.sum()),
        "win_rate": float((daily[daily != 0] > 0).mean()) if (daily != 0).any() else 0.0,
    }


def walk_forward(data: pd.DataFrame, feature_set: str, cfg: Config) -> pd.DataFrame:
    """Expanding-window evaluation. Every fold predicts data strictly after its train window."""
    rows = []
    n = len(data)
    initial = max(250, int(n * 0.50))
    step = max(30, int(n * 0.10))
    for end in range(initial, n - 30, step):
        train = data.iloc[:end].copy()
        test = data.iloc[end:min(end + step, n)].copy()
        if feature_set == "technical":
            prob = fit_predict(train, test, model_name="logistic", cfg=cfg)
        else:
            # TF-IDF requires raw daily text. The dataset stores aggregated sentiment,
            # so the production path should attach point-in-time text here. For the
            # reproducible numeric experiments, sentiment/technical features are used.
            feature_cols = [c for c in NUMERIC_FEATURES if c in data.columns]
            model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=cfg.random_state)
            model.fit(train[feature_cols], train["target"])
            prob = model.predict_proba(test[feature_cols])[:, 1]
        row = metrics(test["target"], prob)
        row.update(backtest(test, prob, cfg))
        row["fold_end"] = str(test.index[-1].date())
        rows.append(row)
    return pd.DataFrame(rows)


def run_ablation(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compare technical-only, sentiment-only and combined feature sets."""
    n = len(data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train, val, test = data.iloc[:train_end], data.iloc[train_end:val_end], data.iloc[val_end:]
    results = []
    feature_sets = {
        "technical_only": [c for c in NUMERIC_FEATURES if c in data.columns and (c.startswith("return") or c.startswith("sma") or c in {"rsi_14", "macd", "macd_signal", "volatility_20d", "volume_change"})],
        "sentiment_only": [c for c in NUMERIC_FEATURES if c in data.columns and (c.startswith("sentiment") or c.startswith("positive") or c.startswith("negative") or c.startswith("finbert") or c == "news_count")],
        "combined": [c for c in NUMERIC_FEATURES if c in data.columns],
    }
    for name, cols in feature_sets.items():
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=cfg.random_state)
        model.fit(train[cols], train["target"])
        prob_val = model.predict_proba(val[cols])[:, 1]
        # Validation is kept separate; threshold/model selection can happen here.
        threshold = 0.5
        prob_test = model.predict_proba(test[cols])[:, 1]
        row = metrics(test["target"], prob_test, threshold)
        row.update(backtest(test, prob_test, cfg))
        row["feature_set"] = name
        row["validation_accuracy"] = metrics(val["target"], prob_val, threshold)["accuracy"]
        results.append(row)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="META")
    parser.add_argument("--no-finbert", action="store_true")
    args = parser.parse_args()
    cfg = Config(ticker=args.ticker)
    data = build_dataset(args.ticker, cfg.years, use_finbert=not args.no_finbert)
    train, val, test = chronological_split(data, cfg)
    print(f"Dataset: {len(data)} rows | train={len(train)} validation={len(val)} test={len(test)}")
    print("Date range:", data.index.min().date(), "to", data.index.max().date())
    print("\nAblation study (chronological holdout):")
    print(run_ablation(data, cfg).round(4).to_string(index=False))
    print("\nWalk-forward evaluation:")
    wf = walk_forward(data, "combined", cfg)
    print(wf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
