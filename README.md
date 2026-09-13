# News2Trade — Financial News Sentiment & Stock Direction

An end-to-end NLP + quantitative-finance pipeline that studies whether financial news contains information useful for predicting next-day stock direction and generating trading signals.

## Pipeline

**Timestamped financial news → point-in-time alignment → VADER + FinBERT → TF-IDF → technical indicators → chronological validation → walk-forward evaluation → trading backtest**

## What was improved

The original notebook is retained as the historical/academic implementation. The research-grade implementation is in `research_pipeline.py`.

1. **Chronological train/validation/test split** — no random shuffling across time.
2. **Leakage-free TF-IDF** — vocabulary and IDF weights are fitted only on the training window and then applied to future data.
3. **Publication-time alignment** — article pages are queried for absolute `datePublished` metadata rather than relying on relative labels such as `2d` or `5h`.
4. **Realistic backtesting** — signals are converted into next-day returns and transaction costs are included.
5. **Buy-and-hold baseline** — strategy performance is directly compared with simply holding the stock.
6. **Financial evaluation** — cumulative return, Sharpe ratio, maximum drawdown, trade count and win rate are reported alongside ML metrics.
7. **Technical features** — momentum, multi-period returns, SMA ratios, RSI, MACD, volatility and volume changes are included.
8. **FinBERT** — `ProsusAI/finbert` provides finance-domain sentiment features in addition to VADER.
9. **Walk-forward validation** — expanding historical windows are used to test performance on strictly subsequent periods.
10. **Ablation study** — technical-only, sentiment-only, technical+sentiment, TF-IDF-only and combined models are compared.

## Models

- HistGradientBoosting for numeric technical/sentiment features
- Logistic Regression for sparse TF-IDF representations
- Combined TF-IDF + market + sentiment model
- FinBERT for finance-specific sentiment extraction

## Target

`target_return = Close[t+1] / Close[t] - 1`

`target = 1` when the next trading day's return is positive, otherwise `0`.

The strategy goes long when the predicted probability is at least 0.5 and stays in cash otherwise. A configurable transaction cost is deducted when the position changes.

## Evaluation

Classification:

- Accuracy
- Precision
- Recall
- F1

Trading:

- Strategy return
- Buy-and-hold return
- Sharpe ratio
- Maximum drawdown
- Number of trades
- Win rate

## Run

Install dependencies:

```bash
pip install -r requirements_improved.txt
```

Run the research pipeline:

```bash
python research_pipeline.py --ticker META --pages 100
```

The first run downloads the FinBERT model from Hugging Face and may take time. Increase `--pages` for more historical news, subject to the source's availability and terms.

## Important research caveats

This project is an experimental research system, not financial advice or a production trading system. News scraping can break when the source changes its HTML structure. A production implementation should use a licensed news provider with reliable publication timestamps. Results should be interpreted with out-of-sample testing, transaction costs and benchmark comparisons rather than classification accuracy alone.

## Original implementation

`Ishan_openproject_2-2.ipynb` contains the original exploratory implementation. It is preserved for comparison with the leakage-controlled research pipeline.
