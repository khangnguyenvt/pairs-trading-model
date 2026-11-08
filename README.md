# Friction-Aware Pairs Trading Model (V5)

An intraday statistical-arbitrage strategy on 14 cointegrated S&P 500 pairs. It uses a
Kalman-filtered hedge ratio and tiered Z-score entries, and it is backtested on 1-minute
bars. Every fill is priced against a **Level-3 order book**, so all reported performance is
**net of commissions and slippage**.

## Strategy

### Universe
14 pairs chosen through cointegration screening (Engle-Granger / Johansen plus
spurious-correlation tests on 2022–2023 data):

```
GEN/ADBE  CTAS/HWM  JCI/DE    FSLR/CSGP  EL/MKC    JBHT/ODFL  EW/CI
KDP/CLX   LMT/CPRT  PGR/NTRS  AMP/AFL    DASH/GPC  CDW/AZO    MCO/BRO
```

### Signal
| Component | Rule |
|---|---|
| Hedge ratio | 1D Kalman filter on log prices (δ = 1e-5, R = 1e-3). The **prior** β is used at each bar, so there is no look-ahead. |
| Spread / Z | `spread = log(A) − β·log(B)`, Z-score over a 1,950-bar (5-day) EWM window |
| Tiered entry | \|Z\| ≥ 1.75 → 25% of leg · \|Z\| ≥ 2.25 → 50% · \|Z\| ≥ 3.0 → 100% |
| Exit | \|Z\| ≤ 0.5, or a 5-day time stop |
| Entry filters | spread more than 150 bps from its mean · no 5-day leg move above 7% · rolling correlation ≥ 0.10 · a mean-crossing within the last 10 days |
| Pair weighting | ADF re-scored every 30 trading days on a ~6-month window: p < 0.01 → 100%, p < 0.05 → 50%, otherwise 0% |
| Execution delay | Signals execute on the bar **after** they are generated |

### Execution (friction model)
| Component | Rule |
|---|---|
| Fills | VWAP walk through L1 → L2 → L3 of the order book, plus a 5 bps penalty for any size left unfilled beyond L3 |
| Commission | $0.005 per share |
| Liquidity cap | Order size capped at 50% of L1 depth (both legs scaled together to keep β) |
| Min rebalance | Rebalances under $500 are skipped |
| Spread gate | New entries skipped when the round-trip bid-ask cost is above 80 bps |
| Sizing | $100k equity, $10k per leg, dollar-neutral via β |

## Results

These metrics are computed from the daily net-return series in [`results/`](results/)
produced by this model. Annualized return is CAGR; Sharpe = annualized return /
annualized volatility, with no risk-free rate.

| Metric (net of friction) | In-sample 2022–2024 | Out-of-sample 2024–2026 |
|---|:-:|:-:|
| Annualized return | 0.55% | 2.04% |
| Annualized volatility | 1.34% | 1.58% |
| Sharpe ratio | 0.41 | **1.30** |
| Max drawdown | −2.62% | −1.17% |

In a capacity sweep, net Sharpe falls below 1.0 at about **$17.5k per leg** (~$490k
notional), the point where order-book depth runs out.

## Usage

```bash
pip install -r requirements.txt
export PAIRS_DATA_DIR=/path/to/data      # defaults to ./data
python pairs_trading_model.py 2024-2026  # or: 2022-2024 | 2022 | 2023
```

The script prints a gross-vs-net report, a friction breakdown, per-pair stats and the
kill-switch diagnostic. It writes the daily net returns to `results/pairs_returns_<mode>.csv`.

### Data layout
Market data is not included because of its size and licensing. The model expects:

```
$PAIRS_DATA_DIR/
├── ohlc-2022/ohlc-2022-01/ … ohlc-2022-12/   # 2022 is split into monthly subfolders
├── ohlc-2023/  ohlc-2024/  ohlc-2025/  ohlc-2026/
└── order_book-<year>/order_book-<year>-<MM>/
```

- **OHLC files** (`<TICKER>_*.csv`): 1-minute bars with columns `window_start`
  (nanoseconds since epoch, UTC), `close`, `volume`.
- **Order-book files** (`<TICKER>_*.csv`): `timestamp` (e.g. `2023-01-03 09:30:00 EST`)
  and `l{1,2,3}_{bid,ask}_{px,sz}`.

## Repository structure

```
pairs_trading_model.py   # the complete model: signals, Kalman, ADF, L3 slippage engine, report
results/                 # daily net returns, in-sample and out-of-sample
requirements.txt
```
