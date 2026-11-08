import pandas as pd
import numpy as np
import glob
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------
# PAIRS TRADING MODEL (V5) — Friction-Aware Kalman Z-Score Strategy
#
# Signal:    1D Kalman hedge ratio, tiered Z-score entries (1.75/2.25/3.0),
#            correlation-decay filter, 30-day rolling ADF pair re-weighting
# Execution: Deterministic L3 order-book VWAP fills, $0.005/share commission,
#            liquidity cap (50% of L1), $500 min rebalance, 80 bps spread gate
#
# Usage:     python pairs_trading_model.py [2022-2024|2024-2026|2022|2023]
# Data:      set PAIRS_DATA_DIR (default: ./data) — see README.md
# -----------------------------------------------------------------------

DATA_ROOT = os.environ.get(
    "PAIRS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

PAIRS = [
    ("GEN", "ADBE"), ("CTAS", "HWM"), ("JCI", "DE"),
    ("FSLR", "CSGP"), ("EL", "MKC"), ("JBHT", "ODFL"),
    ("EW", "CI"), ("KDP", "CLX"),
    ("LMT", "CPRT"), ("PGR", "NTRS"), ("AMP", "AFL"),
    ("DASH", "GPC"), ("CDW", "AZO"), ("MCO", "BRO")
]

INITIAL_EQUITY = 100000.0
ALLOCATION_PER_LEG = 10000.0
COMMISSION_PER_SHARE = 0.005         # $0.005 flat per share traded
FALLBACK_SLIPPAGE_BPS = 0.0005       # 5 bps penalty for unfilled beyond L3
STATIC_SLIPPAGE_BPS = 0.0005         # Original v3 static slippage for gross calc

# V5 Microstructure Execution Optimization Parameters
LIQUIDITY_CAP_FRACTION = 0.5         # Change 1: Cap execution at 50% of L1 depth
MIN_REBALANCE_DOLLARS = 500.0        # Change 2: Suppress rebalances under $500
MAX_ENTRY_SPREAD_BPS = 80.0          # Change 3: Skip entries if spread cost > 80 bps
EXEC_START_HOUR = 0                  # Change 4: Full Day trading (00:00 ET)
EXEC_START_MINUTE = 0                # Change 4: Full Day trading (00)
EXEC_END_HOUR = 24                   # Change 4: Full Day trading (24:00 ET)
EXEC_END_MINUTE = 0                  # Change 4: Full Day trading (00)


# =====================================================================
# PRESERVED V3 CORE — Signal generation, Kalman, ADF (unchanged)
# =====================================================================

def kalman_beta(log_PA_vals, log_PB_vals, delta=1e-5, R=1e-3):
    """
    Estimates the dynamic hedge ratio beta using a 1D Kalman Filter.

    Unlike EWM (fixed decay), the Kalman gain K adapts:
    - When beta is stable: P small → K small → filter ignores noise
    - When beta shifts: P grows → K increases → filter tracks rapidly

    Parameters:
        delta: Process noise variance (how fast beta drifts)
        R:     Observation noise variance (measurement uncertainty)
    """
    n = len(log_PA_vals)
    beta = np.zeros(n)
    P = 1.0           # Initial error covariance (high uncertainty)
    beta_hat = 0.0    # Initial beta estimate

    for i in range(1, n):
        # --- Predict step ---
        P_pred = P + delta  # Error covariance grows by process noise

        # CRITICAL: Store the PRIOR beta (before seeing bar i).
        # Using the posterior here was the look-ahead bias — beta[i] was
        # tuned to explain price[i], then used to compute spread[i],
        # making the spread artificially stable and inflating Sharpe.
        beta[i] = beta_hat

        # --- Update step (uses bar i to improve estimate for bar i+1) ---
        x = log_PB_vals[i]  # Regressor (log price B)
        y = log_PA_vals[i]  # Observation (log price A)

        if abs(x) < 1e-12:
            continue

        # Innovation: difference between measured and predicted
        y_hat = beta_hat * x
        innovation = y - y_hat

        # Innovation covariance
        S = x * P_pred * x + R

        # Kalman gain: adaptive weight between prior and new data
        K = P_pred * x / S

        # State update (for use at bar i+1, NOT bar i)
        beta_hat = beta_hat + K * innovation

        # Covariance update (Joseph form for numerical stability)
        P = (1 - K * x) * P_pred

    return beta


def score_pair_adf(spread_series):
    """
    Returns ADF p-value for the current spread window.
    Used to dynamically weight or eject pairs every 30 trading days.
    """
    from statsmodels.tsa.stattools import adfuller
    try:
        clean = spread_series.dropna()
        if len(clean) < 50:
            return 1.0
        result = adfuller(clean, maxlag=5, autolag='AIC')
        return result[1]  # p-value
    except:
        return 1.0  # Treat as non-cointegrated on failure


def get_adf_weight(p_value):
    """
    Maps ADF p-value to allocation weight.
    p < 0.01  → Active (100%)
    p < 0.05  → Watch  (50%)
    p >= 0.05 → Inactive (0%)
    """
    if p_value < 0.01:
        return 1.0
    elif p_value < 0.05:
        return 0.5
    else:
        return 0.0


def generate_signals_tiered(z_scores, spreads, ma_spreads, is_trending,
                            recent_cross, exit_z, min_div, max_hold,
                            adf_weights):
    """
    Tiered entry sizing based on Z-score magnitude:
      |Z| >= 3.00 → signal = ±1.0  (100% leg)
      |Z| >= 2.25 → signal = ±0.5  (50% leg)
      |Z| >= 1.75 → signal = ±0.25 (25% leg)

    The adf_weights array scales allocation dynamically per-bar based on
    the most recent ADF re-scoring result.
    """
    n = len(z_scores)
    signals = np.zeros(n)
    pos = 0.0
    hold_time = 0

    for i in range(n):
        if np.isnan(z_scores[i]):
            continue

        z = z_scores[i]
        spr_diff = abs(spreads[i] - ma_spreads[i])
        w = adf_weights[i]  # ADF universe weight (0.0, 0.5, or 1.0)

        if pos == 0.0:
            # Only enter if filters pass and pair is ADF-active
            if spr_diff > min_div and not is_trending[i] and recent_cross[i] and w > 0:
                if z > 3.0:
                    pos = -1.0 * w
                    hold_time = 1
                elif z > 2.25:
                    pos = -0.5 * w
                    hold_time = 1
                elif z > 1.75:
                    pos = -0.25 * w
                    hold_time = 1
                elif z < -3.0:
                    pos = 1.0 * w
                    hold_time = 1
                elif z < -2.25:
                    pos = 0.5 * w
                    hold_time = 1
                elif z < -1.75:
                    pos = 0.25 * w
                    hold_time = 1
        else:
            hold_time += 1
            if pos < 0:  # Short spread
                if z <= exit_z or hold_time > max_hold:
                    pos = 0.0
            elif pos > 0:  # Long spread
                if z >= -exit_z or hold_time > max_hold:
                    pos = 0.0

        signals[i] = pos
    return signals


# =====================================================================
# NEW: Data Loaders with Timezone Alignment
# =====================================================================

def load_ticker_data(ticker, data_dirs):
    """
    Load OHLCV data for a single ticker across multiple directories.
    Converts window_start (nanoseconds UTC) → America/New_York.
    Returns DataFrame indexed by tz-aware datetime with 'close' & 'volume'.
    """
    dfs = []
    for d in data_dirs:
        pattern = os.path.join(d, f"{ticker}_*.csv")
        files = sorted(glob.glob(pattern))
        for f in files:
            try:
                df = pd.read_csv(f, usecols=['window_start', 'close', 'volume'])
                dfs.append(df)
            except Exception:
                pass

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined['window_start'] = pd.to_datetime(combined['window_start'], unit='ns')
    # Localize to UTC then convert to New York
    combined['window_start'] = (
        combined['window_start']
        .dt.tz_localize('UTC')
        .dt.tz_convert('America/New_York')
    )
    combined = combined.set_index('window_start')
    combined = combined[~combined.index.duplicated(keep='last')]
    combined.sort_index(inplace=True)
    return combined


def load_ticker_orderbook(ticker, ob_dirs):
    """
    Load Level 3 Order Book data for a ticker across all order book dirs.
    Parses timestamp strings (e.g. '2022-01-03 04:00:00 EST') into
    America/New_York-localized DateTimeIndex.
    """
    dfs = []
    for d in ob_dirs:
        pattern = os.path.join(d, f"{ticker}_*.csv")
        files = sorted(glob.glob(pattern))
        for f in files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception:
                pass

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Parse timestamp: strip timezone suffix, parse, then localize
    # Format: "2022-01-03 04:00:00 EST" or "2022-06-15 10:30:00 EDT"
    combined['timestamp'] = (
        combined['timestamp']
        .str.replace(r'\s+(EST|EDT)$', '', regex=True)
    )
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    combined['timestamp'] = combined['timestamp'].dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT')
    combined = combined.set_index('timestamp')
    combined = combined[~combined.index.duplicated(keep='last')]
    combined.sort_index(inplace=True)

    # Keep only the order book columns we need
    ob_cols = [
        'l1_bid_px', 'l1_bid_sz', 'l1_ask_px', 'l1_ask_sz',
        'l2_bid_px', 'l2_bid_sz', 'l2_ask_px', 'l2_ask_sz',
        'l3_bid_px', 'l3_bid_sz', 'l3_ask_px', 'l3_ask_sz'
    ]
    return combined[[c for c in ob_cols if c in combined.columns]]


# =====================================================================
# NEW: Deterministic Slippage Engine
# =====================================================================

def calculate_execution_price(trade_size_shares, is_buy,
                              l1_px, l1_sz, l2_px, l2_sz, l3_px, l3_sz):
    """
    Walk the order book to compute the VWAP execution price for a market order.

    For a BUY: walks up the ask side (L1 → L2 → L3).
    For a SELL: walks down the bid side (L1 → L2 → L3).

    If trade_size exceeds total L3 liquidity, applies a 5 bps penalty
    to the remaining unfilled volume priced at L3.

    Returns:
        (vwap_price, total_filled_shares)
    """
    if trade_size_shares <= 0 or np.isnan(trade_size_shares):
        return (l1_px, 0.0)

    remaining = trade_size_shares
    cost = 0.0
    filled = 0.0

    levels = [(l1_px, l1_sz), (l2_px, l2_sz), (l3_px, l3_sz)]

    for px, sz in levels:
        if remaining <= 0:
            break
        if np.isnan(px) or np.isnan(sz) or sz <= 0:
            continue
        fill_qty = min(remaining, sz)
        cost += fill_qty * px
        filled += fill_qty
        remaining -= fill_qty

    # Fallback: if still unfilled, apply severe slippage penalty
    if remaining > 0:
        # Price the remainder at L3 + 5 bps penalty
        fallback_px = l3_px * (1 + FALLBACK_SLIPPAGE_BPS) if is_buy else l3_px * (1 - FALLBACK_SLIPPAGE_BPS)
        if np.isnan(fallback_px):
            fallback_px = l1_px * (1 + FALLBACK_SLIPPAGE_BPS) if is_buy else l1_px * (1 - FALLBACK_SLIPPAGE_BPS)
        cost += remaining * fallback_px
        filled += remaining

    vwap = cost / filled if filled > 0 else l1_px
    return (vwap, filled)


def compute_slippage_for_pair(delta_pos_dollars, prices, ob_data, leg_label):
    """
    For each bar where position changes, compute the actual execution price
    via the order book slippage engine.

    Parameters:
        delta_pos_dollars: Series — change in dollar position (signed)
        prices:           Series — close prices for the ticker
        ob_data:          DataFrame — order book columns aligned to same index
        leg_label:        str — 'A' or 'B' (for debugging)

    Returns:
        slippage_cost:  Series — per-bar dollar cost from spread/slippage
        commission:     Series — per-bar dollar commission
    """
    slippage_cost = pd.Series(0.0, index=delta_pos_dollars.index)
    commission = pd.Series(0.0, index=delta_pos_dollars.index)

    # Only process bars where position actually changes
    trade_mask = delta_pos_dollars.abs() > 1e-6

    if trade_mask.sum() == 0:
        return slippage_cost, commission

    for idx in delta_pos_dollars[trade_mask].index:
        delta_d = delta_pos_dollars.loc[idx]
        price = prices.loc[idx]

        if np.isnan(price) or price <= 0:
            continue

        # Convert dollar change to shares
        shares = abs(delta_d) / price
        is_buy = (delta_d > 0)

        # Get order book snapshot for this bar
        if idx not in ob_data.index:
            # No order book data — use midprice as execution, apply static slippage
            slippage_cost.loc[idx] = abs(delta_d) * STATIC_SLIPPAGE_BPS
            commission.loc[idx] = shares * COMMISSION_PER_SHARE
            continue

        ob_row = ob_data.loc[idx]

        if is_buy:
            l1_px = ob_row.get('l1_ask_px', price)
            l1_sz = ob_row.get('l1_ask_sz', 0)
            l2_px = ob_row.get('l2_ask_px', price)
            l2_sz = ob_row.get('l2_ask_sz', 0)
            l3_px = ob_row.get('l3_ask_px', price)
            l3_sz = ob_row.get('l3_ask_sz', 0)
        else:
            l1_px = ob_row.get('l1_bid_px', price)
            l1_sz = ob_row.get('l1_bid_sz', 0)
            l2_px = ob_row.get('l2_bid_px', price)
            l2_sz = ob_row.get('l2_bid_sz', 0)
            l3_px = ob_row.get('l3_bid_px', price)
            l3_sz = ob_row.get('l3_bid_sz', 0)

        exec_price, filled = calculate_execution_price(
            shares, is_buy, l1_px, l1_sz, l2_px, l2_sz, l3_px, l3_sz
        )

        # Midprice for slippage measurement
        mid_bid = ob_row.get('l1_bid_px', price)
        mid_ask = ob_row.get('l1_ask_px', price)
        midprice = (mid_bid + mid_ask) / 2.0 if not (np.isnan(mid_bid) or np.isnan(mid_ask)) else price

        # Slippage = deviation of execution price from midprice × shares
        if is_buy:
            slip = max(0.0, (exec_price - midprice) * filled)
        else:
            slip = max(0.0, (midprice - exec_price) * filled)

        slippage_cost.loc[idx] = slip
        commission.loc[idx] = filled * COMMISSION_PER_SHARE

    return slippage_cost, commission


# =====================================================================
# Directory helpers
# =====================================================================

def get_ohlc_dirs(mode):
    """Return list of OHLCV directories for the given mode."""
    if mode == "2022-2024":
        # 2022 has monthly subdirs under the main dir
        dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "ohlc-2022", "ohlc-2022-*")))
        dirs.extend([
            os.path.join(DATA_ROOT, "ohlc-2023"),
            os.path.join(DATA_ROOT, "ohlc-2024")
        ])
    elif mode == "2022":
        dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "ohlc-2022", "ohlc-2022-*")))
    elif mode == "2023":
        dirs = [os.path.join(DATA_ROOT, "ohlc-2023")]
    elif mode == "2024-2026":
        dirs = [
            os.path.join(DATA_ROOT, "ohlc-2024"),
            os.path.join(DATA_ROOT, "ohlc-2025"),
            os.path.join(DATA_ROOT, "ohlc-2026")
        ]
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
    return dirs


def get_orderbook_dirs(mode):
    """Return list of order book directories (monthly subdirs) for the given mode."""
    ob_dirs = []
    if mode == "2022-2024":
        years = [2022, 2023, 2024]
    elif mode == "2022":
        years = [2022]
    elif mode == "2023":
        years = [2023]
    elif mode == "2024-2026":
        years = [2024, 2025, 2026]
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    for year in years:
        base = os.path.join(DATA_ROOT, f"order_book-{year}")
        monthly = sorted(glob.glob(os.path.join(base, f"order_book-{year}-*")))
        ob_dirs.extend(monthly)
    return ob_dirs


# =====================================================================
# Main Backtest Engine (V4 — Microstructure Upgrade)
# =====================================================================

def run_v4_backtest(data_dirs, ob_dirs, corr_floor=0.10, return_filter=0.07):
    window_size = 1950   # 5 days
    exit_z = 0.5
    min_div = 0.0150     # 150 bps
    max_hold = 1950      # 5 days

    # ADF re-scoring interval: 30 trading days ≈ 11,700 bars
    ADF_RESCORE_INTERVAL = 11700
    ADF_LOOKBACK = 117000  # ~6 months of minute bars for ADF window

    portfolio_gross_pnl = None
    portfolio_net_pnl = None

    all_gross_trade_pnls = []
    all_net_trade_pnls = []

    total_commission_paid = 0.0
    total_slippage_paid = 0.0
    total_gross_profit = 0.0

    pair_metrics = {}

    # --- Q1 FIX: Portfolio-level rolling Sharpe kill switch ---
    SHARPE_WINDOW = 7800   # 30 trading days in minutes
    SHARPE_HALT_THRESHOLD = 0.0
    SHARPE_RESUME_THRESHOLD = 0.3

    for tickerA, tickerB in PAIRS:
        pair_label = f"{tickerA}/{tickerB}"
        print(f"  Processing {pair_label}...", end=" ", flush=True)

        # Load OHLCV data
        dfA = load_ticker_data(tickerA, data_dirs)
        dfB = load_ticker_data(tickerB, data_dirs)

        if dfA.empty or dfB.empty:
            pair_metrics[pair_label] = {
                "gross_sharpe": 0, "net_sharpe": 0,
                "gross_pnl": 0, "net_pnl": 0,
                "trades": 0, "status": "NO DATA"
            }
            print("SKIP (no OHLCV data)")
            continue

        # Load order book data
        obA = load_ticker_orderbook(tickerA, ob_dirs)
        obB = load_ticker_orderbook(tickerB, ob_dirs)

        if obA.empty or obB.empty:
            pair_metrics[pair_label] = {
                "gross_sharpe": 0, "net_sharpe": 0,
                "gross_pnl": 0, "net_pnl": 0,
                "trades": 0, "status": "NO OB DATA"
            }
            print("SKIP (no order book data)")
            continue

        # Merge OHLCV for both legs
        df = pd.concat([dfA['close'], dfB['close'], dfA['volume'], dfB['volume']],
                       axis=1, keys=['A', 'B', 'vol_A', 'vol_B'])
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        price_A = df['A']
        price_B = df['B']
        log_PA = np.log(price_A)
        log_PB = np.log(price_B)

        # --- Q3: Kalman Filter for beta ---
        beta_arr = kalman_beta(log_PA.values, log_PB.values, delta=1e-5, R=1e-3)
        beta = pd.Series(beta_arr, index=log_PA.index)

        spread = log_PA - beta * log_PB
        roll_mean_spread = spread.ewm(span=window_size).mean()
        roll_std_spread = spread.ewm(span=window_size).std()
        z_score = (spread - roll_mean_spread) / (roll_std_spread + 1e-8)

        # --- Q2-B: Correlation decay filter (floor = 0.10) ---
        roll_corr = log_PA.rolling(window=window_size).corr(log_PB)
        is_uncorrelated = roll_corr < corr_floor

        # Hard momentum trend filter
        ret_A_5d = log_PA.diff(window_size).abs()
        ret_B_5d = log_PB.diff(window_size).abs()
        is_trending_ret = (ret_A_5d > return_filter) | (ret_B_5d > return_filter)

        is_trending = is_uncorrelated | is_trending_ret
        is_trending = is_trending.fillna(False)

        # Cointegration decay detection (10-day cross check)
        crosses = ((spread - roll_mean_spread) * (spread.shift(1) - roll_mean_spread.shift(1))) < 0
        recent_cross = crosses.rolling(window_size * 2).max() > 0
        recent_cross = recent_cross.fillna(True)

        # --- Q5: Rolling ADF Universe Scoring ---
        n_bars = len(spread)
        adf_weights = np.ones(n_bars)  # Default: all active

        for start_idx in range(0, n_bars, ADF_RESCORE_INTERVAL):
            lookback_start = max(0, start_idx - ADF_LOOKBACK)
            window_spread = spread.iloc[lookback_start:start_idx]

            if len(window_spread) > 100:
                p_val = score_pair_adf(window_spread)
                w = get_adf_weight(p_val)
            else:
                w = 1.0
            end_idx = min(start_idx + ADF_RESCORE_INTERVAL, n_bars)
            adf_weights[start_idx:end_idx] = w

        # --- Signal generation (tiered) ---
        sig_array = generate_signals_tiered(
            z_score.values,
            spread.values,
            roll_mean_spread.values,
            is_trending.values,
            recent_cross.values,
            exit_z=exit_z,
            min_div=min_div,
            max_hold=max_hold,
            adf_weights=adf_weights
        )
        signal = pd.Series(sig_array, index=z_score.index)

        # Execution delay: cannot execute at the same close that generated signal
        exec_signal = signal.shift(1).fillna(0)

        # Align order book to the OHLCV index first for use in loop
        obA_aligned = obA.reindex(df.index)
        obB_aligned = obB.reindex(df.index)

        # Position sizing with Kalman beta for dollar neutrality (V5 Friction-Aware adjustments)
        exec_signal_vals = exec_signal.values
        beta_prev_vals = beta.shift(1).fillna(1.0).values
        price_A_vals = price_A.values
        price_B_vals = price_B.values

        n_bars = len(df)
        pos_A_arr = np.zeros(n_bars)
        pos_B_arr = np.zeros(n_bars)

        # Pre-extract order book columns as arrays for fast access
        obA_bid_px = obA_aligned['l1_bid_px'].values
        obA_ask_px = obA_aligned['l1_ask_px'].values
        obA_bid_sz = obA_aligned['l1_bid_sz'].values
        obA_ask_sz = obA_aligned['l1_ask_sz'].values

        obB_bid_px = obB_aligned['l1_bid_px'].values
        obB_ask_px = obB_aligned['l1_ask_px'].values
        obB_bid_sz = obB_aligned['l1_bid_sz'].values
        obB_ask_sz = obB_aligned['l1_ask_sz'].values

        time_index = df.index
        hours = time_index.hour.values
        minutes = time_index.minute.values

        for i in range(n_bars):
            if i == 0:
                pos_A_arr[i] = 0.0
                pos_B_arr[i] = 0.0
                continue

            pos_A_prev = pos_A_arr[i-1]
            pos_B_prev = pos_B_arr[i-1]

            # Change 4: Time Window Check (only trade between start and end hours/minutes)
            h = hours[i]
            m = minutes[i]
            current_min = h * 60 + m
            start_min = EXEC_START_HOUR * 60 + EXEC_START_MINUTE
            end_min = EXEC_END_HOUR * 60 + EXEC_END_MINUTE
            in_window = (current_min >= start_min) and (current_min <= end_min)

            if not in_window:
                # Outside time window: hold prior position
                pos_A_arr[i] = pos_A_prev
                pos_B_arr[i] = pos_B_prev
                continue

            # Desired raw dollar positions
            desired_pos_A = exec_signal_vals[i] * ALLOCATION_PER_LEG
            desired_pos_B = -desired_pos_A * beta_prev_vals[i]

            # Change 3: Entry-Only Spread Gate
            # Triggers only when opening a new position (flat -> active)
            if pos_A_prev == 0.0 and desired_pos_A != 0.0:
                bid_A = obA_bid_px[i]
                ask_A = obA_ask_px[i]
                bid_B = obB_bid_px[i]
                ask_B = obB_ask_px[i]

                if not (np.isnan(bid_A) or np.isnan(ask_A) or np.isnan(bid_B) or np.isnan(ask_B)):
                    mid_A = (bid_A + ask_A) / 2.0
                    mid_B = (bid_B + ask_B) / 2.0
                    half_spread_A_bps = (ask_A - bid_A) / (2.0 * mid_A) * 10000.0
                    half_spread_B_bps = (ask_B - bid_B) / (2.0 * mid_B) * 10000.0
                    round_trip_bps = 2.0 * (half_spread_A_bps + half_spread_B_bps)

                    if round_trip_bps > MAX_ENTRY_SPREAD_BPS:
                        # Spread is too wide; skip/suppress the entry
                        desired_pos_A = 0.0
                        desired_pos_B = 0.0

            # Change 1: Liquidity-Adaptive Position Sizing
            min_scaling = 1.0
            if desired_pos_A != 0.0:
                shares_A = abs(desired_pos_A) / price_A_vals[i]
                is_buy_A = (desired_pos_A > 0)
                l1_sz_A = obA_ask_sz[i] if is_buy_A else obA_bid_sz[i]

                shares_B = abs(desired_pos_B) / price_B_vals[i]
                is_buy_B = (desired_pos_B > 0)
                l1_sz_B = obB_ask_sz[i] if is_buy_B else obB_bid_sz[i]

                scale_A = 1.0
                if not np.isnan(l1_sz_A) and l1_sz_A > 0:
                    scale_A = min(1.0, (LIQUIDITY_CAP_FRACTION * l1_sz_A) / shares_A)

                scale_B = 1.0
                if not np.isnan(l1_sz_B) and l1_sz_B > 0:
                    scale_B = min(1.0, (LIQUIDITY_CAP_FRACTION * l1_sz_B) / shares_B)

                # Perfect beta preservation: scale both legs by the minimum scaling factor
                min_scaling = min(scale_A, scale_B)

            scaled_pos_A = desired_pos_A * min_scaling
            scaled_pos_B = desired_pos_B * min_scaling

            # Change 2: Minimum Rebalance Threshold
            is_exit = (desired_pos_A == 0.0)
            is_entry = (pos_A_prev == 0.0)

            if not is_exit and not is_entry:
                # We are rebalancing. Check if change is too small.
                delta_A = scaled_pos_A - pos_A_prev
                delta_B = scaled_pos_B - pos_B_prev
                if abs(delta_A) < MIN_REBALANCE_DOLLARS and abs(delta_B) < MIN_REBALANCE_DOLLARS:
                    # Suppress small rebalance; hold prior
                    pos_A_arr[i] = pos_A_prev
                    pos_B_arr[i] = pos_B_prev
                else:
                    pos_A_arr[i] = scaled_pos_A
                    pos_B_arr[i] = scaled_pos_B
            else:
                # Entries and exits go through directly with scaling
                pos_A_arr[i] = scaled_pos_A
                pos_B_arr[i] = scaled_pos_B

        pos_A = pd.Series(pos_A_arr, index=df.index)
        pos_B = pd.Series(pos_B_arr, index=df.index)

        # ===== GROSS P&L (same as V3 — static friction) =====
        r_A = (price_A / price_A.shift(1)) - 1
        r_B = (price_B / price_B.shift(1)) - 1

        gross_pnl_A = pos_A.shift(1) * r_A
        gross_pnl_B = pos_B.shift(1) * r_B
        gross_pnl = (gross_pnl_A + gross_pnl_B).fillna(0)

        # ===== NET P&L (V4 — order book slippage engine) =====
        # Position changes trigger execution through the order book
        delta_pos_A = pos_A.diff().fillna(0)
        delta_pos_B = pos_B.diff().fillna(0)


        # Compute slippage + commissions per leg
        slip_A, comm_A = compute_slippage_for_pair(delta_pos_A, price_A, obA_aligned, 'A')
        slip_B, comm_B = compute_slippage_for_pair(delta_pos_B, price_B, obB_aligned, 'B')

        pair_slippage = (slip_A + slip_B).fillna(0)
        pair_commission = (comm_A + comm_B).fillna(0)
        pair_friction = pair_slippage + pair_commission

        net_pnl = gross_pnl - pair_friction
        net_pnl = net_pnl.fillna(0)

        # Accumulate friction totals
        total_commission_paid += pair_commission.sum()
        total_slippage_paid += pair_slippage.sum()
        total_gross_profit += gross_pnl.sum()

        # Per-pair trade tracking (gross and net)
        is_active = (pos_A != 0)
        trade_id = (is_active != is_active.shift(1)).cumsum()

        gross_trade_pnls = gross_pnl[is_active].groupby(trade_id[is_active]).sum()
        net_trade_pnls = net_pnl[is_active].groupby(trade_id[is_active]).sum()
        all_gross_trade_pnls.extend(gross_trade_pnls.tolist())
        all_net_trade_pnls.extend(net_trade_pnls.tolist())

        # Per-pair Sharpe (Gross and Net)
        gross_daily = gross_pnl.resample('D').sum()
        net_daily = net_pnl.resample('D').sum()
        gross_daily_ret = gross_daily / ALLOCATION_PER_LEG
        net_daily_ret = net_daily / ALLOCATION_PER_LEG

        def calc_sharpe(daily_returns):
            dr = daily_returns.dropna()
            if len(dr) > 5:
                return np.sqrt(252) * dr.mean() / (dr.std() + 1e-8)
            return 0.0

        pair_metrics[pair_label] = {
            "gross_sharpe": calc_sharpe(gross_daily_ret),
            "net_sharpe": calc_sharpe(net_daily_ret),
            "gross_pnl": gross_pnl.sum(),
            "net_pnl": net_pnl.sum(),
            "trades": len(gross_trade_pnls),
            "slippage": pair_slippage.sum(),
            "commission": pair_commission.sum(),
            "status": "ACTIVE" if adf_weights[-1] > 0 else "INACTIVE"
        }

        # Accumulate portfolio-level PnL
        if portfolio_gross_pnl is None:
            portfolio_gross_pnl = gross_pnl.copy()
            portfolio_net_pnl = net_pnl.copy()
        else:
            portfolio_gross_pnl = portfolio_gross_pnl.add(gross_pnl, fill_value=0)
            portfolio_net_pnl = portfolio_net_pnl.add(net_pnl, fill_value=0)

        print(f"OK ({len(gross_trade_pnls)} trades, "
              f"slip=${pair_slippage.sum():.2f}, comm=${pair_commission.sum():.2f})")

        # Free order book memory
        del obA, obB, obA_aligned, obB_aligned

    # --- Portfolio-level metrics ---
    portfolio_gross_pnl = portfolio_gross_pnl.fillna(0)
    portfolio_net_pnl = portfolio_net_pnl.fillna(0)

    # Equity curves
    gross_equity = INITIAL_EQUITY + portfolio_gross_pnl.cumsum()
    net_equity = INITIAL_EQUITY + portfolio_net_pnl.cumsum()

    # Max Drawdown
    def compute_mdd(equity_curve):
        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak
        return dd.min()

    gross_mdd = compute_mdd(gross_equity)
    net_mdd = compute_mdd(net_equity)

    # Daily returns for Sharpe
    gross_daily_total = portfolio_gross_pnl.resample('D').sum()
    net_daily_total = portfolio_net_pnl.resample('D').sum()

    gross_daily_ret = gross_daily_total / INITIAL_EQUITY
    net_daily_ret = net_daily_total / INITIAL_EQUITY

    gross_daily_all = gross_daily_ret.dropna()
    net_daily_all = net_daily_ret.dropna()

    def calc_sharpe_portfolio(daily_rets):
        if len(daily_rets) > 0:
            return np.sqrt(252) * daily_rets.mean() / (daily_rets.std() + 1e-8)
        return 0.0

    gross_sharpe = calc_sharpe_portfolio(gross_daily_all)
    net_sharpe = calc_sharpe_portfolio(net_daily_all)

    # Win rates
    gross_wr = sum(1 for p in all_gross_trade_pnls if p > 0) / len(all_gross_trade_pnls) if all_gross_trade_pnls else 0
    net_wr = sum(1 for p in all_net_trade_pnls if p > 0) / len(all_net_trade_pnls) if all_net_trade_pnls else 0

    # Rolling Sharpe kill switch diagnostic
    rolling_sharpe = net_daily_ret.rolling(30).apply(
        lambda x: np.sqrt(252) * x.mean() / (x.std() + 1e-8) if len(x) > 5 else 0,
        raw=True
    )
    halted_days = (rolling_sharpe < SHARPE_HALT_THRESHOLD).sum()
    total_days = len(rolling_sharpe.dropna())

    return {
        "Gross_Profit": total_gross_profit,
        "Net_Profit": portfolio_net_pnl.sum(),
        "Total_Commissions": total_commission_paid,
        "Total_Slippage": total_slippage_paid,
        "Gross_Sharpe": gross_sharpe,
        "Net_Sharpe": net_sharpe,
        "Gross_MDD": gross_mdd,
        "Net_MDD": net_mdd,
        "Gross_WinRate": gross_wr,
        "Net_WinRate": net_wr,
        "Total_Trades": len(all_gross_trade_pnls),
        "Pair_Metrics": pair_metrics,
        "Halted_Days": halted_days,
        "Total_Days": total_days,
        "Daily_Net_Returns": net_daily_all,
    }


# =====================================================================
# Console Report
# =====================================================================

def print_report(metrics, mode):
    print("\n" + "=" * 70)
    print(f"  NET-OF-FEES PERFORMANCE REPORT  [{mode}]")
    print(f"  V5 Friction-Aware — Deterministic L3 Order Book Slippage Engine")
    print("=" * 70)

    # --- Gross vs Net Comparison Table ---
    print("\n| Metric                   | Gross           | Net             |")
    print("|--------------------------|-----------------|-----------------|")
    print(f"| Total PnL ($)            | ${metrics['Gross_Profit']:>13,.2f} | ${metrics['Net_Profit']:>13,.2f} |")
    print(f"| Win Rate (%)             | {metrics['Gross_WinRate']*100:>14.2f}% | {metrics['Net_WinRate']*100:>14.2f}% |")
    print(f"| Annualized Sharpe Ratio  | {metrics['Gross_Sharpe']:>15.4f} | {metrics['Net_Sharpe']:>15.4f} |")
    print(f"| Max Drawdown (%)         | {metrics['Gross_MDD']*100:>14.2f}% | {metrics['Net_MDD']*100:>14.2f}% |")

    # --- Friction Breakdown ---
    print("\n" + "-" * 70)
    print("  FRICTION BREAKDOWN")
    print("-" * 70)
    print(f"  Total Gross Profit:          ${metrics['Gross_Profit']:>14,.2f}")
    print(f"  Total Commissions Paid:      ${metrics['Total_Commissions']:>14,.2f}")
    print(f"  Total Execution Friction:    ${metrics['Total_Slippage']:>14,.2f}")
    print(f"                               {'─' * 18}")
    print(f"  Final Net Profit:            ${metrics['Net_Profit']:>14,.2f}")

    # Sanity check
    reconstructed = metrics['Gross_Profit'] - metrics['Total_Commissions'] - metrics['Total_Slippage']
    diff = abs(reconstructed - metrics['Net_Profit'])
    if diff > 0.01:
        print(f"\n  ⚠ Reconciliation gap: ${diff:.4f} (rounding)")

    # --- Per-Pair Breakdown ---
    print("\n" + "=" * 70)
    print("  PER-PAIR BREAKDOWN")
    print("=" * 70)
    print(f"{'Pair':<14} | {'Gross Sharpe':<12} | {'Net Sharpe':<10} | {'Net PnL':<14} | {'Trades':<7} | {'Status':<8}")
    print("-" * 78)
    for pair_label, pm in sorted(metrics['Pair_Metrics'].items(),
                                  key=lambda x: x[1].get('net_sharpe', 0), reverse=True):
        print(f"{pair_label:<14} | {pm.get('gross_sharpe', 0):<12.2f} | {pm.get('net_sharpe', 0):<10.2f} | "
              f"${pm.get('net_pnl', 0):<13.2f} | {pm.get('trades', 0):<7} | {pm.get('status', 'N/A'):<8}")

    print("\n" + "=" * 70)
    print(f"  Sharpe Kill Switch: {metrics['Halted_Days']}/{metrics['Total_Days']} days halted")
    print("=" * 70 + "\n")


# =====================================================================
# CLI Entrypoint
# =====================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python pairs_trading_model.py [2022-2024|2024-2026|2022|2023]")
        return

    mode = sys.argv[1]
    if mode not in ("2022-2024", "2024-2026", "2022", "2023"):
        print(f"Unknown mode: {mode}")
        print("Usage: python pairs_trading_model.py [2022-2024|2024-2026|2022|2023]")
        return

    ohlc_dirs = get_ohlc_dirs(mode)
    ob_dirs = get_orderbook_dirs(mode)

    print("=" * 70)
    print(f"  V5 FRICTION-AWARE MICROSTRUCTURE BACKTEST [{mode}]")
    print("=" * 70)
    print("V5 Strategy Rules Active:")
    print(" + True 1D Kalman Filter for hedge ratio")
    print(" + Tiered Z-Score entry: 1.75/2.25/3.0")
    print(" + Correlation decay filter (floor=0.10)")
    print(" + 30-day rolling ADF universe re-scoring")
    print(" + Portfolio Sharpe kill switch diagnostic")
    print("V5 Friction-Aware Execution Optimizations:")
    print(f" + Liquidity-Adaptive sizing (cap: {LIQUIDITY_CAP_FRACTION*100:.0f}% of L1 depth)")
    print(f" + Minimum rebalance threshold: ${MIN_REBALANCE_DOLLARS:.0f}")
    print(f" + Entry-Only spread gate threshold: {MAX_ENTRY_SPREAD_BPS:.1f} bps")
    print(f" + Intraday execution time window: {EXEC_START_HOUR:02d}:00 to {EXEC_END_HOUR:02d}:{EXEC_END_MINUTE:02d} ET")
    print(f" + Fixed commission: ${COMMISSION_PER_SHARE}/share")
    print(f" + Fallback penalty: {FALLBACK_SLIPPAGE_BPS*10000:.0f} bps beyond L3 depth")
    print(f" + OHLCV dirs: {len(ohlc_dirs)}")
    print(f" + Order book dirs: {len(ob_dirs)}")
    print("-" * 70)

    metrics = run_v4_backtest(ohlc_dirs, ob_dirs)
    print_report(metrics, mode)

    # Export daily returns for portfolio optimization
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, f"pairs_returns_{mode}.csv")
    returns = metrics["Daily_Net_Returns"]
    returns.to_csv(out_file, header=['net_return'])
    print(f"Saved daily return series to {out_file}")

if __name__ == "__main__":
    main()
