# pt_helper.py
import os
import numpy as np
import pandas as pd
import yfinance as yf
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from pykalman import KalmanFilter
from scipy.optimize import minimize
from joblib import Parallel, delayed
from scipy.optimize import minimize
from pykalman import KalmanFilter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# ============================================================
# ===================== DATA LOADING =========================
# ============================================================

def load_or_update_prices(
    sector_map,
    start_date,
    price_path="Data/prices.csv",
):
    tickers = sorted(sector_map.keys())
    today = pd.Timestamp.today().normalize()
    start_date = pd.to_datetime(start_date)

    if not os.path.exists(price_path):
        return download_and_save_prices(tickers, start_date, price_path)

    prices = pd.read_csv(price_path, parse_dates=["Date"], index_col="Date")

    ticker_mismatch = set(prices.columns) != set(tickers)

    data_stale = (
        prices.index.max() < today
        or prices.index.min() > start_date
    )

    if ticker_mismatch or data_stale:
        return download_and_save_prices(tickers, start_date, price_path)

    return prices


def download_and_save_prices(tickers_init, start_date, price_path):
    # Ensure unique tickers and always include indices
    tickers = sorted(set(tickers_init) | {"QQQ", "XIU.TO"})

    prices = yf.download(
        tickers,
        start=start_date,
        auto_adjust=True,
        progress=False,
    )["Close"].ffill()

    prices.to_csv(price_path)
    return prices


# ============================================================
# ===================== PAIR SELECTION =======================
# ============================================================

def find_valid_pairs(log_prices, sector_map, coint_alpha=0.05):
    skip_list = {"SPY", "QQQ", "XIU.TO"}
    valid_pairs = []

    for y_ticker, x_ticker in combinations(log_prices.columns, 2):
        if y_ticker in skip_list or x_ticker in skip_list:
            continue

        if y_ticker.endswith(".TO") != x_ticker.endswith(".TO"):
            continue

        if sector_map[y_ticker] != sector_map[x_ticker]:
            continue

        y = log_prices[y_ticker]
        x = log_prices[x_ticker]

        _, pval, _ = coint(y, x)
        if pval >= coint_alpha:
            continue

        beta = np.polyfit(x, y, 1)[0]

        valid_pairs.append({
            "Y": y_ticker,
            "X": x_ticker,
            "coint_pval": pval,
            "beta_ols": beta,
        })

    return pd.DataFrame(valid_pairs)


def for_each_pair(pairs_df, log_prices, callback, sigma_mult):
    results = []

    for _, row in pairs_df.iterrows():
        y_ticker, x_ticker = row["Y"], row["X"]

        if y_ticker not in log_prices or x_ticker not in log_prices:
            continue

        out = callback(
            log_prices[y_ticker],
            log_prices[x_ticker],
            y_ticker,
            x_ticker,
            row,
            sigma_mult,
        )

        if out is not None:
            results.append(out)

    return results


# ============================================================
# ===================== TRAINING LOGIC =======================
# ============================================================

def train_pair_callback(y, x, y_ticker, x_ticker, row, sigma_mult):
    data = analyze_pair(y.values, x.values, sigma_mult)
    if data is None or not beta_stability_ok(data["beta_t"]):
        return None

    is_stat, adf_p = adf_stationary(data["spread"])
    if not is_stat:
        return None

    z = (data["spread"] - data["theta"]) / data["ou_std"]
    entries = np.sum((np.abs(z[1:]) > 2) & (np.abs(z[:-1]) <= 2))
    signals_per_year = entries / (len(z) / 252)

    return {
        "Y": y_ticker,
        "X": x_ticker,
        "coint_pval": row["coint_pval"],
        "beta_ols": row["beta_ols"],
        "beta_kf": np.mean(data["beta_t"]),
        "alpha_kf": np.mean(data["alpha_t"]),
        "ou_mu": data["mu"],
        "ou_std": data["ou_std"],
        "half_life_days": data["half_life_days"],
        "signals_per_year": signals_per_year,
        "theta": data["theta"],
        "adf_pval": adf_p,
    }


# ============================================================
# ================== KALMAN + OU CORE ========================
# ============================================================

def KFHedgeRatio(x, y):
    delta = 5e-3
    trans_cov = delta / (1 - delta) * np.eye(2)

    obs_mat = np.expand_dims(
        np.vstack([x, np.ones(len(x))]).T,
        axis=1
    )

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=2,
        transition_covariance=trans_cov,
        initial_state_mean=[0, 0],
        initial_state_covariance=np.ones((2, 2)),
    )

    state_means, _ = kf.filter(y)
    return state_means


def ou_log_likelihood(params, x):
    theta, mu, sigma = params
    x_lag = x[:-1]
    x_next = x[1:]

    mean = theta + (x_lag - theta) * np.exp(-mu)
    var = (sigma**2 / (2 * mu)) * (1 - np.exp(-2 * mu))
    var = np.maximum(var, 1e-8)

    ll = -0.5 * np.sum(
        np.log(2 * np.pi * var) + (x_next - mean) ** 2 / var
    )
    return -ll


def fit_ou(spread):
    init = [np.mean(spread), 0.1, np.std(spread)]
    bounds = [(None, None), (1e-6, None), (1e-6, None)]

    res = minimize(ou_log_likelihood, init, args=(spread,), bounds=bounds)
    if not res.success:
        return np.nan, np.nan, np.nan

    return res.x


def ou_valid(mu, min_mu=0.002):
    return mu > min_mu


def ou_stationary_std(mu, sigma):
    return sigma / np.sqrt(2 * mu)


def analyze_pair(y, x, k):
    state = KFHedgeRatio(x, y)
    beta_t, alpha_t = state[:, 0], state[:, 1]

    spread = y - (beta_t * x + alpha_t)

    theta, mu, sigma = fit_ou(spread)
    if not ou_valid(mu):
        return None

    ou_std = ou_stationary_std(mu, sigma)
    half_life = np.log(2) / mu

    return {
        "beta_t": beta_t,
        "alpha_t": alpha_t,
        "spread": spread,
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "ou_std": ou_std,
        "half_life_days": half_life,
    }


def beta_stability_ok(beta_t, max_cv=1.0):
    mean = np.mean(beta_t)
    std = np.std(beta_t)
    return abs(mean) > 1e-6 and (std / abs(mean)) < max_cv


def adf_stationary(series, alpha=0.1):
    _, pval, *_ = adfuller(series, autolag="AIC")
    return pval < alpha, pval












def backtest_pair_signals(
    signals_df,
    initial_capital=10_000,
    log_test=None,
    pos_size=100,
    commission=0.001,  # 0.1%
    slippage=0.001,  # 0.1%
):
    """
    PAIR-LEVEL backtest using dynamic KFHedgeRatio for beta.

    Changes:
    - Commission/slippage applied on both entry and exit
    - Uses KFHedgeRatio for beta_t dynamically at trade time
    """

    df = signals_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    open_pairs = {}  # pair_id -> trade state
    trades = []
    equity = initial_capital
    equity_curve = []

    # -------- GROUP EVENTS BY DATE --------
    for date, day_df in df.groupby("date"):

        # Expect 2 rows per pair per event
        for _, g in day_df.groupby(day_df.index // 2):

            if len(g) != 2:
                continue

            r1, r2 = g.iloc[0], g.iloc[1]

            s1, s2 = r1["signal"], r2["signal"]
            t1, t2 = r1["stock"], r2["stock"]

            pair_id = tuple(sorted([t1, t2]))

            # ---------- DYNAMIC BETA ----------
            y_prices = log_test[t2].loc[:date].values
            x_prices = log_test[t1].loc[:date].values

            state = KFHedgeRatio(x_prices, y_prices)
            beta_exec = state[-2, 0]

            # ---------- ENTRY ----------
            if s1 != 0 and s2 != 0:

                if pair_id in open_pairs:
                    continue

                shares1 = pos_size / r1["price"]
                shares2 = pos_size * abs(beta_exec) / r2["price"]

                entry_cost = (r1["price"] * shares1 + r2["price"] * shares2) * (
                    commission + slippage
                )

                equity -= entry_cost

                open_pairs[pair_id] = {
                    "entry_date": date,
                    "legs": {
                        t1: {"dir": s1, "price": r1["price"], "shares": shares1},
                        t2: {"dir": s2, "price": r2["price"], "shares": shares2},
                    },
                }

            # ---------- EXIT ----------
            elif s1 == 0 and s2 == 0:

                if pair_id not in open_pairs:
                    continue

                trade = open_pairs.pop(pair_id)
                pnl = 0.0

                for stock, leg in trade["legs"].items():
                    exit_price = g[g["stock"] == stock]["price"].iloc[0]

                    trade_pnl = (exit_price - leg["price"]) * leg["dir"] * leg["shares"]

                    dollar_traded = exit_price * leg["shares"]
                    trade_pnl -= dollar_traded * (commission + slippage)

                    pnl += trade_pnl

                equity += pnl
                equity_curve.append((date, equity))

                trades.append(
                    {
                        "pair": f"{pair_id[0]} / {pair_id[1]}",
                        "entry_date": trade["entry_date"],
                        "exit_date": date,
                        "pnl": pnl,
                        "return": pnl / initial_capital,
                    }
                )

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        raise ValueError("No completed pair trades.")

    # ---------- METRICS ----------
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]

    start = trades_df["entry_date"].min()
    end = trades_df["exit_date"].max()
    years = (end - start).days / 365.25

    cagr = (equity / initial_capital) ** (1 / years) - 1 if years > 0 else np.nan

    summary = {
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return": equity - initial_capital,
        "CAGR": cagr,
        "total_trades": len(trades_df),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(trades_df),
        "avg_trade_pnl": trades_df["pnl"].mean(),
        "median_trade_pnl": trades_df["pnl"].median(),
        "max_win": trades_df["pnl"].max(),
        "max_loss": trades_df["pnl"].min(),
        "profit_factor": (
            wins["pnl"].sum() / abs(losses["pnl"].sum()) if not losses.empty else np.inf
        ),
    }

    equity_curve = (
        pd.DataFrame(equity_curve, columns=["date", "equity"])
        .drop_duplicates("date")
        .set_index("date")["equity"]
    )

    return trades_df, summary, equity_curve





# ML RANKING HELPER FUNCTIONS


def top_k_mean_target(df, k):
    return df[df["Rank"] <= k].groupby("Date")["Target"].mean().mean()


def daily_spearman(df):
    ic = df.groupby("Date").apply(
        lambda x: (
            x["Pred_Score"].corr(x["Target"], method="spearman")
            if x["Target"].nunique() > 1
            else np.nan
        )
    )
    return ic.mean(), ic.std()


def build_pairs_from_selection_csv(
    csv_path="Data/selection_pairs.csv",
    tsx_index="XIU.TO",
    us_index="QQQ",
):
    df = pd.read_csv(csv_path)

    pairs = []

    for _, row in df.iterrows():
        y = row["Y"]
        x = row["X"]

        # Determine index
        if y.endswith(".TO") or x.endswith(".TO"):
            index = tsx_index
        else:
            index = us_index

        pairs.append([y, x, index])

    return pairs


def compute_index_features(price_df, index_tickers=("XIU.TO", "QQQ")):
    """
    Compute index-level features once per index ticker.
    Returns dict: idx_ticker -> DataFrame
    """
    idx_feature_map = {}

    for idx in index_tickers:
        if idx not in price_df.columns:
            raise ValueError(f"Index ticker {idx} not found in price_df")

        idx_close = pd.to_numeric(price_df[idx], errors="coerce")
        idx_returns = np.log(idx_close).diff()

        idx_vol_21 = idx_returns.rolling(21).std()
        idx_vol_z = (idx_vol_21 - idx_vol_21.rolling(252).mean()) / idx_vol_21.rolling(
            252
        ).std()

        idx_trend_strength = (
            idx_returns.rolling(63).mean().abs() / idx_returns.rolling(63).std()
        )

        idx_feature_map[idx] = pd.DataFrame(
            {
                "idx_vol_z": idx_vol_z,
                "idx_trend_strength": idx_trend_strength,
            },
            index=price_df.index,
        )

    return idx_feature_map


def KFHedgeRatio_xg(x, y, delta=2e-4):
    trans_cov = delta / (1 - delta) * np.eye(2)

    obs_mat = np.expand_dims(np.vstack([x, np.ones(len(x))]).T, axis=1)

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=3,
        transition_covariance=trans_cov,
        initial_state_mean=[0.0, 0.0],
        initial_state_covariance=np.eye(2),
    )

    state_means, _ = kf.filter(y)
    return state_means  # ← NO NEGATION


def ou_log_likelihood_xg(params, x):
    theta, mu, sigma = params
    dt = 1

    x_lag = x[:-1]
    x_next = x[1:]

    mean = theta + (x_lag - theta) * np.exp(-mu * dt)
    var = (sigma**2 / (2 * mu)) * (1 - np.exp(-2 * mu * dt))
    var = np.maximum(var, 1e-8)

    ll = -0.5 * np.sum(np.log(2 * np.pi * var) + (x_next - mean) ** 2 / var)
    return -ll


def fit_ou_xg(spread):
    init = [0.0, 0.1, np.std(spread)]
    bounds = [(None, None), (1e-6, None), (1e-6, None)]
    res = minimize(ou_log_likelihood_xg, init, args=(spread,), bounds=bounds)
    return res.x




def pair_feature_engineer(
    data,
    y_ticker,
    x_ticker,
    idx_features,
    idx_ticker,
    start,
    end,
    rolling_window=21,
    N_mult=2,
    ou_window=126,
):
    # -----------------------------
    # Slice data
    # -----------------------------
    data = data.loc[start:end]
    ou_refit_freq = 5  #
    y_close = data[y_ticker]
    x_close = data[x_ticker]

    y_returns = np.log(y_close).diff()
    x_returns = np.log(x_close).diff()

    df = pd.DataFrame(index=data.index)
    df["Y_Close"] = y_close
    df["X_Close"] = x_close

    # -----------------------------
    # Kalman Filter hedge ratio
    # -----------------------------
    state = KFHedgeRatio_xg(x_close.values, y_close.values)
    df["beta_kf"] = state[:, 0]
    df["alpha_kf"] = state[:, 1]

    df["beta_kf_mean"] = df["beta_kf"].rolling(rolling_window).mean()
    df["beta_kf_std"] = df["beta_kf"].rolling(rolling_window).std()

    df["alpha_kf_mean"] = df["alpha_kf"].rolling(rolling_window).mean()
    df["alpha_kf_std"] = df["alpha_kf"].rolling(rolling_window).std()

    # -----------------------------
    # Index regime features
    # -----------------------------
    idx_df = idx_features[idx_ticker]
    df["idx_trend_strength"] = idx_df["idx_trend_strength"]
    df["idx_vol_z"] = idx_df["idx_vol_z"]

    # -----------------------------
    # Correlation regime
    # -----------------------------
    pair_corr = y_returns.rolling(63).corr(x_returns)
    df["corr_z"] = (pair_corr - pair_corr.rolling(252).mean()) / pair_corr.rolling(
        252
    ).std()

    # -----------------------------
    # KF spread (RAW)
    # -----------------------------
    df["spread"] = df["Y_Close"] - (df["beta_kf"] * df["X_Close"] + df["alpha_kf"])
    df["spread_log_change"] = np.log(df["spread"].abs() + 1e-8).diff()

    spread = df["spread"].values

    # =====================================================
    # OU FIT — refit every K days, forward-fill
    # =====================================================
    ou_mu = np.full(len(df), np.nan)
    ou_theta = np.full(len(df), np.nan)
    ou_sigma = np.full(len(df), np.nan)
    half_life = np.full(len(df), np.nan)

    last_fit_idx = None
    last_params = None

    for t in range(len(spread)):
        if t < ou_window:
            continue

        should_refit = last_fit_idx is None or (t - last_fit_idx) >= ou_refit_freq

        if should_refit:
            window_spread = spread[t - ou_window : t]

            try:
                theta, mu, sigma = fit_ou_xg(window_spread)

                if mu > 0 and sigma > 0:
                    hl = np.log(2) / mu
                    last_params = (theta, mu, sigma, hl)
                    last_fit_idx = t
            except Exception:
                pass

        if last_params is not None:
            ou_theta[t], ou_mu[t], ou_sigma[t], half_life[t] = last_params

    df["ou_theta"] = ou_theta
    df["ou_mu"] = ou_mu
    df["ou_sigma"] = ou_sigma
    df["half_life_days"] = half_life

    # -----------------------------
    # OU-derived features
    # -----------------------------
    df["ou_std"] = df["ou_sigma"] / np.sqrt(2 * df["ou_mu"])
    df["spread_centered"] = df["spread"] - df["ou_theta"]
    df["ou_z"] = (df["spread_centered"] / df["ou_std"]).clip(-3, 3)
    df["abs_ou_z"] = df["ou_z"].abs()

    df["upper"] = N_mult * df["ou_std"]
    df["lower"] = -N_mult * df["ou_std"]

    df["ou_mu_change"] = df["ou_mu"].diff().abs()
    df["ou_mu_stable"] = (df["ou_mu_change"] < 0.05).astype(int)

    df["reversion_pressure"] = df["abs_ou_z"] * df["ou_mu"]
    df.loc[df["half_life_days"] > 252, "half_life_days"] = np.nan

    df["expected_reversion_days"] = df["half_life_days"] * np.log(df["abs_ou_z"] + 1)

    df["ou_snr"] = df["ou_mu"] / (df["ou_sigma"] ** 2)
    df["ou_tradeable"] = (df["ou_mu"] > 0.05).astype(int)

    # -----------------------------
    # Beta stability
    # -----------------------------
    df["beta_drift"] = df["beta_kf"].diff().abs().rolling(rolling_window).mean()
    df["beta_cv"] = (
        df["beta_kf"].rolling(rolling_window).std()
        / df["beta_kf"].rolling(rolling_window).mean().abs()
    )

    # -----------------------------
    # Regime conflict
    # -----------------------------
    df["trend_conflict"] = (
        (df["idx_trend_strength"] > 1.0) & (df["abs_ou_z"] > 1.5)
    ).astype(int)

    # -----------------------------
    # Signal density
    # -----------------------------
    z = df["ou_z"].replace([np.inf, -np.inf], np.nan)
    df["signals_per_year"] = z.rolling(
        rolling_window, min_periods=rolling_window
    ).apply(lambda x: np.sum((np.abs(x[1:]) > 2) & (np.abs(x[:-1]) <= 2)), raw=True) * (
        252 / rolling_window
    )
    df["signals_per_year_log"] = np.log1p(df["signals_per_year"])

    # -----------------------------
    # Labels
    # -----------------------------
    df["Y"] = y_ticker
    df["X"] = x_ticker

    # -----------------------------
    # Feature set (FIXED BUG HERE)
    # -----------------------------
    feature_cols = [
        "signals_per_year_log",
        "beta_kf_mean",
        "ou_mu_stable",
        "reversion_pressure",
        "ou_snr",
        "beta_drift",
        "alpha_kf_mean",
        "abs_ou_z",
        "expected_reversion_days",
        "beta_cv",
        "spread_log_change",
        "half_life_days",
        "corr_z",
        "idx_trend_strength",
        "idx_vol_z",
    ]

    df = df.dropna(subset=feature_cols)

    return df, feature_cols


def create_target(df, N):
    df = df.copy()
    df["target"] = np.nan

    for y, x in df[["Y", "X"]].drop_duplicates().values:
        mask = (df["Y"] == y) & (df["X"] == x)
        pair_df = df.loc[mask]

        z = np.abs(pair_df["ou_z"].values)
        

        tgt = []

        for i in range(len(z) - N):
            target = (z[i] - z[i+N]) / abs(z[i])
            tgt.append(target)

        df.loc[mask, "target"] = np.concatenate([tgt, [np.nan] * N])

    return df


def build_pair_dataset(
    price_df, pairs_list, start, end, rolling_window=21, N=3, n_jobs=-1, N_mult=2
):
    """
    Faster version:
    - Index features computed once
    - Pair logic unchanged
    """

    idx_features = compute_index_features(price_df)

    def _process_single_pair(y_ticker, x_ticker, idx_ticker):
        try:
            df, feature_cols = pair_feature_engineer(
                price_df,
                y_ticker,
                x_ticker,
                idx_features,  # ✅ correct
                idx_ticker,  # ✅ correct
                start,
                end,
                rolling_window,
                N_mult,
            )

            df = create_target(df, N=N)
            return df, feature_cols
        except Exception as e:
            print(f"[SKIP] {y_ticker}-{x_ticker}: {e}")
            return None

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_single_pair)(y, x, idx) for y, x, idx in pairs_list
    )

    results = [r for r in results if r is not None]

    if not results:
        raise ValueError("No valid pair datasets produced.")

    # Unzip
    dfs, feature_cols_list = zip(*results)

    # (Optional but good) ensure feature columns are consistent
    feature_cols = feature_cols_list[0]
    for fc in feature_cols_list[1:]:
        if fc != feature_cols:
            raise ValueError("Inconsistent feature columns across pairs")

    final_df = pd.concat(dfs, axis=0).sort_index()

    return final_df, feature_cols


def plot(df, pairs_with_index, N=2):
    """
    Plot OU z-score ONLY for multiple pairs.
    - 6 subplots per figure (3x2)
    - Green ▲ at lower band
    - Red ▼ at upper band
    """

    pairs_2d = [[y, x] for y, x in pairs_with_index]

    for batch_start in range(0, len(pairs_2d), 6):
        batch = pairs_2d[batch_start : batch_start + 6]

        fig = make_subplots(
            rows=3,
            cols=2,
            shared_xaxes=False,
            subplot_titles=[f"{y}/{x}" for y, x in batch],
            vertical_spacing=0.10,
            horizontal_spacing=0.08,
        )

        for i, (y, x) in enumerate(batch):
            row = i // 2 + 1   # 2 columns
            col = i % 2 + 1

            subset = df[(df["Y"] == y) & (df["X"] == x)].copy()
            if subset.empty:
                continue

            subset = subset.sort_index()
            z = subset["ou_z"]

            # --- OU Z line ---
            fig.add_trace(
                go.Scatter(
                    x=subset.index,
                    y=z,
                    mode="lines",
                    line=dict(color="purple"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # --- Bands ---
            fig.add_trace(
                go.Scatter(
                    x=subset.index,
                    y=[N] * len(subset),
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            
            fig.add_trace(
                go.Scatter(
                    x=subset.index,
                    y=[-N] * len(subset),
                    mode="lines",
                    line=dict(color="green", dash="dash"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )


            # --- Touch (crossing) events only ---
            z_prev = z.shift(1)

            upper_touch = (z_prev < N) & (z >= N)
            lower_touch = (z_prev > -N) & (z <= -N)

            fig.add_trace(
                go.Scatter(
                    x=subset.index[upper_touch],
                    y=z[upper_touch],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        color="red",
                        size=9,
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=subset.index[lower_touch],
                    y=z[lower_touch],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        color="green",
                        size=9,
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            fig.update_yaxes(range=[-4, 4], row=row, col=col)

        fig.update_layout(
            title=f"OU Z-Score Diagnostics (Pairs {batch_start+1}–{batch_start+len(batch)})",
            template="plotly_white",
            height=900,
            hovermode="x unified",
        )

        fig.show()
