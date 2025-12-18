import os
import pickle
from matplotlib import cm
from matplotlib.dates import MonthLocator, YearLocator
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler

if not hasattr(np, "NINF"):
    np.NINF = -np.inf
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from hmmlearn import hmm
import pyfolio.timeseries as pf_ts
import ta
import warnings


def normalize_data(df):
    """Normalize the feature columns using StandardScaler."""

    return df


def get_data(ticker, ticker2, start_date, end_date):
    # Download data
    data1 = yf.download(
        ticker, start=start_date, end=end_date, auto_adjust=True, progress=False
    )

    data2 = yf.download(
        ticker2, start=start_date, end=end_date, auto_adjust=True, progress=False
    )
    if isinstance(data1.columns, pd.MultiIndex):
        data1.columns = data1.columns.get_level_values(0)

    if isinstance(data2.columns, pd.MultiIndex):
        data2.columns = data2.columns.get_level_values(0)

    # Rename columns of second ticker
    data2 = data2.add_suffix("2")

    # Compute returns
    data1["returns"] = data1["Close"].pct_change()
    data2["returns2"] = data2["Close2"].pct_change()

    # Merge on index (dates)
    final_data = data1.join(data2, how="inner")

    # Optional: save
    final_data.to_csv("./Data/1_Start_Data.csv", index=True)

    return final_data


def plot_results(results_df, ticker):

    cols = ["returns", "HMM Prob(R0)", "HMM Prob(R1)", "Signal [RF]", "signal"]
    results_df[cols].to_csv("./Data/3_Final_Data.csv", index=True)
    if results_df.empty:
        print(f"No results to plot for {ticker}.")
        return

    df = results_df.copy()

    # Buy & Hold SPY
    df["bh_cum_rets"] = (1 + df["returns"]).cumprod()

    # Strategy returns
    df["strategy_returns"] = 0.0

    # SPY logic
    spy_mask = df["signal"].isin([1, -1])
    df.loc[spy_mask, "strategy_returns"] = (
        df.loc[spy_mask, "returns"] * df.loc[spy_mask, "signal"]
    )

    # GLD logic
    gld_mask = df["signal"] == 2
    df.loc[gld_mask, "strategy_returns"] = df.loc[gld_mask, "returns2"]

    df["strategy_cum_rets"] = (1 + df["strategy_returns"]).cumprod()

    # --- PLOTTING ---
    plt.figure(figsize=(15, 7))

    # Buy & hold
    plt.plot(
        df.index,
        df["bh_cum_rets"],
        label="Buy & Hold (SPY)",
        color="gray",
        alpha=0.6,
        linewidth=2,
    )

    # Plot continuous colored segments
    start = 0
    current_asset = None

    for i in range(1, len(df)):
        sig = df.iloc[i]["signal"]

        asset = "GLD" if sig == 2 else "SPY"

        if current_asset is None:
            current_asset = asset

        if asset != current_asset:
            segment = df.iloc[start:i]
            color = "gold" if current_asset == "GLD" else "green"
            plt.plot(
                segment.index, segment["strategy_cum_rets"], color=color, linewidth=2.5
            )
            start = i
            current_asset = asset

    # Plot final segment
    segment = df.iloc[start:]
    color = "gold" if current_asset == "GLD" else "green"
    plt.plot(
        segment.index,
        segment["strategy_cum_rets"],
        color=color,
        linewidth=2.5,
        label="Strategy (SPY=Green, GLD=Gold)",
    )

    plt.title(f"{ticker} Strategy Cumulative Returns", fontsize=16)
    plt.xlabel("Date", fontsize=15)
    plt.ylabel("Cumulative Returns", fontsize=15)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


def compute_perf_stats(results_to_plot):
    df = results_to_plot.copy()

    # Buy & Hold SPY
    df["bh_returns"] = df["returns"].fillna(0)

    # Strategy returns
    df["strategy_returns"] = 0.0

    spy_mask = df["signal"].isin([1, -1, 0])
    df.loc[spy_mask, "strategy_returns"] = (
        df.loc[spy_mask, "returns"] * df.loc[spy_mask, "signal"]
    )

    gld_mask = df["signal"] == 2
    df.loc[gld_mask, "strategy_returns"] = df.loc[gld_mask, "returns2"]

    strategy_stats = pf_ts.perf_stats(df["strategy_returns"])
    bh_stats = pf_ts.perf_stats(df["bh_returns"])

    perf_table = pd.concat([bh_stats, strategy_stats], axis=1)
    perf_table.columns = ["Buy & Hold (SPY)", "Strategy (SPY + GLD)"]

    desired_metrics = [
        "Annual return",
        "Cumulative returns",
        "Annual volatility",
        "Sharpe ratio",
        "Calmar ratio",
        "Max drawdown",
        "Sortino ratio",
    ]

    final_table = perf_table.loc[desired_metrics]

    for idx in final_table.index:
        for col in final_table.columns:
            val = final_table.loc[idx, col]
            if idx in [
                "Annual return",
                "Cumulative returns",
                "Annual volatility",
                "Max drawdown",
            ]:
                final_table.loc[idx, col] = f"{val * 100:.2f}%"
            else:
                final_table.loc[idx, col] = f"{val:.2f}"

    print(final_table)


def engineer_features_SPEC(data_df, num_lead):

    data = data_df.copy()

    data["Open-Close"] = (data["Open"] - data["Close"]) / data["Open"]
    data["High-Low"] = (data["High"] - data["Low"]) / data["Low"]
    data["Return_1d"] = data["Close"].pct_change()
    data["std_5"] = (
        data["Return_1d"]
        .rolling(window=5, min_periods=1)
        .apply(lambda x: np.std(x[:-1]), raw=False)
    )
    data_df["vol_5"] = data_df["returns"].rolling(5).std()
    data["Up_Days_5"] = (data["Return_1d"] > 0).rolling(5).mean()
    data["Up_Days_10"] = (data["Return_1d"] > 0).rolling(10).mean()
    data["Trend_Eff_10"] = data["Close"].pct_change(10).abs() / (
        data["Close"].diff().abs().rolling(10).sum()
    )
    data["Range_Pos_20"] = (data["Close"] - data["Low"].rolling(20).min()) / (
        data["High"].rolling(20).max() - data["Low"].rolling(20).min()
    )
    data["Vol_Exp_5_20"] = data["std_5"] / data["Return_1d"].rolling(20).std()
    data["VPT"] = (1 + data["Return_1d"]) * data["Volume"]
    data["VPT_cum"] = data["VPT"].cumsum()
    data["VPT_Momentum"] = data["VPT_cum"].diff()
    data["VPT_Direction"] = np.sign(data["VPT_Momentum"])
    data["Vol_MA20"] = data["Volume"].rolling(20).mean()
    data["Vol_Ratio"] = data["Volume"] / data["Vol_MA20"]
    data["VolRank_20"] = data["std_5"].rank(pct=True)
    data["Vol_Residual"] = data["Volume"] - data["Vol_MA20"]
    data["Vol_Trend"] = data["Vol_Ratio"] * np.sign(data["Close"].pct_change(5))
    data["ATR"] = ta.volatility.AverageTrueRange(
        high=data["High"], low=data["Low"], close=data["Close"], window=14
    ).average_true_range()
    data["ATR_pct"] = data["ATR"] / data["Close"]
    data["RSI_14"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
    data["ROC_5"] = data["Close"].pct_change(5)
    bb = ta.volatility.BollingerBands(data["Close"], window=20, window_dev=2)
    data["BB_Pct"] = (data["Close"] - bb.bollinger_mavg()) / (
        bb.bollinger_hband() - bb.bollinger_lband()
    )
    data["BB_width"] = bb.bollinger_hband() - bb.bollinger_lband()
    data["EMA10"] = data["Close"].ewm(span=10).mean()
    data["EMA20"] = data["Close"].ewm(span=20).mean()
    data["EMA50"] = data["Close"].ewm(span=50).mean()
    data["EMA100"] = data["Close"].ewm(span=100).mean()
    data["EMA_Slope"] = data["EMA20"] - data["EMA50"]
    data["EMA_Slope_Long"] = data["EMA20"] - data["EMA100"]
    data["ADX_14"] = ta.trend.ADXIndicator(
        data["High"], data["Low"], data["Close"], window=14
    ).adx()
    data["ADX_Slope"] = data["ADX_14"].diff()
    macd = ta.trend.MACD(data["Close"])
    data["MACD_Hist"] = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(
        data["High"], data["Low"], data["Close"], window=14
    )
    data["Stoch_K"] = stoch.stoch()
    data["Close_Z20"] = (data["Close"] - data["Close"].rolling(20).mean()) / data[
        "Close"
    ].rolling(20).std()
    data["Ret_3"] = data["Close"].pct_change(3)
    data["Ret_10"] = data["Close"].pct_change(10)
    data["Trend_Align"] = (
        np.sign(data["Ret_3"]) + np.sign(data["ROC_5"]) + np.sign(data["Ret_10"])
    ) / 3
    data["EMA_Stack"] = (
        (data["EMA10"] > data["EMA20"]).astype(int)
        + (data["EMA20"] > data["EMA50"]).astype(int)
        + (data["EMA50"] > data["EMA100"]).astype(int)
    )
    data.to_csv("./Data/2_Mid_Process_data.csv")

    final_feature_columns = [
        "Open-Close",
        "High-Low",
        "std_5",
        "Vol_Ratio",
        "VPT_Momentum",
        "Return_1d",
        "ATR_pct",
        "RSI_14",
        "ROC_5",
        "BB_Pct",
        "EMA_Slope",
        "ADX_14",
        "VolRank_20",
        "Vol_Residual",
        "VPT_Direction",
        "EMA_Slope_Long",
        "ADX_Slope",
        "MACD_Hist",
        "Vol_Trend",
        "Ret_10",
        "BB_width",
        "Stoch_K",
        "Close_Z20",
        "Ret_3",
    ]

    data["y_signal"] = (data["returns"].shift(-num_lead) > 0.002).astype(int)

    return data, final_feature_columns


def plot_regimes_from_df(tracking_df, price_df, prob_threshold=0.5):

    price_col = "Adj Close" if "Adj Close" in price_df.columns else "Close"

    n_regimes = 2
    fig, axs = plt.subplots(n_regimes, 1, figsize=(16, 8), sharex=True)
    colours = cm.rainbow(np.linspace(0, 1, n_regimes))

    common_index = price_df.index.intersection(tracking_df.index)
    price_data = price_df.loc[common_index]
    tracking_df = tracking_df.loc[common_index]

    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = tracking_df[f"HMM Prob(R{i})"] > prob_threshold
        ax.plot(
            price_data.index[mask],
            price_data[price_col][mask],
            ".",
            linestyle="none",
            c=colour,
        )
        ax.set_title(f"Regime {i} (Prob > {prob_threshold:.2f})")
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def run_backtest_STABLE(
    data_df,
    feature_list,
    backtest_signal_start_date_str,
    num_lead,
    hmm_n_past_years_data,
    rf_n_past_months_data,
    hmm_train_nth_week,
    rf_train_nth_week,
    thresh_prob,
    params=None,
):
    os.makedirs("./ML_Models", exist_ok=True)
    hmm_pickle_path = "./ML_Models/hmm_model.pkl"
    rf_pickle_path = "./ML_Models/rf_model.pkl"

    hmm_params = params.get("hmm_params", {})
    rf_params = params.get("rf_params", {})

    hmm_window_days = int(hmm_n_past_years_data * 252)  # a years
    rf_window_days = int(rf_n_past_months_data * 21)  # b months (~21 trading days each)
    X = int(hmm_train_nth_week * 5)  # X days = x weeks * 5
    Y = int(rf_train_nth_week * 5)  # Y days = y weeks * 5

    data_df["signal"] = 0.0
    data_df["EMA_5"] = data_df["Close"].ewm(span=2, adjust=False).mean()
    data_df["EMA_20"] = data_df["Close"].ewm(span=8, adjust=False).mean()
    log_spy = np.log(data_df["Close"])
    log_gld = np.log(data_df["Close2"])
    window = 120
    beta = log_spy.rolling(window).cov(log_gld) / log_gld.rolling(window).var()
    spread = log_spy - beta * log_gld
    spread_mean = spread.rolling(window).mean()
    spread_std = spread.rolling(window).std()
    z_spread = (spread - spread_mean) / spread_std

    start_date = pd.to_datetime(backtest_signal_start_date_str)
    start_idx = data_df.index.get_loc(data_df.index[data_df.index >= start_date][0])

    hmm_model = None
    rf = None
    if os.path.exists(hmm_pickle_path):
        with open(hmm_pickle_path, "rb") as f:
            hmm_model = pickle.load(f)
    if os.path.exists(rf_pickle_path):
        with open(rf_pickle_path, "rb") as f:
            rf = pickle.load(f)

    for t in range(start_idx, len(data_df)):

    # ----- Train HMM every X days -----
        if (t - start_idx) % X == 0:
            hmm_start = max(0, t - hmm_window_days)
            hmm_train = data_df.iloc[hmm_start:t].copy()

            # Scale returns in HMM training window only
            scaler_hmm = StandardScaler()
            hmm_train["norm_returns"] = scaler_hmm.fit_transform(hmm_train[["returns"]])

            hmm_train_nonan = hmm_train[["norm_returns"]].dropna()
            hmm_model = hmm.GaussianHMM(**hmm_params)
            hmm_model.fit(hmm_train_nonan.values)

            # Save HMM
            with open(hmm_pickle_path, "wb") as f:
                pickle.dump(hmm_model, f)

        # ----- Train RF every Y days -----
        if (t - start_idx) % Y == 0:
            rf_start = max(0, t - 900)
            rf_train = data_df.iloc[rf_start:t].copy()

            # Scale returns only in RF window
            scaler_rf = StandardScaler()
            rf_train["norm_returns"] = scaler_rf.fit_transform(rf_train[["returns"]])

            # Make sure we have y_signal and features
            rf_train = rf_train.dropna(subset=["norm_returns"] + feature_list + ["y_signal"])

            # Assign HMM regimes (based on RF training window)
            regimes = hmm_model.predict(rf_train[["norm_returns"]])
            rf_train["regime"] = regimes

            # Use only regime 0 for RF training
            df0 = rf_train[rf_train["regime"] == 0].tail(rf_window_days)

            rf = RandomForestClassifier(**rf_params)
            rf.fit(df0[feature_list], df0["y_signal"])

            # Save RF
            with open(rf_pickle_path, "wb") as f:
                pickle.dump(rf, f)

    # ----- Make predictions -----
        if t - num_lead < 0:
            continue

        feature_row = data_df.iloc[[t - num_lead]][feature_list].copy()
        last_obs = np.array([[data_df["returns"].iloc[t - 1]]])
        last_obs_norm = scaler_hmm.transform(last_obs)  # normalize same as HMM training

        p0, p1 = hmm_model.predict_proba(last_obs_norm)[0]
        p0 = np.clip(p0, 0.1, 0.9)
        p1 = np.clip(p1, 0.1, 0.9)

        try:
            s0 = rf.predict_proba(feature_row)[0][1]
        except:
            s0 = 0.5

        data_df.loc[data_df.index[t], "HMM Prob(R0)"] = p0
        data_df.loc[data_df.index[t], "HMM Prob(R1)"] = p1
        data_df.loc[data_df.index[t], "Signal [RF]"] = s0


        z_t = z_spread.iloc[t]

        ema_signal = 0
        if (
            data_df.loc[data_df.index[t], "EMA_5"]
            > data_df.loc[data_df.index[t], "EMA_20"]
        ):
            ema_signal = 1
        elif (
            data_df.loc[data_df.index[t], "EMA_5"]
            < data_df.loc[data_df.index[t], "EMA_20"]
        ):
            ema_signal = -1
            
        
        if p0 > p1 and abs(p0 - p1) > thresh_prob:
            if s0 > 0.5 and ema_signal == 1:
                data_df.loc[data_df.index[t], "signal"] = 1

            elif s0 < 0.5 and ema_signal == -1:
                data_df.loc[data_df.index[t], "signal"] = -1

            elif z_t > 1:
                data_df.loc[data_df.index[t], "signal"] = 2

            else:
                data_df.loc[data_df.index[t], "signal"] = 0

        elif p1 > p0 and abs(p0 - p1) > thresh_prob:
            data_df.loc[data_df.index[t], "signal"] = 2

        else:
            data_df.loc[data_df.index[t], "signal"] = 0

    return data_df


# TO-DO:
# change hmm training window dynamically based on market volatility
# add dynamic position sizing based on the confidence level of hmm and rf
# set stop loss based on ATR

# pair trading with gold and spy
# train another RF only on the relation between gld and spy (hedge ratio,spread,z-score) and thier features, with a threshold on spread z score
#


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    TICKER = "SPY"
    TICKER2 = "GLD"
    START_DATE = "1993-01-01"
    BACKTEST_SIGNAL_START_DATE = "2025-06-12"
    END_DATE = "2025-12-12"
    NUM_LEAD = 1

    raw_data = get_data(TICKER, TICKER2, START_DATE, END_DATE)
    data_with_features, feature_cols = engineer_features_SPEC(raw_data.copy(), NUM_LEAD)

    print(f"Data prepared. Number of features: {len(feature_cols)}")
    print(f"Data shape after preprocessing: {data_with_features.shape}")

    params_opt = {
        # HMM hyperparameters
        "hmm_params": {
            "n_components": 2,  # number of hidden states
            "covariance_type": "full",  # allows correlated volatility; 'diag' = simpler, less flexible
            "n_iter": 591,  # max training steps; higher = slower but more stable (200–600)
            "tol": 0.00025461513333633457,  # convergence threshold; lower = more precise but slower (1e-2 to 1e-4)
            "random_state": 42,  # seed
        },
        # RF hyperparameters
        "rf_params": {
            "n_estimators": 300,  # number of trees; higher = smoother & less noise but slower (150–400)
            "max_depth": 4  # tree depth; higher = more complex/overfits, lower = smoother/general (2–5)
            # ,"min_samples_leaf": 30   # minimum samples per leaf; higher = smoother/less noise (10–50)
            ,
            "max_features": "sqrt",  # features per split; lower = less correlated trees (usually 'sqrt')
            "bootstrap": True,  # sample with replacement; True = required for stable RF behaviour
            "random_state": 42,  # seed
            "n_jobs": -1,  # uses all CPU cores to speed up training
        },
    }

    results_df = run_backtest_STABLE(
        data_with_features.copy(),
        feature_cols,
        BACKTEST_SIGNAL_START_DATE,
        NUM_LEAD,
        hmm_n_past_years_data=3,
        rf_n_past_months_data=10,
        hmm_train_nth_week=2,
        rf_train_nth_week=1,
        thresh_prob=0.75,
        params=params_opt,
    )

    plot_regimes_from_df(results_df, raw_data, prob_threshold=0.5)
    results_to_plot = results_df[
        results_df.index >= pd.to_datetime(BACKTEST_SIGNAL_START_DATE)
    ].copy()
    plot_results(results_to_plot, TICKER)
    compute_perf_stats(results_to_plot)
