# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
import yfinance as yf

import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM


def fit_transform_scaler(X_train_df, X_test_df):
    """
    Fits scaler on TRAIN data only and transforms both.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.values)
    X_test_scaled = scaler.transform(X_test_df.values)
    return X_train_scaled, X_test_scaled, scaler


def fit_transform_pca(X_train_scaled, X_test_scaled, variance_target):
    """
    Fits PCA on TRAIN data only, selects components by variance target.
    """
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cum_var >= variance_target) + 1

    X_train_final = X_train_pca[:, :n_components]
    X_test_final = pca.transform(X_test_scaled)[:, :n_components]

    return X_train_final, X_test_final, pca, n_components


def hmm_engineer_features(start_date, end_date):

    spx = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)
    vix = yf.download("^VIX", start=start_date, end=end_date, auto_adjust=True)
    vix3m = yf.download("^VIX3M", start=start_date, end=end_date, auto_adjust=True)
    vix6m = yf.download("^VIX6M", start=start_date, end=end_date, auto_adjust=True)
    russell = yf.download("^RUT", start=start_date, end=end_date, auto_adjust=True)
    hyg = yf.download("HYG", start=start_date, end=end_date, auto_adjust=True)
    lqd = yf.download("LQD", start=start_date, end=end_date, auto_adjust=True)

    # Flatten columns
    for df_ in [spx, vix, vix3m, vix6m, russell, hyg, lqd]:
        df_.columns = df_.columns.get_level_values(0)

    # Rename
    spx = spx.rename(columns={"Close": "spx_Close"})
    vix = vix.rename(columns={"Close": "vix_Close"})
    vix3m = vix3m.rename(columns={"Close": "vix3m_Close"})
    vix6m = vix6m.rename(columns={"Close": "vix6m_Close"})
    russell = russell.rename(columns={"Close": "rut_Close"})
    hyg = hyg.rename(columns={"Close": "HYG"})
    lqd = lqd.rename(columns={"Close": "LQD"})

    # =========================
    # Feature Engineering
    # =========================
    spx["SPX_50D_MA"] = spx["spx_Close"].rolling(50).mean()
    spx["SPX_200D_MA"] = spx["spx_Close"].rolling(200).mean()
    spx["SPX_Above50D"] = (spx["spx_Close"] > spx["SPX_50D_MA"]).astype(int)
    spx["SPX_Above200D"] = (spx["spx_Close"] > spx["SPX_200D_MA"]).astype(int)

    df = (
        spx[["spx_Close", "SPX_Above50D", "SPX_Above200D"]]
        .join(vix[["vix_Close"]])
        .join(vix3m[["vix3m_Close"]])
        .join(vix6m[["vix6m_Close"]])
        .join(russell[["rut_Close"]])
        .join(hyg[["HYG"]])
        .join(lqd[["LQD"]])
        .dropna()
    )

    df["Credit_Spread"] = np.log(df["HYG"]) - np.log(df["LQD"])

    # Returns
    df["SPX_Daily_Return"] = df["spx_Close"].pct_change()
    df["SPX_21D_Return"] = df["spx_Close"].pct_change(21)
    df["SPX_63D_Return"] = df["spx_Close"].pct_change(63)
    df["SPX_126D_Return"] = df["spx_Close"].pct_change(126)

    df["RUT_21D_Return"] = df["rut_Close"].pct_change(21)
    df["RUT_63D_Return"] = df["rut_Close"].pct_change(63)

    df["SPX_vs_RUT_21D"] = df["SPX_21D_Return"] - df["RUT_21D_Return"]
    df["SPX_vs_RUT_63D"] = df["SPX_63D_Return"] - df["RUT_63D_Return"]

    # Volatility
    def realized_vol(x, window=21, td=252):
        return x.rolling(window).std() * np.sqrt(td)

    df["SPX_21D_RealVol"] = realized_vol(df["SPX_Daily_Return"], 21)
    df["SPX_63D_RealVol"] = realized_vol(df["SPX_Daily_Return"], 63)

    # VIX features
    df["VIX_1D_Change"] = df["vix_Close"].diff(1)
    df["VIX_5D_Change"] = df["vix_Close"].diff(5)
    df["VIX_to_SPXRealVol"] = df["vix_Close"] / df["SPX_21D_RealVol"]
    df["VIX3M_VIX"] = df["vix3m_Close"] / df["vix_Close"]
    df["VIX6M_VIX"] = df["vix6m_Close"] / df["vix_Close"]

    df = df.dropna()
    df["date"] = df.index
    df = df.reset_index(drop=True)

    # =========================
    # Feature Selection
    # =========================
    exclude = {
        "date",
        "spx_Close",
        "rut_Close",
        "HYG",
        "LQD",
        "SPX_50D_MA",
        "SPX_200D_MA",
        "vix3m_Close",
        "vix6m_Close",
    }

    feature_cols = [c for c in df.select_dtypes(np.number).columns if c not in exclude]

    # print(f"PCA components used: {n_components}")

    return df, feature_cols


# ---------------------------------------------------


# HMM Control Panel

start_date = "2018-12-31"
end_date = "2025-12-15"
HMM_variance_target = 0.55
HMM_train_n_years = 5
HMM_retrain_n_weeks = 4


hmm_params = {
    "n_components": 2,
    "covariance_type": "diag",
    "tol": 1e-4,
    "n_iter": 600,
    "random_state": 42,
}

# ---------------------------------------------------


train_for_n_days = HMM_train_n_years * 252
retrain_hmm_every_n_days = HMM_retrain_n_weeks * 5
df, feature_cols = hmm_engineer_features(start_date, end_date)

# Output columns
df["Regime"] = pd.NA
df["Regime0_Prob"] = np.nan
df["Regime1_Prob"] = np.nan

n = len(df)
start_loc = df.index[df["date"] >= start_date][0]
i = start_loc

while i < n:

    train_idx = df.index[max(0, i - train_for_n_days) : i]
    test_idx = df.index[i : min(i + retrain_hmm_every_n_days, n)]

    X_train_df = (
        df.loc[train_idx, feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    )

    X_test_df = (
        df.loc[test_idx, feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    )

    if len(X_train_df) < 200 or len(X_test_df) == 0:
        i += retrain_hmm_every_n_days
        continue

    X_train_scaled, X_test_scaled, _ = fit_transform_scaler(X_train_df, X_test_df)

    X_train_final, X_test_final, _, _ = fit_transform_pca(
        X_train_scaled, X_test_scaled, HMM_variance_target
    )

    hmm = GaussianHMM(**hmm_params)

    hmm.fit(X_train_final)

    test_states = hmm.predict(X_test_final)
    test_probs = hmm.predict_proba(X_test_final)

    train_states = hmm.predict(X_train_final)

    tmp = df.loc[X_train_df.index].copy()
    tmp["Regime"] = train_states

    vol_by_regime = tmp.groupby("Regime")["SPX_21D_RealVol"].mean()

    order = vol_by_regime.sort_values().index
    mapping = {old: new for new, old in enumerate(order)}

    test_states = pd.Series(test_states, index=X_test_df.index).map(mapping)
    test_probs = test_probs[:, order]

    df.loc[X_test_df.index, "Regime"] = test_states.values
    df.loc[X_test_df.index, ["Regime0_Prob", "Regime1_Prob"]] = test_probs

    i += retrain_hmm_every_n_days


# =========================
# Plot (OUT-OF-SAMPLE)
# =========================


plot_df = df[df["Regime"].notna()].copy()
plot_df["Regime"] = plot_df["Regime"].astype(str)

# Define color mapping dynamically
unique_regimes = plot_df["Regime"].unique()
color_map = {"0": "green", "1": "red"}
if "2" in unique_regimes:
    color_map["2"] = "yellow"  # Add third regime if it exists


fig = px.scatter(
    plot_df,
    x="date",
    y="spx_Close",
    color="Regime",
    color_discrete_map=color_map,
    title="2-Regime HMM (Out-of-Sample, Walk-Forward, No Leakage)",
    labels={"spx_Close": "S&P 500"},
)

fig.update_traces(marker=dict(size=4))
fig.update_layout(template="plotly_dark")
fig.show()
