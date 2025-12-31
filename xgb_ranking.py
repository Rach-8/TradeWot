import pandas as pd
import numpy as np
from Tools.helper_functions import *
from xgboost import XGBRanker


pairs = build_pairs_from_selection_csv("Data/selected_pairs.csv")
print(len(pairs))

price_df = pd.read_csv("Data/Raw_Data/prices.csv", index_col=0, parse_dates=True)


price_df = price_df.apply(pd.to_numeric, errors="coerce")
df, feature_cols = build_pair_dataset(
    price_df, pairs, "2024-04-15", "2025-12-31", N=5, N_mult=2
)
df.to_csv("Data/Raw_Data/features.csv")


model = None
train_window = 100
test_window = 20


predictions = []
true_values = []
dates = []
pair_ids = []
X = df[feature_cols]
y = df["target"]
dates_unique = df.index.unique().sort_values()


for day in range(0, len(dates_unique) - train_window - test_window, test_window):

    train_dates = dates_unique[day : day + train_window]
    test_dates = dates_unique[day + train_window : day + train_window + test_window]

    train_mask = df.index.isin(train_dates)
    test_mask = df.index.isin(test_dates)

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    # Skip if no data
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"[SKIP] Empty train/test for window starting {train_dates[0]}")
        continue

    # Skip if train has less than 2 classes
    if y_train.nunique() < 2:
        print(
            f"[SKIP] Not enough classes in train for window starting {train_dates[0]}"
        )
        continue

    # Skip if test has no samples
    if len(y_test) == 0:
        print(f"[SKIP] Empty test for window starting {train_dates[0]}")
        continue

    # Skip if test has only one class (ROC AUC would fail)
    test_has_two_classes = y_test.nunique() > 1

    train_group = X_train.groupby(X_train.index).size().to_list()

    model = XGBRanker(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="rank:pairwise",
        random_state=42,
    )

    model.fit(X_train, y_train, group=train_group)

    y_pred_score = model.predict(X_test)

    predictions.extend(y_pred_score)
    true_values.extend(y_test.values)
    dates.extend(X_test.index)
    pair_ids.extend(df.loc[test_mask, "Y"] + "|" + df.loc[test_mask, "X"])

    # Optional: warn if test has only one class
    if not test_has_two_classes:
        print(
            f"[WARN] Test window starting {train_dates[0]} has only one class. ROC AUC will be NaN."
        )

# ============================
# FINAL LIVE PREDICTION BLOCK
# ============================

# Index where the last full test window ended
last_day_used = day + train_window + test_window
remaining_dates = dates_unique[last_day_used:]

if len(remaining_dates) > 0:
    print(f"[LIVE] Predicting remaining {len(remaining_dates)} days")

    # Use the most recent fully-observed training window
    train_dates = dates_unique[last_day_used - train_window : last_day_used]

    # Training data must have valid targets
    train_mask = df.index.isin(train_dates) & y.notna()
    live_mask = df.index.isin(remaining_dates)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_live = X.loc[live_mask]

    if len(X_train) > 0 and y_train.nunique() >= 2 and len(X_live) > 0:

        train_group = X_train.groupby(X_train.index).size().to_list()

        model = XGBRanker(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective="rank:pairwise",
            random_state=42,
        )

        model.fit(X_train, y_train, group=train_group)

        live_scores = model.predict(X_live)

        predictions.extend(live_scores)
        true_values.extend([np.nan] * len(live_scores))  # forward / unlabeled
        dates.extend(X_live.index)
        pair_ids.extend(df.loc[live_mask, "Y"] + "|" + df.loc[live_mask, "X"])

y_true = np.array(true_values)
y_pred_score = np.array(predictions)


results_df = pd.DataFrame(
    {
        "Date": dates,
        "Pair": pair_ids,
        "Pred_Score": y_pred_score,
        "Target": y_true,
    }
)
results_df = results_df.sort_values(["Date", "Pred_Score"], ascending=[True, False])
results_df["Rank"] = results_df.groupby("Date")["Pred_Score"].rank(
    ascending=False, method="first"
)



for k in [1, 3, 5, 10]:
    print(f"Top-{k} mean target: {top_k_mean_target(results_df, k):.4f}")
baseline = results_df["Target"].mean()
print(f"Baseline target mean: {baseline:.4f}")
for k in [1, 3, 5]:
    topk = top_k_mean_target(results_df, k)
    lift = topk / baseline
    print(f"Top-{k} lift: {lift:.2f}x")
ic_mean, ic_std = daily_spearman(results_df)
print(f"Daily Spearman IC: {ic_mean:.4f} ± {ic_std:.4f}")
rank_profile = results_df.groupby("Rank")["Target"].mean().loc[lambda x: x.index <= 5]
print(rank_profile)


df_reset = df.reset_index().rename(columns={"index": "Date"})
results_df[["Y", "X"]] = results_df["Pair"].str.rsplit("|", n=1, expand=True)
eval_df = results_df.merge(
    df_reset[["Date", "Y", "X", "spread_centered","ou_z", "upper", "lower", "beta_kf"]],
    on=["Date", "Y", "X"],
    how="left",
)

N_rank = 20

eval_df = eval_df[eval_df["Rank"] <= N_rank]
eval_df = eval_df.drop(columns=["Pair", "Pred_Score", "Target"])
eval_df = eval_df.set_index("Date").sort_index()
eval_df.to_csv("Data/ranked_pairs.csv")



extreme_df = eval_df[
    (abs(eval_df["ou_z"]) >= 2)
].copy()
extreme_df = extreme_df.sort_index()
extreme_df.to_csv("Data/trades.csv")




# output a signals df and call backtesting function on it

pairs_list = list(
    eval_df
    .iloc[-N_rank:][["Y", "X"]]
    .drop_duplicates()
    .itertuples(index=False, name=None)
)
#plot(df,pairs_list,2)


def percent_reverted_by_rank(
    eval_df,
    full_df,
    horizon=5,
    mean_tol=0.0,
):
    eval_df = eval_df.copy()
    full_df = full_df.copy()

    # Handle Date as index or column
    if "Date" not in eval_df.columns:
        eval_df = eval_df.reset_index()

    if "Date" not in full_df.columns:
        full_df = full_df.reset_index()

    eval_df["Date"] = pd.to_datetime(eval_df["Date"])
    full_df["Date"] = pd.to_datetime(full_df["Date"])

    full_df = full_df.sort_values("Date")

    results = []

    for _, row in eval_df.iterrows():
        date = row["Date"]
        rank = row["Rank"]
        y = row["Y"]
        x = row["X"]
        z0 = row["ou_z"]

        pair_ts = full_df[
            (full_df["Y"] == y) &
            (full_df["X"] == x)
        ].set_index("Date")

        if date not in pair_ts.index:
            continue

        future = pair_ts.loc[date:].iloc[1:horizon + 1]["ou_z"]

        if future.empty:
            continue

        if mean_tol == 0:
            reverted = np.any(np.sign(future) != np.sign(z0))
        else:
            reverted = np.any(np.abs(future) <= mean_tol)

        results.append({"Rank": rank, "reverted": int(reverted)})

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return pd.DataFrame(columns=["Rank", "n_signals", "pct_reverted"])

    summary = (
        results_df
        .groupby("Rank")
        .agg(
            n_signals=("reverted", "count"),
            pct_reverted=("reverted", "mean"),
        )
        .reset_index()
    )

    summary["pct_reverted"] *= 100
    return summary


summary = percent_reverted_by_rank(
    eval_df=eval_df.reset_index(),   # Date is index
    full_df=df_reset,
    horizon=5
)
