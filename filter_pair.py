# selection.py
import pandas as pd
import numpy as np
from Tools.sector_maps import *
from Tools.helper_functions import *

# Control Panel
train_start, train_end = "2022-12-31", "2025-12-31"
sm = sector_nasdaq_tsx
sigma_mult = 2
max_half_life = 5

# Download data
prices = load_or_update_prices(
    sector_map=sm,
    start_date=train_start,
    price_path="Data/Raw_Data/prices.csv"
)


valid_tickers = [
    t for t in prices.columns
    if prices[t].first_valid_index() <= pd.to_datetime(train_start)
]

prices = prices[valid_tickers]
train_prices = prices.loc[train_start:train_end]
log_train = np.log(train_prices)
print(f"Training : {train_prices.index.min()} to {train_prices.index.max()}")



# Find Pairs
pairs_df = find_valid_pairs(log_train, sm)

# Train
train_results = for_each_pair(
    pairs_df,
    log_train,
    train_pair_callback,
    sigma_mult
)

train_results = [r for r in train_results if r is not None]

if not train_results:
    raise ValueError("No valid pairs passed training filters")

train_params_df = pd.DataFrame(train_results)


# Filter Logic
filtered_pairs_df = train_params_df[
    (train_params_df["half_life_days"] < max_half_life)
    & (train_params_df["beta_ols"].abs() < 2)
    & (train_params_df["ou_mu"] > 0.1)
    & (train_params_df["ou_std"] < 0.04)
    & ~(
        (train_params_df["beta_kf"].abs() > 1.2)
        & (train_params_df["alpha_kf"] > 0.15)
    )
]

# Stats
print("\n Filteration Stats : \n")
for col in filtered_pairs_df.select_dtypes(include="number").columns:
    print(
        f"{col}: min={filtered_pairs_df[col].min():.6f}, "
        f"max={filtered_pairs_df[col].max():.6f}, "
        f"mean={filtered_pairs_df[col].mean():.6f}"
    )
filtered_pairs_df.to_csv("Data/selected_pairs.csv", index=False)
print("Saved Filtered Pairs")

