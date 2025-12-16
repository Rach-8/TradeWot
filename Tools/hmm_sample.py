import warnings
from hmmlearn.hmm import GaussianHMM
from matplotlib import cm, pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import yfinance


def get_data(ticker, start_date, end_date):
    """Downloads and prepares stock data."""
    data = yfinance.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        group_by="ticker"
    )[ticker]

    # Calculate features
    data["Returns"] = data["Close"].pct_change()
    data["Vol_20"] = data["Returns"].rolling(20).std()
    data["Vol_5"] = data["Returns"].rolling(5).std()
    data["Trend_Strength"] = data["Returns"].rolling(10).mean().abs()
    data["Noise_Ratio"] = data["Vol_5"] / data["Vol_20"]

    # Drop missing values
    data.dropna(inplace=True)
    
    return data

def plot_hidden_states(hmm_model, df, X, title):
    hidden_states = hmm_model.predict(X)
    
    # Compute stats
    stats = label_hmm_states(hmm_model, X)

    # Label the states based on the highest abs_mean (Trending) and lowest abs_mean (Sideways)
    trending_state = max(stats, key=lambda s: stats[s]["mean"])  # State with the highest abs_mean
    sideways_state = min(stats, key=lambda s: stats[s]["mean"])  # State with the lowest abs_mean
    state_labels = {trending_state: "Trending", sideways_state: "Sideways"}

    fig, axs = plt.subplots(
        hmm_model.n_components,
        sharex=True,
        figsize=(14, 6)
    )

    colors = cm.rainbow(np.linspace(0, 1, hmm_model.n_components))

    for i, (ax, c) in enumerate(zip(axs, colors)):
        mask = hidden_states == i
        ax.plot(df.index[mask], df["Close"][mask], ".", c=c)
        ax.set_title(f"{title} — State {i} ({state_labels.get(i, 'Unknown')})\n"
                     f"Mean={stats[i]['mean']}, Vol={stats[i]['vol']}")
        ax.grid(True)

    plt.show()

def label_hmm_states(hmm_model, X):
    hidden_states = hmm_model.predict(X)

    stats = {}
    for s in range(hmm_model.n_components):
        rets = X[hidden_states == s][:, 0]  # first column = returns
        stats[s] = {
            "mean": round(float(np.mean(rets)), 6),
            "vol": round(float(np.std(rets)), 6),
            "abs_mean": round(float(np.abs(np.mean(rets))), 6),
        }
    return stats

def normalize_data(df):
    """Normalize the feature columns using StandardScaler."""
    scaler = StandardScaler()
    
    # Select the features to normalize
    features = [
        "Returns", "Vol_20", "Trend_Strength", "Noise_Ratio"
    ]
    
    # Normalize the features
    df[features] = scaler.fit_transform(df[features])
    
    return df

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    df = get_data("SPY", "1993-01-01", "2025-12-12")
    df = normalize_data(df)

    # Extended training period to include a more diverse range of market conditions
    train_start = "2012-01-01" #always use 8 years of data for regime labeling
    train_end   = "2020-01-11"
    test_start  = "2020-01-12"
    test_end    = "2023-06-12"

    train_df = df.loc[train_start:train_end]
    test_df  = df.loc[test_start:test_end]

    # Improved feature selection
    X_train = np.column_stack([
        train_df["Returns"],
        #train_df["Vol_20"],
       #train_df["Trend_Strength"],  # Added feature
        #train_df["Noise_Ratio"],  # Added feature
    ])

    X_test = np.column_stack([
        test_df["Returns"],
       # test_df["Vol_20"],
       # test_df["Trend_Strength"],  # Added feature
       # test_df["Noise_Ratio"],  # Added feature
    ])

    # HMM model with adjusted parameters
    hmm_model = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        n_iter=591,  # Adjusted iterations to prevent overfitting
        tol=0.00025461513333633457,   # Adjusted tolerance
        random_state=42
    ).fit(X_train)

    stats = label_hmm_states(hmm_model, X_train)
    print("State stats:", stats)

    history = X_train.copy()
    next_day_states = []

    # Improved prediction with smoothing
    for i in range(len(X_test)):
        last_states = hmm_model.predict(history)[-5:]  # Use last 5 states for smoothing
        next_state = np.argmax(np.bincount(last_states))  # Most common state in the last 5
        next_day_states.append(next_state)

        history = np.vstack([history, X_test[i]])

    test_df = test_df.copy()
    test_df["Predicted_Regime"] = next_day_states
    counts = test_df["Predicted_Regime"].value_counts().sort_index()

    print(f"Regime 0 count: {counts.get(0, 0)}")
    print(f"Regime 1 count: {counts.get(1, 0)}")

    plot_hidden_states(hmm_model, test_df, X_test, "TEST")
