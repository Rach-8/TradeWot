# regime_hmm_train.py

from __future__ import print_function
import pickle
import warnings
from hmmlearn.hmm import GaussianHMM
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import numpy as np
import pandas as pd

def plot_hidden_states(hmm_model, df, returns_col="Adj Close"):
    """
    Plot adjusted closing prices masked by hidden states.
    """
    hidden_states = hmm_model.predict(np.column_stack([df["Returns"]]))
    fig, axs = plt.subplots(hmm_model.n_components, sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, hmm_model.n_components))

    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax.plot_date(df.index[mask], df[returns_col][mask], ".", linestyle="none", c=colour)
        ax.set_title(f"Hidden State #{i}")
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)
    plt.show()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")


    csv_filepath = "Data/SPY.csv"
    pickle_path = "Models/hmm_model_spy.pkl"


    train_start = pd.Timestamp("1993-01-29")
    train_end   = pd.Timestamp("2014-12-31")
    test_start  = pd.Timestamp("2015-01-01")
    test_end    = pd.Timestamp("2020-12-31")


    df = pd.read_csv(
        csv_filepath,
        header=0,
        names=["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
    )
    df["Date"] = pd.to_datetime(df["Date"])

 
    train_df = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].copy()
    train_df.set_index("Date", inplace=True)
    train_df["Returns"] = train_df["Adj Close"].pct_change()
    train_df.dropna(inplace=True)
    rets_train = np.column_stack([train_df["Returns"]])


    hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
    hmm_model.fit(rets_train)
    print("Train Model Score:", hmm_model.score(rets_train))


    test_df = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)].copy()
    test_df.set_index("Date", inplace=True)
    test_df["Returns"] = test_df["Adj Close"].pct_change()
    test_df.dropna(inplace=True)


    test_df["HiddenState"] = hmm_model.predict(np.column_stack([test_df["Returns"]]))


    plot_hidden_states(hmm_model, test_df)

  
    print("Pickling HMM model...")
    pickle.dump(hmm_model, open(pickle_path, "wb"))
    print("...HMM model pickled.")
