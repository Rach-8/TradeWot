# parameter_optimization.py
import warnings
import pandas as pd
import numpy as np
import optuna

from run_backtest import get_data, engineer_features_SPEC, run_backtest_STABLE


# ------------------------------
# Utility function
# ------------------------------
def compute_strategy_returns(df):
    """
    Convert signals to strategy returns.
    Assumes 'returns' column exists and 'signal' column has trading signals.
    """
    df = df.copy()
    df["strategy_returns"] = df["signal"].shift(1) * df["returns"]
    df.dropna(subset=["strategy_returns"], inplace=True)
    return df["strategy_returns"]


# ------------------------------
# Constants
# ------------------------------
TICKER = "SPY"
START_DATE = "1993-01-01"
BACKTEST_SIGNAL_START_DATE = "2020-01-12"
END_DATE = "2023-06-12"
NUM_LEAD = 1

# ------------------------------
# Load data and engineer features
# ------------------------------
raw_data = get_data(TICKER, START_DATE, END_DATE)
data_with_features, feature_cols = engineer_features_SPEC(raw_data.copy(), NUM_LEAD)


# ------------------------------
# Optuna objective function
# ------------------------------
def objective(trial):
    try:
        # HMM parameters
        hmm_params = {
            "n_components": 2,
            "covariance_type": "full",
            "n_iter": trial.suggest_int("hmm_n_iter", 200, 800),
            "tol": trial.suggest_float("hmm_tol", 1e-4, 1e-2, log=True),
            "random_state": 42,
        }

        # Random Forest parameters
        rf_params = {
            "n_estimators": trial.suggest_int("rf_trees", 100, 500),
            "max_depth": trial.suggest_int("rf_depth", 2, 8),
            "min_samples_leaf": trial.suggest_int("rf_min_leaf", 5, 50),
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
        }

        params = {"hmm_params": hmm_params, "rf_params": rf_params}

        # Backtest-specific parameters
        hmm_n_past_years_data = trial.suggest_float("hmm_years", 2.0, 8.0)
        rf_n_past_months_data = trial.suggest_int("rf_months", 1, 12)
        hmm_train_nth_week = trial.suggest_int(
            "hmm_freq",
            1,
        )
        rf_train_nth_week = trial.suggest_int("rf_freq", 1, 4)
        thresh1 = trial.suggest_float("thresh1", 0.05, 0.5)
        thresh2 = trial.suggest_float("thresh2", 0.05, 0.75)
        hmm_lookback = trial.suggest_int("hmm_lookback", 1, 5)

        # Run backtest
        results = run_backtest_STABLE(
            data_with_features.copy(),
            feature_cols,
            BACKTEST_SIGNAL_START_DATE,
            NUM_LEAD,
            hmm_n_past_years_data=hmm_n_past_years_data,
            rf_n_past_months_data=rf_n_past_months_data,
            hmm_train_nth_week=hmm_train_nth_week,
            rf_train_nth_week=rf_train_nth_week,
            thresh1=thresh1,
            thresh2=thresh2,
            hmm_lookback=hmm_lookback,
            params=params,
        )

        # Compute strategy returns
        strategy_returns = compute_strategy_returns(results)

        # Skip trial if returns contain inf or NaN
        if strategy_returns.isnull().any() or np.isinf(strategy_returns).any():
            raise optuna.TrialPruned()

        # Compute CAGR
        cum_returns = (1 + strategy_returns).cumprod()
        total_years = len(strategy_returns) / 252
        cagr = cum_returns.iloc[-1] ** (1 / total_years) - 1

        # Compute Max Drawdown
        max_dd = (cum_returns / cum_returns.cummax() - 1).min()

        # Skip trial if max_dd is 0 to avoid division by zero
        if max_dd == 0:
            raise optuna.TrialPruned()

    except Exception:
        # Skip this trial if any error occurs
        raise optuna.TrialPruned()

    # Objective: maximize risk-adjusted return
    return cagr


# ------------------------------
# Run the optimization
# ------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, n_jobs=-1)

    print("Best parameters:", study.best_params)
