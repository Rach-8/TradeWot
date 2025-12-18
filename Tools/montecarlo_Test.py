# monte_carlo_backtest.py
import pandas as pd
import numpy as np
from run_backtest import run_backtest_STABLE
import warnings

warnings.filterwarnings("ignore")


def monte_carlo_robustness_test(run_backtest_func, n_simulations=10, **kwargs):
    """
    Monte Carlo robustness test for overfitting.
    Runs the backtest function multiple times and evaluates sensitivity to randomness.

    Parameters
    ----------
    run_backtest_func : function
        Your backtest function (run_backtest_STABLE)
    n_simulations : int
        Number of Monte Carlo runs.
    kwargs : dict
        Arguments to pass to run_backtest_STABLE

    Returns
    -------
    summary_df : pd.DataFrame
        Summary of key metrics for each simulation.
    robustness_metrics : dict
        Ratios of worst-case to mean for key stats to assess overfitting.
    """
    results_list = []

    for sim in range(n_simulations):
        print(f"\n--- Monte Carlo Run {sim+1}/{n_simulations} ---")
        results_df = run_backtest_func(**kwargs)

        bt_results = results_df[
            results_df.index >= pd.to_datetime(kwargs["backtest_signal_start_date_str"])
        ]
        portfolio_returns = bt_results["signal"].fillna(0) * bt_results["returns"]

        cum_return = (1 + portfolio_returns).prod() - 1
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = (
            portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            if portfolio_returns.std() > 0
            else 0
        )
        running_max = (1 + portfolio_returns).cumprod().cummax()
        drawdown = (1 + portfolio_returns).cumprod() / running_max - 1
        max_dd = drawdown.min()

        print(
            f"Cumulative Return: {cum_return:.4f}, Sharpe: {sharpe:.4f}, Max Drawdown: {max_dd:.4f}"
        )

        results_list.append(
            {
                "Simulation": sim + 1,
                "CumulativeReturn": cum_return,
                "AnnualVol": annual_vol,
                "Sharpe": sharpe,
                "MaxDrawdown": max_dd,
            }
        )

    summary_df = pd.DataFrame(results_list)
    print("\n--- Monte Carlo Summary ---")
    print(summary_df.describe())

    # --- Robustness / overfitting metrics ---
    robustness_metrics = {
        "CumulativeReturn_Worst_to_Mean": summary_df["CumulativeReturn"].min()
        / summary_df["CumulativeReturn"].mean(),
        "Sharpe_Worst_to_Mean": summary_df["Sharpe"].min()
        / summary_df["Sharpe"].mean(),
        "MaxDrawdown_Worst_to_Mean": summary_df["MaxDrawdown"].min()
        / summary_df["MaxDrawdown"].mean(),
    }

    print("\n--- Robustness Metrics ---")
    for k, v in robustness_metrics.items():
        print(f"{k}: {v:.3f}")

    return summary_df, robustness_metrics


if __name__ == "__main__":
    from run_backtest import (
        get_data,
        engineer_features_SPEC,
    )  # <-- replace with your actual module

    TICKER = "SPY"
    TICKER2 = "GLD"
    START_DATE = "1993-01-01"
    END_DATE = "2024-01-12"
    BACKTEST_SIGNAL_START_DATE = "2022-01-12"
    NUM_LEAD = 1

    # Prepare data and features
    raw_data = get_data(TICKER, TICKER2, START_DATE, END_DATE)
    data_with_features, feature_cols = engineer_features_SPEC(raw_data.copy(), NUM_LEAD)

    print(f"Data prepared. Number of features: {len(feature_cols)}")
    print(f"Data shape after preprocessing: {data_with_features.shape}")

    # Define HMM + RF parameters (without fixed random_state to test robustness)
    params_opt = {
        "hmm_params": {
            "n_components": 2,
            "covariance_type": "full",
            "n_iter": 591,
            "tol": 0.00025461513333633457,
        },
        "rf_params": {
            "n_estimators": 300,
            "max_depth": 4,
            "min_samples_leaf": 30,
            "max_features": "sqrt",
            "bootstrap": True,
            "n_jobs": -1,
        },
    }

    # Run Monte Carlo robustness test
    mc_results, robustness = monte_carlo_robustness_test(
        run_backtest_STABLE,
        n_simulations=30,
        data_df=data_with_features.copy(),
        feature_list=feature_cols,
        backtest_signal_start_date_str=BACKTEST_SIGNAL_START_DATE,
        num_lead=NUM_LEAD,
        hmm_n_past_years_data=3,
        rf_n_past_months_data=24,
        hmm_train_nth_week=4,
        rf_train_nth_week=2,
        thresh_prob=0.75,
        params=params_opt,
    )
