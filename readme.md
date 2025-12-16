
I was looking at the different large market indexes, when I noticed that market was either in sideways short choppy pattern or in a medium to long term trending pattern.

Used Backtesting Python library to test different strategies, some were really good however they only worked in a trending upward markets and would incur heavy losses in choppy volatile times.

Then after alot mistakes and backtesting I had two final modularized trade strategies :
Trending Strategy : Uses bollinger bands and 200 sma. Creates stable profits.                  
Sideways Strategy : Uses ATR multipliers, ADX theresholds, RSI bands. Plays safe, protects capital.

                    

img

Just Trending Strategy from 1993 to 2025, 
Return [%]                            3200
Buy & Hold Return [%]                 2500

But, we are incurring heavy losses in the choppy sideways parts of the market, as seen in red.

img

Sideways Strategy applied to one of the recent most unpredictable sideways markets.
Return [%]                            5.08816
Buy & Hold Return [%]                 1.50549

So in a scenario where Trending strategy is losing large amount of capital, Sideways Strategy can even profit some.


Now the most important part would be to be able to predict if the market is sideways or trending.
It does not matter how perfect the strategies are if they cannot be deployed in thier best eniroment.

Using mechanical methods with triggers was not enough to predict the regime of the market, as it depended on many latent and hidden variables.
But, there happens to be very specific types of ML Models that are regulary used by quantitatives. 

They are called HMMs - Hidden Markov Models. 
They by design assume that latent variables and states exist which are not observable directly. Thus when trained on a large dataset of market returns, as those returns were indirectly affected by the regime changes, fitting an HHM to the market return data allows for regime predictions.


img

The model trained on 1993 to 2015 data predicting using untested data.

Currently I am working on integration my strategies with its predictions, then next step is the backtesting.



def engineer_features_SPEC(data_df, num_lead):
    """RF-friendly feature engineering: keep raw features + add pct_change for non-stationary ones."""
    
    data = data_df.copy()
    
    # --- Basic price/volume features ---
    data['Open-Close'] = (data['Open'] - data['Close']) / data['Open']
    data['High-Low'] = (data['High'] - data['Low']) / data['Low']
    data['Return_1d'] = data['Close'].pct_change()
    data['std_5'] = data['Return_1d'].rolling(5).std()
    
    # VPT
    data["VPT"] = (1 + data["Return_1d"]) * data["Volume"]
    data["VPT_cum"] = data["VPT"].cumsum()
    data["VPT_Momentum"] = data["VPT_cum"].diff()
    data["VPT_Direction"] = np.sign(data["VPT_Momentum"])
    
    # Volume features
    data["Vol_MA20"] = data["Volume"].rolling(20).mean()
    data["Vol_Ratio"] = data["Volume"] / data["Vol_MA20"]
    data["VolRank_20"] = data["std_5"].rank(pct=True)
    data["Vol_Residual"] = data["Volume"] - data["Vol_MA20"]
    data["Vol_Trend"] = data["Vol_Ratio"] * np.sign(data["Close"].pct_change(5))
    
    # --- Volatility / ATR ---
    data["ATR"] = ta.volatility.AverageTrueRange(
        high=data["High"], low=data["Low"], close=data["Close"], window=14
    ).average_true_range()
    data["ATR_pct"] = data["ATR"] / data["Close"]
    
    # --- RSI / ROC ---
    data["RSI_14"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
    data["ROC_5"] = data["Close"].pct_change(5)
    
    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(data["Close"], window=20, window_dev=2)
    data["BB_Pct"] = (data["Close"] - bb.bollinger_mavg()) / (bb.bollinger_hband() - bb.bollinger_lband())
    data["BB_width"] = bb.bollinger_hband() - bb.bollinger_lband()
    
    # --- EMA slopes ---
    data["EMA10"] = data["Close"].ewm(span=10).mean()
    data["EMA20"] = data["Close"].ewm(span=20).mean()
    data["EMA50"] = data["Close"].ewm(span=50).mean()
    data["EMA100"] = data["Close"].ewm(span=100).mean()
    data["EMA_Slope"] = data["EMA20"] - data["EMA50"]
    data["EMA_Slope_Long"] = data["EMA20"] - data["EMA100"]
    
    # --- ADX ---
    data["ADX_14"] = ta.trend.ADXIndicator(data["High"], data["Low"], data["Close"], window=14).adx()
    data["ADX_Slope"] = data["ADX_14"].diff()
    
    # --- MACD histogram ---
    macd = ta.trend.MACD(data["Close"])
    data["MACD_Hist"] = macd.macd_diff()
    
    # --- Stochastic oscillator ---
    stoch = ta.momentum.StochasticOscillator(data["High"], data["Low"], data["Close"], window=14)
    data["Stoch_K"] = stoch.stoch()
    
    # --- Other features ---
    data["Close_Z20"] = (data["Close"] - data["Close"].rolling(20).mean()) / data["Close"].rolling(20).std()
    data["Ret_3"] = data["Close"].pct_change(3)
    data["Ret_10"] = data["Close"].pct_change(10)
    
    # --- Raw feature list ---
    raw_features = [
        'Open-Close', 'High-Low', 'std_5', 'Vol_Ratio', 'VPT_Momentum',
        'Return_1d', 'ATR_pct', 'RSI_14', 'ROC_5', 'BB_Pct',
        'EMA_Slope', 'ADX_14', 'VolRank_20', 'Vol_Residual', 'VPT_Direction',
        'EMA_Slope_Long', 'ADX_Slope', 'MACD_Hist', 'Vol_Trend', 'Ret_10',
        'BB_width', 'Stoch_K', 'Close_Z20', 'Ret_3'
    ]
    
    # --- Hybrid stationarity: add pct_change features ---
    pct_features = []
    for feature in raw_features:
        series = data[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 20 or series.nunique() <= 1:
            continue
        try:
            pvalue = adfuller(series, regression='c', autolag='AIC')[1]
            if pvalue > 0.05:
                pct_feature_name = feature + "_pct"
                data[pct_feature_name] = series.pct_change()
                pct_features.append(pct_feature_name)
        except:
            # Skip if ADF fails
            continue
    
    # --- Final feature list ---
    final_feature_columns = raw_features + pct_features
    
    # --- Target ---
    data['y_signal'] = np.where(data['returns'].shift(-num_lead) > 0, 1, 0)
    
    return data, final_feature_columns

def run_backtest__custAI_params(data_df, feature_list, backtest_signal_start_date_str, params):
    """
    Runs the event-driven backtest with optional N-week retraining.
    All hyperparameters for HMM and RF, as well as backtest parameters, are passed via 'params' dict.
    """
    # --- Extract backtest parameters ---
    num_lead = params.get("num_lead", 1)
    threshold = params.get("threshold", 0.03)
    n_year_window = params.get("n_year_window", 4)
    retrain_every_n_weeks = params.get("retrain_every_n_weeks", 1)
    window_size = n_year_window * 252
    train_test_split_days = params.get("train_test_split_days", 1)
    min_samples_for_rf = params.get("min_samples_for_rf", 30)
    retrain_days = retrain_every_n_weeks * 7

    # HMM and RF hyperparameters
    hmm_params = params.get("hmm_params", {})
    rf_params = params.get("rf_params", {})

    # --- Initialize signal column ---
    data_df['signal'] = 0.0

    # --- Determine start index ---
    target_signal_start_datetime = pd.to_datetime(backtest_signal_start_date_str)
    dates_after_target = data_df.index[data_df.index >= target_signal_start_datetime]
    if dates_after_target.empty:
        print(f"Target signal start date {backtest_signal_start_date_str} is after the dataset ends.")
        return data_df

    first_target_date_idx = data_df.index.get_loc(dates_after_target[0])
    loop_start_index = max(window_size, first_target_date_idx)
    print(f"Backtest will start from: {data_df.index[loop_start_index].strftime('%Y-%m-%d')}")

    # --- Stored models ---
    hmm_model = None
    model0 = None
    model1 = None
    last_retrain_date = None

    # --- Main loop ---
    for t in range(loop_start_index, len(data_df)):
        current_date = data_df.index[t]
        data_sample = data_df.iloc[t - window_size : t].copy()
        if len(data_sample) < window_size:
            continue

        # Decide whether retraining is needed
        need_retrain = last_retrain_date is None or (current_date - last_retrain_date).days >= retrain_days

        if need_retrain:
            last_retrain_date = current_date
            model_training_data = data_sample.iloc[:-train_test_split_days].copy()

            if len(model_training_data) < (min_samples_for_rf * 2):
                hmm_model = None
                model0 = None
                model1 = None
            else:
                # --- HMM Training ---
                hmm_train_features = model_training_data[['returns']].copy().dropna()
                if len(hmm_train_features) >= min_samples_for_rf:
                    hmm_model = hmm.GaussianHMM(**hmm_params)
                    hmm_model.fit(hmm_train_features)
                    regimes = hmm_model.predict(hmm_train_features)
                    model_training_data.loc[hmm_train_features.index, 'regime'] = regimes
                else:
                    hmm_model = None

                # --- RF Training ---
                model0, model1 = None, None
                if 'regime' in model_training_data.columns and feature_list:
                    reg0 = model_training_data[model_training_data['regime'] == 0]
                    reg1 = model_training_data[model_training_data['regime'] == 1]

                    clf = RandomForestClassifier(**rf_params)

                    # RF 0
                    if len(reg0) >= min_samples_for_rf and len(reg0['y_signal'].unique()) > 1:
                        X0 = reg0[feature_list].iloc[:-num_lead, :]
                        y0 = reg0['y_signal'].iloc[:-num_lead]
                        model0 = clf.fit(X0, y0)

                    # RF 1
                    if len(reg1) >= min_samples_for_rf and len(reg1['y_signal'].unique()) > 1:
                        X1 = reg1[feature_list].iloc[:-num_lead, :]
                        y1 = reg1['y_signal'].iloc[:-num_lead]
                        model1 = clf.fit(X1, y1)

        # --- Daily prediction ---
        features_for_pred = data_sample[feature_list].iloc[-num_lead:] if feature_list else pd.DataFrame()
        hmm_pred_features = data_sample[['returns']].iloc[-1:]
        next_day_probs = np.array([0.5, 0.5])

        if hmm_model and hmm_pred_features.notnull().all().all():
            last_state_probs = hmm_model.predict_proba(hmm_pred_features)[0]
            next_day_probs = last_state_probs @ hmm_model.transmat_

        signal0 = model0.predict_proba(features_for_pred)[0][1] if model0 else 0.0
        signal1 = model1.predict_proba(features_for_pred)[0][1] if model1 else 0.0

        if next_day_probs[0] > next_day_probs[1]:
            final_signal = signal0
        elif next_day_probs[1] > next_day_probs[0]:
            final_signal = signal1
        else:
            final_signal = 0.0

        # Threshold to get discrete signal
        data_df.loc[current_date, 'signal'] = (
            1 if final_signal > (0.5 + threshold) else
            -1 if final_signal < (0.5 - threshold) else
            0
        )

    return data_df
