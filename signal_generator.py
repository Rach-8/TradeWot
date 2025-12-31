# use snapshot of the trained model to predict from 19th december to now, 
# every day and log its rankings, then filter rankings based on abs ou_z threshold 
# and then generate signals from those filtered trades, which stock to short and to 
# long from the pair and at what ratio using the kf beta ratio


# TO TRADE I NEED : 
# Stock A    long or short : 1$             Take Profit = price(A) at Mean=0,    Stop Loss : price(A) at Z-score 25% outside band 
# Stock B    long or short : 1$*Beta(t)     Take Profit = price(B) at Mean=0,    Stop Loss : price(B) at Z-score 25% outside band 

#also run feature ranking on the xgboost model and only keep the important features
# see if retraining the model makes the metrics better