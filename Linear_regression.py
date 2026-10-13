import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import joblib

dataset=pd.read_csv("data/training/final_training_data.csv")
#print(dataset.columns)
train=dataset[dataset["season"]<=2024]
test=dataset[dataset["season"]==2025]
X_train=train[["appearances","yellow_cards","red_cards","goals","assists","minutes_played","age","market_value_in_eur"]]
Y_train=train["target_value_in_eur"]
X_test=test[["appearances","yellow_cards","red_cards","goals","assists","minutes_played","age","market_value_in_eur"]]
Y_test=test["target_value_in_eur"]
reg=linear_model.LinearRegression()
reg.fit(X_train,Y_train)
joblib.dump(reg,"Models/Linear_regression_model.pkl")
#print(reg.coef_)
#print(reg.intercept_)
predections=reg.predict(X_test)
mae=mean_absolute_error(Y_test,predections)
r2=r2_score(Y_test,predections)
print(f"mae = {mae}, r2 = {r2}")
results=test[["player_name","season","appearances","goals","assists","age","market_value_in_eur","target_value_in_eur"]].copy()
results["predictions"]=(predections/1000000).round() * 1000000
results["evaluation_difference"]=results["predictions"]-results["target_value_in_eur"]
results.to_csv("data/final/regression_predictions.csv",index=False)




