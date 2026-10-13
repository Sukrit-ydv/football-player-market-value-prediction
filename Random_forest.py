import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import joblib

dataset = pd.read_csv("data/training/final_training_data.csv")

train=dataset[dataset["season"]<=2024]
test=dataset[dataset["season"]==2025]
X_train=train[["appearances","yellow_cards","red_cards","goals","assists","minutes_played","age","market_value_in_eur"]]
Y_train=train["target_value_in_eur"]
X_test=test[["appearances","yellow_cards","red_cards","goals","assists","minutes_played","age","market_value_in_eur"]]
Y_test=test["target_value_in_eur"]

model=RandomForestRegressor(
    criterion="squared_error",
    n_jobs=-1,
    random_state=10,
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5
)

model.fit(X_train,Y_train)
training_data_pred=model.predict(X_train)
r2=r2_score(Y_train,training_data_pred)
m=mean_absolute_error(Y_train,training_data_pred)
print(f"for training data: \n r2={r2} , mean_absolute_error={m}")
test_data_pred=model.predict(X_test)
r2t=r2_score(Y_test,test_data_pred)
mt=mean_absolute_error(Y_test,test_data_pred)
print(f"for testing data: \n r2={r2t} , mean_absolute_error={mt}")

results=test[["player_name","season","appearances","goals","assists","age","market_value_in_eur","target_value_in_eur","minutes_played","yellow_cards","red_cards"]].copy()
results['predictions']=(test_data_pred/1000000).round() * 1000000
results["evaluation_difference"]=results["predictions"]-results["target_value_in_eur"]
results.to_csv("data/final/Random_forest_predictions.csv",index=False)
joblib.dump(model,"Models/random_forests_model.pkl")

player=input("Enter player name: ")
scoutingreport=results[results["player_name"]==player]
if scoutingreport.empty:
    print("No such player could be found")
else:
    row=scoutingreport.iloc[0]
    print(f"Player Name = {row["player_name"]}")
    print(f"Goals = {row["goals"]}")
    print(f"Assists = {row["assists"]}")
    print(f"Market Value at the season start = €{row["market_value_in_eur"]:,.2f}")
    print(f"Predic  ted Market Value at season end = €{row["predictions"]:,.2f}")
    print(f"Actual Market Value at season end = €{row["target_value_in_eur"]:,.2f}")
    print(f"Difference in prediction and actual value = €{row["evaluation_difference"]:,.2f}")

