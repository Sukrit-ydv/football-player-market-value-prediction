import pandas as pd

def getyear(date) :

    if date.month >= 7:
        return date.year
    else:
        return date.year-1

data=pd.read_csv("data/updated/player_seasonal_stats.csv")
value=pd.read_csv("data/raw/player_valuations.csv")
value=value.drop(columns=["current_club_name","current_club_id","player_club_domestic_competition_id"])
#print(value.columns)
value["date"] = pd.to_datetime(value["date"])
value["season"]=value["date"].apply(getyear)
value=value.groupby(["season","player_id"]).agg(
    target_value_in_eur=("market_value_in_eur","last")
).reset_index()
#print(value.columns)
value=value.sort_values(["player_id","season"])
value["market_value_in_eur"]=value.groupby("player_id")["target_value_in_eur"].shift(1)
data=data.merge(value,on=["player_id","season"],how="left")
#print(data.columns)
#print(data.head)
#print(data[data["player_name"]=="Raphinha"])
#data.to_csv("data/seasonal_stats_with_market_value.csv",index=False)
data=data.sort_values(["player_id","season"])
training_data=data.dropna(subset=["market_value_in_eur","target_value_in_eur"]).copy()
#print(data[data["player_name"]=="Raphinha"])
print(training_data[training_data["player_name"]=="Raphinha"])
training_data.to_csv("data/training/training_data.csv",index=False)