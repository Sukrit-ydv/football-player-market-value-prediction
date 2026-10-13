import pandas as pd

apps=pd.read_csv("data/updated/updated_appearances.csv")
games=pd.read_csv("data/updated/updatedgames.csv")
value=pd.read_csv("data/updated/player_valuations.csv")
players=pd.read_csv("data/updated/updated_players.csv")
apps=apps.merge(games, on="game_id")
apps=apps.merge(players,on="player_id")
apps=apps.groupby(["player_id","season"]).agg(
    player_name=("player_name","first"),
    position=("position","first"),
    date_of_birth=("date_of_birth","first"),
    appearances=("game_id","count"),
    yellow_cards=("yellow_cards","sum"),
    red_cards=("red_cards","sum"),
    goals=("goals","sum"),
    assists=("assists","sum"),
    minutes_played=("minutes_played","sum")
).reset_index()
apps["date_of_birth"]=pd.to_datetime(apps["date_of_birth"])
apps["age"]=apps["season"]-apps["date_of_birth"].dt.year
apps=apps.drop(columns=["date_of_birth"])
print(apps.columns)
print(apps.head)
apps.to_csv("data/player_seasonal_stats.csv",index=False)
