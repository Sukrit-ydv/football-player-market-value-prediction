import pandas as pd
import joblib

Rf_data=pd.read_csv("data/final/Random_forest_predictions.csv")
Lr_data=pd.read_csv("data/final/regression_predictions.csv")
Rf_model=joblib.load("Models/random_forests_model.pkl")
Lr_model=joblib.load("Models/Linear_regression_model.pkl")

choice=1
while(choice!=0) :
    print("------------------------\n0 = Quit Program \n1 = Print Random Forest Feature Importance Values\n2 = Player Stats Of Last Season\n3 = Display Predicted Player Valuation\n4 = Predict Value Using Custom Data\n5 = Display Full Player Stats\n6 = Players Valued Below Model Estimates (Can be used for scouting)\n7 = Top 10 Players By Predicted Market Value Increase\n8 = Top 10 Players by Predicted Market Value Decrease\n9 = Regression Model Stats vs Random Forest Stats\n10 = Visualizing The Random Forest Actual vs Predicted Market Values\n------------------------\n")
    choice=int(input("enter your choice: "))
    match choice:
        case 0:
            print("Bye")
        case 1:
            imp=Rf_model.feature_importances_
            features=[
                "Appearances",
                "Yellow cards",
                "Red cards",
                "Goals",
                "Assists",
                "Minutes played",
                "Age",
                "Market value in euros"
            ]
            result=pd.DataFrame({
                "Features":features,
                "Importance":imp
            })
            result=result.sort_values(by="Importance",ascending=False)
            print(result)
        case 2:
            player=input("Enter the players Full Name : ")
            stats_report=Rf_data[Rf_data["player_name"]==player]
            if stats_report.empty:
                print("No such player could be found")
            else:
                row=stats_report.iloc[0]
                print(f"Appearances = {row["appearances"]}")
                print(f"Minutes played = {row["minutes_played"]}")
                print(f"Goals Scored = {row["goals"]}")
                print(f"Assists Given = {row["assists"]}")
                print(f"Yellow Cards Recieved = {row["yellow_cards"]}")
                print(f"Red Cards Recieved = {row["red_cards"]}")
        case 3:
            player=input("Enter the players Full Name : ")
            value_report=Rf_data[Rf_data["player_name"]==player]
            if value_report.empty:
                print("No such player could be found")
            else:
                row=value_report.iloc[0]
                print(f"Player vlaue at the start of season {row["season"]} = €{row["market_value_in_eur"]:,.2f}")
                print(f"Predicted value at the end of the season {row["season"]} = €{row["predictions"]:,.2f}")
                print(f"Actual player value at the end of the season {row["season"]} = €{row["target_value_in_eur"]:,.2f}")
                print(f"Prediction vs Actual value difference = {row["evaluation_difference"]:,.2f}")
        case 4:
            appearances=int(input("Enter appearances of the player = "))
            yellow_cards=int(input("Enter the number of yellow cards recieved = "))
            red_cards=int(input("Enter the number of red cards recieved = "))
            goals=int(input("Enter the number of goals scored = "))
            assists=int(input("Enter the number of assists given = "))
            minutes_played=int(input("Enter the number of minutes played = "))
            age=int(input("Enter the player age = "))
            market_value_in_eur=int(input("Enter the market value at the season start = "))
            p=pd.DataFrame({
                "appearances" : [appearances],
                "yellow_cards" : [yellow_cards],
                "red_cards" : [red_cards],
                "goals" : [goals],
                "assists" : [assists],
                "minutes_played" : [minutes_played],
                "age" : [age],
                "market_value_in_eur" : [market_value_in_eur]
            })
            pred=Rf_model.predict(p)[0]
            pred = (pred/1000000).round() * 1000000
            print(f"The predicted value is {pred:,.2f}")
        case 5:
            player=input("Enter the players Full Name : ")
            value_report=Rf_data[Rf_data["player_name"]==player]
            if value_report.empty:
                print("No such player could be found")
            else:
                row=value_report.iloc[0]
                print("------------------------")
                print(f"Appearances = {row["appearances"]}")
                print(f"Minutes played = {row["minutes_played"]}")
                print(f"Goals Scored = {row["goals"]}")
                print(f"Assists Given = {row["assists"]}")
                print(f"Yellow Cards Recieved = {row["yellow_cards"]}")
                print(f"Red Cards Recieved = {row["red_cards"]}")
                print("------------------------")
                print(f"Player vlaue at the start of season {row["season"]} = €{row["market_value_in_eur"]:,.2f}")
                print(f"Predicted value at the end of the season {row["season"]} = €{row["predictions"]:,.2f}")
                print(f"Actual player value at the end of the season {row["season"]} = €{row["target_value_in_eur"]:,.2f}")
                print(f"Prediction vs Actual value difference = {row["evaluation_difference"]:,.2f}")
        case 6:
            Scouting=Rf_data[Rf_data["target_value_in_eur"] < Rf_data["predictions"]].copy()
            Scouting=Scouting.sort_values(by="evaluation_difference",ascending=False)
            num=int(input("Enter top how many players u want to view: "))
            pd.set_option("display.max_columns",None)
            print(Scouting.head(num))
        case 7:
            import matplotlib.pyplot as plt
            Player_rise = Rf_data.copy()
            Player_rise["predicted_value_increase"] = (Player_rise["predictions"]-Player_rise["market_value_in_eur"])
            Player_rise=Player_rise.sort_values(by="predicted_value_increase",ascending=False)
            pd.set_option("display.max_columns",None)
            print(Player_rise.head(10))
            top10 = Player_rise.head(10).sort_values(by="predicted_value_increase",ascending=True)
            plt.figure(figsize=(12, 6))
            plt.barh(top10["player_name"],top10["predicted_value_increase"]/1000000)
            plt.xlabel("Predicted Market Value Increase (€ millions)")
            plt.ylabel("Player Name")
            plt.title("Top 10 Players by Predicted Market Value Increase")
            plt.tight_layout()
            plt.show()
        case 8:
            import matplotlib.pyplot as plt
            Player_fall=Rf_data.copy()
            Player_fall["predicted_value_decrease"]=Player_fall["predictions"]-Player_fall["market_value_in_eur"]
            Player_fall=Player_fall.sort_values(by="predicted_value_decrease",ascending=True)
            pd.set_option("display.max_columns",None)
            print(Player_fall.head(10))
            top10=Player_fall.head(10).sort_values(by="predicted_value_decrease",ascending=False)
            plt.figure(figsize=(12, 6))
            plt.barh(top10["player_name"],top10["predicted_value_decrease"]/1000000)
            plt.xlabel("Predicted Change in Market Value (€ millions)")
            plt.ylabel("Player Name")
            plt.title("Top 10 Players by Predicted Market Value Decrease")
            plt.tight_layout()
            plt.show()
        case 9:
            player=input("Enter the players Full Name : ")
            Rf_value_report=Rf_data[Rf_data["player_name"]==player]
            Lr_value_report=Lr_data[Lr_data["player_name"]==player]
            if Rf_value_report.empty or Lr_value_report.empty:
                print("No such player could be found")
            else:
                row=Rf_value_report.iloc[0]
                row2=Lr_value_report.iloc[0]
                print("Random Forests model predictions = ")
                print(f"Player vlaue at the start of season {row["season"]} = €{row["market_value_in_eur"]:,.2f}")
                print(f"Predicted value at the end of the season {row["season"]} = €{row["predictions"]:,.2f}")
                print(f"Actual player value at the end of the season {row["season"]} = €{row["target_value_in_eur"]:,.2f}")
                print(f"Prediction vs Actual value difference = {row["evaluation_difference"]:,.2f}")
                print("------------------------")
                print("Linear Regression model predictions = ")
                print(f"Player vlaue at the start of season {row2["season"]} = €{row2["market_value_in_eur"]:,.2f}")
                print(f"Predicted value at the end of the season {row2["season"]} = €{row2["predictions"]:,.2f}")
                print(f"Actual player value at the end of the season {row2["season"]} = €{row2["target_value_in_eur"]:,.2f}")
                print(f"Prediction vs Actual value difference = {row2["evaluation_difference"]:,.2f}")
        case 10:
            actual = Rf_data["target_value_in_eur"] / 1_000_000
            predicted = Rf_data["predictions"] / 1_000_000
            plt.figure(figsize=(8, 6))
            plt.scatter(actual, predicted, alpha=0.5)
            low = min(actual.min(), predicted.min())
            high = max(actual.max(), predicted.max())
            plt.plot([low, high], [low, high], "r--",label="Perfect prediction")
            plt.xlabel("Actual Ending Market Value (€ millions)")
            plt.ylabel("Predicted Ending Market Value (€ millions)")
            plt.title("Random Forest: Actual vs Predicted Market Value")
            plt.legend()
            plt.tight_layout()
            plt.show()
        case _:
            print("Invalid")