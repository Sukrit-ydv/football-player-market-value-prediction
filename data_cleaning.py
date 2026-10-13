import pandas as pd

training_data=pd.read_csv("data/training/training_data.csv")

print(training_data.shape)
print(training_data.columns)
training_data=training_data.dropna(subset=["age"])
training_data=training_data.dropna(subset=["market_value_in_eur"])
print(training_data.isnull().sum())
#print(training_data[training_data["market_value_in_eur"].isnull()])
training_data.to_csv("data/training/final_training_data.csv",index=False)
