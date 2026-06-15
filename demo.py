import pandas as pd



data=pd.read_csv("data/Corona_NLP_train.csv", encoding="latin1")

data["Sentiment"]=data["Sentiment"].str.replace("Extremely ", "", regex=False)


#print(data[data["OriginalTweet"].isnull()])

#print(data["Sentiment"].unique())
