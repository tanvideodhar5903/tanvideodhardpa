import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


data = pd.read_csv("C:/Users/SLIM5/OneDrive/Documents/Project_semV/diamonds.csv")
print(data.head())

figure1 = px.scatter(data_frame = data, x= "carat", y = "price", size = "depth", color = "cut", trendline = "ols")
figure1.show()

data["size"] = data["x"] * data["y"]*data["z"]
print(data)

figure2 = px.scatter(data_frame = data, x = "size", y = "price", size = "size", color = "cut", trendline = "ols")
figure2.show()

figure3 = px.box(data, x = "cut", y = "price", color = "color")
figure3.show()

figure4 = px.box(data, x = "cut", y = "price", color = "clarity")
figure4.show()

correlation = data.corr()
print(correlation["price"].sort_values(ascending=False))

data["cut"] = data["cut"].map({"Ideal": 1, 
                               "Premium": 2, 
                               "Good": 3,
                               "Very Good": 4,
                               "Fair": 5})

#splitting data
from sklearn.model_selection import train_test_split
x = np.array(data[["carat", "cut", "size"]])
y = np.array(data[["price"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(xtrain, ytrain)


