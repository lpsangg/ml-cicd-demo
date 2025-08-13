import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Đọc dữ liệu
data = pd.read_csv("data/housing.csv")

X = data[["Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms"]]
y = data["Price"]
# Train model
model = LinearRegression()
model.fit(X, y)

# Lưu model
joblib.dump(model, "model/model.pkl")
print("Model trained and saved!")
