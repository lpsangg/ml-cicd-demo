import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Đọc dữ liệu
data = pd.read_csv("data/housing.csv")
X = data[["RM", "LSTAT", "PTRATIO"]]  # Chọn vài cột làm ví dụ
y = data["MEDV"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Lưu model
joblib.dump(model, "model/model.pkl")
print("Model trained and saved!")
