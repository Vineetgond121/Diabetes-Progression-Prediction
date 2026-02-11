import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# App title
st.title("Diabetes Linear Regression Model")

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = list(diabetes.feature_names)

# Important features only
important_features = ["age", "sex", "bmi", "bp", "s5"]
indices = [feature_names.index(f) for f in important_features]
X_imp = X[:, indices]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Mean Squared Error: **{mse:.2f}**")
st.write(f"R-squared: **{r2:.2f}**")

# Sidebar inputs
st.sidebar.header("Patient Inputs")

age = st.sidebar.slider("Age (years)", 20, 80, 45)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)
bp = st.sidebar.slider("Blood Pressure (mmHg)", 60, 180, 120)
s5 = st.sidebar.slider("Serum Measurement (s5)", 90, 200, 120)

# Standardization function (dataset already standardized)
def standardize(value, original_column):
    col = X[:, original_column]
    return (value - col.mean()) / col.std()

# User input vector
user_std = np.array([
    standardize(age, feature_names.index("age")),
    1.0 if sex == "Male" else -1.0,
    standardize(bmi, feature_names.index("bmi")),
    standardize(bp, feature_names.index("bp")),
    standardize(s5, feature_names.index("s5"))
]).reshape(1, -1)

# Prediction
prediction = model.predict(user_std)[0]

st.subheader("Predicted Diabetes Progression")
st.write(f"**{prediction:.2f}**")

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# True vs Predicted
axs[0].scatter(y_test, y_pred, color="blue", alpha=0.5)
axs[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--",
    lw=2
)
axs[0].set_title("True vs Predicted Values")
axs[0].set_xlabel("True Values")
axs[0].set_ylabel("Predicted Values")
axs[0].grid(True)

# BMI vs Prediction
axs[1].scatter(X_test[:, important_features.index("bmi")], y_pred,
               color="green", alpha=0.7)
axs[1].scatter(bmi, prediction, color="red", s=100)
axs[1].set_title("BMI vs Predicted Values")
axs[1].set_xlabel("BMI")
axs[1].set_ylabel("Predicted Diabetes Progression")
axs[1].grid(True)

plt.tight_layout()
st.pyplot(fig)

