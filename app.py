# ======================================================
# CBR PREDICTION WEB APP
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------
st.set_page_config(
    page_title="CBR Prediction App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 California Bearing Ratio (CBR) Prediction")
st.markdown("Machine Learning based estimation for geotechnical applications")

# ------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("CBR Data.xlsx")

df = load_data()

st.subheader("📋 Dataset Preview")
st.dataframe(df.head())

# ------------------------------------------------------
# DEFINE FEATURES AND TARGET
# ------------------------------------------------------
TARGET = "CBR"   # DO NOT CHANGE unless column name differs

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ------------------------------------------------------
# TRAIN-TEST SPLIT
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ------------------------------------------------------
# MACHINE LEARNING PIPELINE
# ------------------------------------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# ------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("RMSE", f"{rmse:.3f}")

# ------------------------------------------------------
# USER INPUT SECTION
# ------------------------------------------------------
st.sidebar.header("🔢 Enter Soil Properties")

input_data = {}

for col in X.columns:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())

    input_data[col] = st.sidebar.number_input(
        label=col,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

input_df = pd.DataFrame([input_data])

# ------------------------------------------------------
# PREDICTION
# ------------------------------------------------------
st.subheader("🎯 Predicted CBR Value")

if st.button("Predict CBR"):
    prediction = model.predict(input_df)[0]

    if prediction < 3:
        st.error(f"Predicted CBR = {prediction:.2f} % (Very Poor Subgrade)")
    elif 3 <= prediction < 5:
        st.warning(f"Predicted CBR = {prediction:.2f} % (Poor Subgrade)")
    elif 5 <= prediction < 10:
        st.info(f"Predicted CBR = {prediction:.2f} % (Fair Subgrade)")
    else:
        st.success(f"Predicted CBR = {prediction:.2f} % (Good Subgrade)")

# ------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------
st.subheader("🔍 Feature Importance")

importances = model.named_steps["rf"].feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values()

fig, ax = plt.subplots(figsize=(7, 4))
feat_imp.plot(kind="barh", ax=ax)
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance")

st.pyplot(fig)

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown("---")
st.markdown("Developed for academic and research purposes")
