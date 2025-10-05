# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import traceback

st.set_page_config(page_title="Sales Forecasting Dashboard", page_icon="📈", layout="wide")
st.title("📈 Sales Forecasting Dashboard — Model & Accuracy")

# Sidebar controls
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV (must contain 'Order Date' & 'Sales')", type=["csv"])
model_choice = st.sidebar.selectbox("Model", ["XGBoost", "LightGBM", "Blend (XGB+LGBM)"])
n_lags = st.sidebar.slider("Number of lag days", min_value=7, max_value=60, value=12, step=1)
n_display = st.sidebar.slider("Number of test points to display (bar chart)", 5, 120, 30)

# Helper: safe MAPE (avoid div-by-zero)
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

if uploaded_file is None:
    st.info("👆 Upload your CSV file (with columns 'Order Date' and 'Sales') to run models.")
    st.stop()

# Main block: everything from here on assumes a file is uploaded
try:
    df = pd.read_csv(uploaded_file)
    if "Order Date" not in df.columns or "Sales" not in df.columns:
        st.error("Dataset must contain 'Order Date' and 'Sales' columns. Please check your file.")
        st.stop()

    # Parse dates robustly
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    null_dates = df["Order Date"].isna().sum()
    if null_dates > 0:
        st.warning(f"⚠️ Found {null_dates} rows with invalid Order Date — they will be ignored.")

    df = df.dropna(subset=["Order Date"])
    # Aggregate daily sales
    df = df.groupby("Order Date")["Sales"].sum().reset_index()
    df = df.set_index("Order Date").asfreq("D", fill_value=0)

    # Feature engineering
    ml_data = pd.DataFrame(df["Sales"]).rename(columns={"Sales": "Sales"})
    # Lags
    for lag in range(1, n_lags + 1):
        ml_data[f"lag{lag}"] = ml_data["Sales"].shift(lag)

    # Rolling features
    ml_data["rolling7"] = ml_data["Sales"].rolling(7, min_periods=1).mean().shift(1)
    ml_data["rolling30"] = ml_data["Sales"].rolling(30, min_periods=1).mean().shift(1)
    ml_data["rolling7_std"] = ml_data["Sales"].rolling(7, min_periods=1).std().shift(1).fillna(0)
    ml_data["rolling30_std"] = ml_data["Sales"].rolling(30, min_periods=1).std().shift(1).fillna(0)
    ml_data["expanding_mean"] = ml_data["Sales"].expanding().mean().shift(1).fillna(method="bfill")

    # Date features
    ml_data["month"] = ml_data.index.month
    ml_data["dayofweek"] = ml_data.index.dayofweek
    ml_data["quarter"] = ml_data.index.quarter
    ml_data["is_weekend"] = (ml_data.index.dayofweek >= 5).astype(int)

    # Drop rows with NaN due to shifting
    ml_data.dropna(inplace=True)

    if len(ml_data) < 50:
        st.warning("Dataset after feature creation is small (<50 rows). Results may be unreliable.")

    # Train-test split (80/20)
    train_size = int(len(ml_data) * 0.8)
    train = ml_data.iloc[:train_size]
    test = ml_data.iloc[train_size:]

    X_train, y_train = train.drop("Sales", axis=1), train["Sales"]
    X_test, y_test = test.drop("Sales", axis=1), test["Sales"]

    # Train models
    if model_choice == "XGBoost":
        model_xgb = xgb.XGBRegressor(
            n_estimators=1500, learning_rate=0.03, max_depth=8,
            subsample=0.9, colsample_bytree=0.9, random_state=42, verbosity=0
        )
        model_xgb.fit(X_train, y_train)
        xgb_pred_train = model_xgb.predict(X_train)
        xgb_pred_test = model_xgb.predict(X_test)

        # Use only XGB preds
        y_pred_train = xgb_pred_train
        y_pred_test = xgb_pred_test

    elif model_choice == "LightGBM":
        model_lgb = lgb.LGBMRegressor(
            n_estimators=1500, learning_rate=0.03, max_depth=8,
            subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1
        )
        model_lgb.fit(X_train, y_train)
        lgb_pred_train = model_lgb.predict(X_train)
        lgb_pred_test = model_lgb.predict(X_test)

        y_pred_train = lgb_pred_train
        y_pred_test = lgb_pred_test

    else:  # Blend
        model_xgb = xgb.XGBRegressor(
            n_estimators=1200, learning_rate=0.05, max_depth=7,
            subsample=0.9, colsample_bytree=0.9, random_state=42, verbosity=0
        )
        model_lgb = lgb.LGBMRegressor(
            n_estimators=1200, learning_rate=0.05, max_depth=7,
            subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1
        )
        model_xgb.fit(X_train, y_train)
        model_lgb.fit(X_train, y_train)

        xgb_pred_train = model_xgb.predict(X_train)
        lgb_pred_train = model_lgb.predict(X_train)
        xgb_pred_test = model_xgb.predict(X_test)
        lgb_pred_test = model_lgb.predict(X_test)

        # Weighted blend (you can change weights)
        w_xgb, w_lgb = 0.6, 0.4
        y_pred_train = w_xgb * xgb_pred_train + w_lgb * lgb_pred_train
        y_pred_test = w_xgb * xgb_pred_test + w_lgb * lgb_pred_test

    # Calculate metrics (Train)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    train_mape = safe_mape(y_train, y_pred_train)

    # Calculate metrics (Test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    test_mape = safe_mape(y_test, y_pred_test)

    # ======= Display metrics neatly =======
    st.subheader("📌 Accuracy Summary (Train vs Test)")
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    tcol1.metric("Train MAE", f"{train_mae:.2f}")
    tcol2.metric("Train RMSE", f"{train_rmse:.2f}")
    tcol3.metric("Train R²", f"{train_r2:.3f}")
    tcol4.metric("Train MAPE", f"{train_mape:.2f}%")

    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("Test MAE", f"{test_mae:.2f}")
    scol2.metric("Test RMSE", f"{test_rmse:.2f}")
    scol3.metric("Test R²", f"{test_r2:.3f}")
    scol4.metric("Test MAPE", f"{test_mape:.2f}%")

    # Detailed metrics table (both)
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²", "MAPE (%)"],
        "Train": [f"{train_mae:.2f}", f"{train_rmse:.2f}", f"{train_r2:.4f}", f"{train_mape:.2f}%"],
        "Test": [f"{test_mae:.2f}", f"{test_rmse:.2f}", f"{test_r2:.4f}", f"{test_mape:.2f}%"]
    })
    st.write("#### 🔍 Detailed Metrics Table")
    st.table(metrics_df)

    # ======= Plots =======
    st.subheader("📈 Actual vs Predicted — Line Chart (Test Set)")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(test.index, y_test, label="Actual", marker='o')
    ax1.plot(test.index, y_pred_test, label="Predicted", marker='x')
    ax1.legend()
    ax1.set_ylabel("Sales")
    st.pyplot(fig1)

    # Bar chart for first n_display test points
    n_show = min(len(y_test), n_display)
    st.subheader(f"📊 Actual vs Predicted — Bar Chart (first {n_show} test points)")
    actual = y_test.values[:n_show]
    predicted = y_pred_test[:n_show]
    x = np.arange(n_show)

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.bar(x - 0.2, actual, width=0.4, label="Actual")
    ax2.bar(x + 0.2, predicted, width=0.4, label="Predicted")
    ax2.legend()
    ax2.set_xlabel("Test index")
    st.pyplot(fig2)

    # Show a small table of actual vs predicted values
    compare_df = pd.DataFrame({
        "date": test.index[:n_show],
        "actual": actual,
        "predicted": np.round(predicted, 2)
    }).set_index("date")
    st.write("#### 🔢 Actual vs Predicted (sample)")
    st.dataframe(compare_df)

except Exception as e:
    st.error("An error occurred while processing the data or training the models.")
    st.text(traceback.format_exc())
