"""NeuralStock Streamlit dashboard."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from model import FeedforwardMLP, StackedLSTM


MODELS_DIR = Path("neural_stock/models")
DATA_PATH = MODELS_DIR / "processed_data.pkl"
METADATA_PATH = MODELS_DIR / "training_metadata.pkl"
LSTM_PATH = MODELS_DIR / "best_lstm_model.pth"
MLP_PATH = MODELS_DIR / "best_mlp_model.pth"
HISTORY_PATH = MODELS_DIR / "training_history.csv"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"


st.set_page_config(page_title="NeuralStock Forecast Dashboard", layout="wide")


def inverse_target(values: np.ndarray, scaler) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    if scaler is None:
        return values.reshape(-1)
    return scaler.inverse_transform(values).reshape(-1)


def scaler_input(scaler, df: pd.DataFrame, columns: list[str]):
    if getattr(scaler, "feature_names_in_", None) is not None:
        return df[columns]
    return df[columns].to_numpy()


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, dict]:
    with DATA_PATH.open("rb") as file:
        data = pickle.load(file)

    metadata = {}
    if METADATA_PATH.exists():
        with METADATA_PATH.open("rb") as file:
            metadata = pickle.load(file)

    df = data["df"].copy()
    df["date"] = pd.to_datetime(df["date"])

    feature_cols = metadata.get("feature_cols") or data.get("feature_cols")
    window_size = metadata.get("window_size", 7)
    input_size = len(feature_cols)
    metadata = {
        "feature_cols": feature_cols,
        "target_col": metadata.get("target_col", "units_sold"),
        "window_size": window_size,
        "feature_scaler": metadata.get("feature_scaler") or data.get("scaler"),
        "demand_scaler": metadata.get("demand_scaler") or data.get("demand_scaler"),
        "lstm_config": metadata.get(
            "lstm_config",
            {
                "input_size": input_size,
                "hidden_size": 128,
                "num_layers": 2,
                "output_size": 1,
                "dropout": 0.2,
            },
        ),
        "mlp_config": metadata.get(
            "mlp_config",
            {
                "input_size": input_size * window_size,
                "hidden_sizes": [128, 64],
                "output_size": 1,
                "dropout": 0.2,
            },
        ),
    }
    return df, metadata


@st.cache_resource(show_spinner=False)
def load_models(lstm_mtime: float, mlp_mtime: float, _metadata: dict):
    del lstm_mtime, mlp_mtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lstm = StackedLSTM(**_metadata["lstm_config"])
    lstm.load_state_dict(torch.load(LSTM_PATH, map_location=device))
    lstm.to(device)
    lstm.eval()

    mlp = FeedforwardMLP(**_metadata["mlp_config"])
    mlp.load_state_dict(torch.load(MLP_PATH, map_location=device))
    mlp.to(device)
    mlp.eval()

    return lstm, mlp, device


def prepare_features(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    feature_cols = metadata["feature_cols"]
    prepared = df.copy()
    prepared[feature_cols] = prepared[feature_cols].apply(pd.to_numeric, errors="coerce")
    prepared[feature_cols] = prepared[feature_cols].ffill().bfill().fillna(0.0)

    scaler = metadata.get("feature_scaler")
    if scaler is not None:
        prepared[feature_cols] = scaler.transform(scaler_input(scaler, prepared, feature_cols))
    return prepared


def predict_batch(model, x: np.ndarray, device: torch.device, flatten: bool = False) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32, device=device)
        if flatten:
            tensor = tensor.flatten(start_dim=1)
        return model(tensor).cpu().numpy().reshape(-1)


@st.cache_data(show_spinner=False)
def run_backtest(
    df: pd.DataFrame,
    _metadata: dict,
    category: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    lstm_mtime: float,
    mlp_mtime: float,
) -> pd.DataFrame:
    del lstm_mtime, mlp_mtime
    lstm, mlp, device = load_models(LSTM_PATH.stat().st_mtime, MLP_PATH.stat().st_mtime, _metadata)
    target_col = _metadata["target_col"]
    feature_cols = _metadata["feature_cols"]
    window_size = _metadata["window_size"]
    demand_scaler = _metadata.get("demand_scaler")

    filtered = df[
        (df["product_category"] == category)
        & (df["date"] >= pd.to_datetime(start_date))
        & (df["date"] <= pd.to_datetime(end_date))
    ].copy()
    if filtered.empty:
        return pd.DataFrame()

    prepared = prepare_features(filtered, _metadata)
    rows = []
    x_windows = []
    actuals = []
    naive = []

    for product_id, group in prepared.groupby("product_id", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        raw_group = filtered[filtered["product_id"] == product_id].sort_values("date").reset_index(drop=True)
        if len(group) <= window_size:
            continue

        for index in range(window_size, len(group)):
            x_windows.append(group.loc[index - window_size : index - 1, feature_cols].to_numpy(dtype=np.float32))
            actuals.append(float(raw_group.loc[index, target_col]))
            naive_prediction = float(raw_group.loc[index - 1, target_col])
            naive.append(naive_prediction)
            rows.append(
                {
                    "date": raw_group.loc[index, "date"],
                    "product_id": product_id,
                    "product_category": category,
                    "actual_demand": float(raw_group.loc[index, target_col]),
                    "naive": naive_prediction,
                }
            )

    if not x_windows:
        return pd.DataFrame()

    x = np.stack(x_windows)
    lstm_pred = inverse_target(predict_batch(lstm, x, device), demand_scaler)
    mlp_pred = inverse_target(predict_batch(mlp, x, device, flatten=True), demand_scaler)

    result = pd.DataFrame(rows)
    result["lstm"] = np.clip(lstm_pred, 0, None)
    result["mlp"] = np.clip(mlp_pred, 0, None)
    result["naive"] = np.clip(np.asarray(naive), 0, None)
    return result


def weekly_summary(backtest: pd.DataFrame) -> pd.DataFrame:
    if backtest.empty:
        return backtest
    weekly = backtest.copy()
    weekly["week"] = weekly["date"].dt.to_period("W").apply(lambda period: period.start_time)
    return (
        weekly.groupby("week", as_index=False)[["actual_demand", "lstm", "mlp", "naive"]]
        .sum()
        .sort_values("week")
    )


def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    denominator = np.maximum(actual.abs(), 1e-5)
    return float((np.abs(actual - predicted) / denominator).mean() * 100)


def build_comparison(weekly: pd.DataFrame) -> pd.DataFrame:
    if weekly.empty:
        return pd.DataFrame()

    rows = []

    for model_name, column in [("LSTM", "lstm"), ("MLP", "mlp"), ("Naive", "naive")]:
        error = weekly["actual_demand"] - weekly[column]

        rows.append(
            {
                "Model": model_name,
                "Weekly MAPE (%)": calculate_mape(
                    weekly["actual_demand"],
                    weekly[column]
                ),
                "Weekly MAE": float(error.abs().mean()),
                "Weekly RMSE": float(np.sqrt((error**2).mean())),
                "R² Score": float(
                    r2_score(
                        weekly["actual_demand"],
                        weekly[column]
                    )
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("Weekly MAPE (%)")


@st.cache_data(show_spinner=False)
def category_mape_table(
    df: pd.DataFrame,
    _metadata: dict,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    lstm_mtime: float,
    mlp_mtime: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    backtests = []
    rows = []
    for category in sorted(df["product_category"].dropna().unique()):
        backtest = run_backtest(df, _metadata, category, start_date, end_date, lstm_mtime, mlp_mtime)
        if backtest.empty:
            continue
        backtests.append(backtest)
        weekly = weekly_summary(backtest)
        comparison = build_comparison(weekly)
        rows.append(
            {
                "Category": category,
                "LSTM MAPE (%)": comparison.loc[comparison["Model"] == "LSTM", "Weekly MAPE (%)"].iloc[0],
                "MLP MAPE (%)": comparison.loc[comparison["Model"] == "MLP", "Weekly MAPE (%)"].iloc[0],
                "Naive MAPE (%)": comparison.loc[comparison["Model"] == "Naive", "Weekly MAPE (%)"].iloc[0],
            }
        )

    all_backtest = pd.concat(backtests, ignore_index=True) if backtests else pd.DataFrame()
    return pd.DataFrame(rows), weekly_summary(all_backtest)


def build_forecast(df: pd.DataFrame, metadata: dict, category: str, weekly: pd.DataFrame) -> pd.DataFrame:
    lstm, _, device = load_models(LSTM_PATH.stat().st_mtime, MLP_PATH.stat().st_mtime, metadata)
    feature_cols = metadata["feature_cols"]
    window_size = metadata["window_size"]
    demand_scaler = metadata.get("demand_scaler")
    category_df = df[df["product_category"] == category].sort_values(["product_id", "date"]).copy()
    prepared = prepare_features(category_df, metadata)

    product_forecasts = []
    for product_id, group in prepared.groupby("product_id", sort=False):
        if len(group) < window_size:
            continue
        window = group.tail(window_size)[feature_cols].to_numpy(dtype=np.float32)
        prediction = inverse_target(predict_batch(lstm, np.expand_dims(window, 0), device), demand_scaler)[0]
        product_forecasts.append(max(0.0, float(prediction)))

    recent_weekly = (
        category_df.assign(week=category_df["date"].dt.to_period("W").apply(lambda period: period.start_time))
        .groupby("week")["units_sold"]
        .sum()
        .tail(6)
    )
    model_signal = float(np.sum(product_forecasts)) if product_forecasts else float(recent_weekly.tail(1).sum())
    recent_mean = float(recent_weekly.tail(4).mean()) if len(recent_weekly) else model_signal
    base_forecast = 0.6 * model_signal + 0.4 * recent_mean

    if len(recent_weekly) >= 2 and recent_weekly.iloc[-2] > 0:
        trend = np.clip((recent_weekly.iloc[-1] / recent_weekly.iloc[-2]) - 1, -0.12, 0.12)
    else:
        trend = 0.0

    if weekly.empty:
        residual_std = max(base_forecast * 0.15, 1.0)
    else:
        residual_std = float((weekly["actual_demand"] - weekly["lstm"]).std())
        if np.isnan(residual_std) or residual_std == 0:
            residual_std = max(base_forecast * 0.15, 1.0)

    last_date = category_df["date"].max()
    forecast_rows = []
    for horizon in range(1, 5):
        forecast = max(0.0, base_forecast * ((1 + trend) ** (horizon - 1)))
        interval = 1.96 * residual_std * np.sqrt(horizon)
        forecast_rows.append(
            {
                "week": last_date + pd.Timedelta(weeks=horizon),
                "forecast": forecast,
                "lower_ci": max(0.0, forecast - interval),
                "upper_ci": forecast + interval,
            }
        )
    return pd.DataFrame(forecast_rows)


def build_reorder_alerts(df: pd.DataFrame, metadata: dict, category: str) -> pd.DataFrame:
    lstm, _, device = load_models(LSTM_PATH.stat().st_mtime, MLP_PATH.stat().st_mtime, metadata)
    feature_cols = metadata["feature_cols"]
    window_size = metadata["window_size"]
    demand_scaler = metadata.get("demand_scaler")
    category_df = df[df["product_category"] == category].sort_values(["product_id", "date"]).copy()
    prepared = prepare_features(category_df, metadata)

    alerts = []
    for product_id, group in prepared.groupby("product_id", sort=False):
        raw_group = category_df[category_df["product_id"] == product_id].sort_values("date")
        if len(group) < window_size:
            continue

        prediction = inverse_target(
            predict_batch(
                lstm,
                np.expand_dims(group.tail(window_size)[feature_cols].to_numpy(dtype=np.float32), 0),
                device,
            ),
            demand_scaler,
        )[0]
        latest = raw_group.iloc[-1]
        stock_on_hand = float(latest.get("stock_on_hand", 0))
        reorder_point = float(latest.get("reorder_point", 0))
        projected_stock = stock_on_hand - max(0.0, prediction)
        if stock_on_hand <= reorder_point or projected_stock <= reorder_point:
            alerts.append(
                {
                    "product_id": product_id,
                    "stock_on_hand": stock_on_hand,
                    "reorder_point": reorder_point,
                    "next_demand_forecast": max(0.0, prediction),
                    "projected_stock": projected_stock,
                    "alert": "Reorder now" if stock_on_hand <= reorder_point else "Reorder soon",
                }
            )

    return pd.DataFrame(alerts).sort_values("projected_stock") if alerts else pd.DataFrame()


@st.cache_data(show_spinner=False)
def feature_importance(df: pd.DataFrame, _metadata: dict, category: str) -> pd.DataFrame:
    feature_cols = _metadata["feature_cols"]
    target_col = _metadata["target_col"]
    filtered = df[df["product_category"] == category].copy()
    if filtered.empty:
        return pd.DataFrame()

    sample = filtered.sample(min(len(filtered), 5000), random_state=42)
    x = sample[feature_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    y = pd.to_numeric(sample[target_col], errors="coerce").ffill().bfill().fillna(0.0)

    model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(x, y)

    importance = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    importance["feature_group"] = np.select(
        [
            importance["feature"].str.contains("lag|rolling", case=False, regex=True),
            importance["feature"].str.contains("day|week|month|quarter|weekend", case=False, regex=True),
            importance["feature"].str.contains("stock|reorder", case=False, regex=True),
            importance["feature"].str.contains("price|discount|promotion", case=False, regex=True),
        ],
        ["Lag/Rolling demand", "Calendar", "Inventory", "Price/Promotion"],
        default="Other",
    )
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def feature_insights(importance: pd.DataFrame) -> tuple[dict, list[tuple[str, list[str]]]]:
    if importance.empty:
        return {}, []

    top_features = importance.head(5)
    group_totals = importance.groupby("feature_group", as_index=False)["importance"].sum()
    strongest_group = group_totals.sort_values("importance", ascending=False).iloc[0]
    lag_share = group_totals.loc[group_totals["feature_group"] == "Lag/Rolling demand", "importance"].sum()
    calendar_share = group_totals.loc[group_totals["feature_group"] == "Calendar", "importance"].sum()
    inventory_share = group_totals.loc[group_totals["feature_group"] == "Inventory", "importance"].sum()
    price_share = group_totals.loc[group_totals["feature_group"] == "Price/Promotion", "importance"].sum()
    top_three_share = importance.head(3)["importance"].sum()
    top_five_share = importance.head(5)["importance"].sum()
    top_ten_share = importance.head(10)["importance"].sum()
    top_calendar = importance[importance["feature_group"] == "Calendar"].head(3)["feature"].tolist()
    top_lag = importance[importance["feature_group"] == "Lag/Rolling demand"].head(3)["feature"].tolist()
    top_inventory = importance[importance["feature_group"] == "Inventory"].head(3)["feature"].tolist()
    top_price = importance[importance["feature_group"] == "Price/Promotion"].head(3)["feature"].tolist()

    metric_summary = {
        "top_feature": str(top_features.iloc[0]["feature"]),
        "top_feature_importance": float(top_features.iloc[0]["importance"]),
        "strongest_group": str(strongest_group["feature_group"]),
        "strongest_group_importance": float(strongest_group["importance"]),
        "top_five_share": float(top_five_share),
    }

    interpretation = [
        (
            "Main Demand Signal",
            [
                f"`{top_features.iloc[0]['feature']}` is the strongest single predictor and accounts for {top_features.iloc[0]['importance']:.1%} of the signal.",
                f"The top 3 features explain {top_three_share:.1%}; the top 5 explain {top_five_share:.1%}; the top 10 explain {top_ten_share:.1%}.",
                "A high top-feature concentration means the forecast is being driven by a few strong patterns; a lower concentration means the model is combining many smaller signals.",
            ],
        ),
        (
            "Feature Family Breakdown",
            [
                f"The strongest group is {strongest_group['feature_group']} with {strongest_group['importance']:.1%} total importance.",
                f"Lag/rolling demand contributes {lag_share:.1%}; calendar variables contribute {calendar_share:.1%}; inventory contributes {inventory_share:.1%}; price/promotion contributes {price_share:.1%}.",
            ],
        ),
    ]

    if lag_share > calendar_share:
        interpretation.append(
            (
                "Lag vs Calendar",
                [
                    "Recent demand history is more influential than calendar timing for this category.",
                    "This usually means short-term sales momentum, rolling averages, or recent spikes are important for the next forecast.",
                    f"Most useful demand-history features: {', '.join(top_lag) if top_lag else 'none among the top drivers'}.",
                ],
            )
        )
    elif calendar_share > lag_share:
        interpretation.append(
            (
                "Lag vs Calendar",
                [
                    "Calendar variables contribute more signal than lag/rolling demand features for this category.",
                    "This points to seasonality or recurring timing effects such as month, quarter, week, or weekend patterns.",
                    f"Most useful calendar features: {', '.join(top_calendar) if top_calendar else 'none among the top drivers'}.",
                ],
            )
        )
    else:
        interpretation.append(
            (
                "Lag vs Calendar",
                [
                    "Lag/rolling demand and calendar variables are contributing similar signal.",
                    "The category appears to need both recent sales behavior and recurring time patterns for accurate forecasts.",
                ],
            )
        )

    interpretation.append(
        (
            "Operational Reading",
            [
                f"Inventory-related variables to watch: {', '.join(top_inventory) if top_inventory else 'not dominant for this category'}.",
                f"Price/promotion variables to watch: {', '.join(top_price) if top_price else 'not dominant for this category'}.",
                "Use the high-importance features as monitoring fields when reviewing reorder alerts and unusual forecast jumps.",
            ],
        )
    )
    interpretation.append(
        (
            "Modeling Note",
            [
                "These importances come from a Random Forest diagnostic model trained on the selected category, so they are directional explanations rather than causal proof.",
                "If a feature has low importance here, it does not mean it is useless everywhere; it means it was less useful for this category's historical demand pattern.",
            ],
        )
    )

    return metric_summary, interpretation


def line_chart(weekly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name, column in [("Actual", "actual_demand"), ("LSTM", "lstm"), ("MLP", "mlp"), ("Naive", "naive")]:
        fig.add_trace(go.Scatter(x=weekly["week"], y=weekly[column], mode="lines+markers", name=name))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Weekly demand")
    return fig


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not history.empty:
        fig.add_trace(
            go.Scatter(
                x=history["week"],
                y=history["actual_demand"],
                mode="lines+markers",
                name="Historical demand",
            )
        )
    fig.add_trace(go.Scatter(x=forecast["week"], y=forecast["forecast"], mode="lines+markers", name="4-week forecast"))
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast["week"], forecast["week"][::-1]]),
            y=pd.concat([forecast["upper_ci"], forecast["lower_ci"][::-1]]),
            fill="toself",
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(31, 119, 180, 0.18)",
            name="95% confidence interval",
        )
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Weekly demand")
    return fig


def loss_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for model_name in sorted(history["model"].unique()):
        model_history = history[history["model"] == model_name]
        fig.add_trace(
            go.Scatter(
                x=model_history["epoch"],
                y=model_history["train_mse"],
                mode="lines+markers",
                name=f"{model_name} train",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=model_history["epoch"],
                y=model_history["val_mse"],
                mode="lines+markers",
                name=f"{model_name} validation",
            )
        )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="MSE loss")
    return fig


def main() -> None:
    st.title("NeuralStock Demand Forecast Dashboard")
    df, metadata = load_data()

    with st.sidebar:
        st.header("Forecast Controls")
        categories = sorted(df["product_category"].dropna().unique().tolist())
        category = st.selectbox("Product category", categories)
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        date_range = st.date_input("Backtest date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if len(date_range) != 2:
            st.stop()
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        st.caption("Inference uses the saved LSTM and MLP checkpoints from `neural_stock/models`.")

    if not LSTM_PATH.exists() or not MLP_PATH.exists():
        st.error("Model weights are missing. Run `python train.py --epochs 30 --batch-size 64` first.")
        st.stop()

    start_time = time.perf_counter()
    backtest = run_backtest(df, metadata, category, start_date, end_date, LSTM_PATH.stat().st_mtime, MLP_PATH.stat().st_mtime)
    inference_seconds = time.perf_counter() - start_time

    if backtest.empty:
        st.warning("No enough rows are available in this date range to build sliding-window forecasts.")
        st.stop()

    weekly = weekly_summary(backtest)
    comparison = build_comparison(weekly)
    category_metrics, all_category_weekly = category_mape_table(
        df,
        metadata,
        start_date,
        end_date,
        LSTM_PATH.stat().st_mtime,
        MLP_PATH.stat().st_mtime,
    )
    all_category_comparison = build_comparison(all_category_weekly)
    forecast = build_forecast(df[df["date"] <= end_date], metadata, category, weekly)
    alerts = build_reorder_alerts(df[df["date"] <= end_date], metadata, category)

    selected_lstm_mape = comparison.loc[comparison["Model"] == "LSTM", "Weekly MAPE (%)"].iloc[0]
    overall_lstm_mape = all_category_comparison.loc[
        all_category_comparison["Model"] == "LSTM", "Weekly MAPE (%)"
    ].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("All-category LSTM MAPE", f"{overall_lstm_mape:.2f}%")
    col2.metric("Inference time", f"{inference_seconds:.2f}s")
    col3.metric("4-week forecast", f"{forecast['forecast'].sum():,.0f} units")
    col4.metric("Reorder alerts", f"{len(alerts)}")

    st.caption(f"Selected category LSTM MAPE: {selected_lstm_mape:.2f}%")
    if overall_lstm_mape <= 12:
        st.success("Acceptance target met: all-category weekly aggregated LSTM MAPE is below 12%.")
    else:
        st.warning(
            "All-category weekly aggregated LSTM MAPE is above 12%. "
            "Train for more epochs before using the dashboard for final evaluation."
        )

    tab_forecast, tab_compare, tab_alerts, tab_features = st.tabs(
        ["Forecast", "Model Comparison", "Reorder Alerts", "Feature Importance"]
    )

    with tab_forecast:
        st.subheader(f"{category} 4-week demand forecast")
        st.plotly_chart(forecast_chart(weekly.tail(12), forecast), use_container_width=True)
        st.dataframe(forecast, use_container_width=True)
        st.download_button(
            "Download forecast CSV",
            forecast.to_csv(index=False).encode("utf-8"),
            file_name=f"{category.lower()}_4_week_forecast.csv",
            mime="text/csv",
        )

    with tab_compare:
        st.subheader("LSTM vs MLP vs naive baseline")
        st.plotly_chart(line_chart(weekly), use_container_width=True)
        st.dataframe(comparison, use_container_width=True)
        st.subheader("Category-level MAPE report")
        st.dataframe(category_metrics, use_container_width=True)

        if HISTORY_PATH.exists():
            history = pd.read_csv(HISTORY_PATH)
            st.subheader("Training and validation loss curves")
            st.caption("Curves are exported from the same training loop that writes TensorBoard scalars.")
            st.plotly_chart(loss_chart(history), use_container_width=True)
        else:
            st.info("Run `python train.py --epochs 30 --batch-size 64` to export TensorBoard logs and loss history.")

        if COMPARISON_PATH.exists():
            st.download_button(
                "Download model comparison CSV",
                comparison.to_csv(index=False).encode("utf-8"),
                file_name=f"{category.lower()}_model_comparison.csv",
                mime="text/csv",
            )

    with tab_alerts:
        st.subheader("Inventory reorder alerts")
        if alerts.empty:
            st.success("No reorder alerts for the selected category.")
        else:
            st.dataframe(alerts, use_container_width=True)
            st.download_button(
                "Download reorder alerts CSV",
                alerts.to_csv(index=False).encode("utf-8"),
                file_name=f"{category.lower()}_reorder_alerts.csv",
                mime="text/csv",
            )

    with tab_features:
        st.subheader("Feature importance insights")
        importance = feature_importance(df, metadata, category)
        if importance.empty:
            st.info("No feature-importance data is available for this category.")
        else:
            top_importance = importance.head(15)
            insight_metrics, insight_sections = feature_insights(importance)

            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Top feature", insight_metrics["top_feature"])
            metric_b.metric("Top feature share", f"{insight_metrics['top_feature_importance']:.1%}")
            metric_c.metric("Top 5 share", f"{insight_metrics['top_five_share']:.1%}")

            st.markdown(
                f"**Strongest feature family:** {insight_metrics['strongest_group']} "
                f"({insight_metrics['strongest_group_importance']:.1%} total importance)"
            )

            for title, bullets in insight_sections:
                st.markdown(f"**{title}**")
                for bullet in bullets:
                    st.markdown(f"- {bullet}")

            group_summary = (
                importance.groupby("feature_group", as_index=False)["importance"]
                .sum()
                .sort_values("importance", ascending=False)
            )
            col_a, col_b = st.columns(2)
            with col_a:
                fig = go.Figure(
                    go.Bar(
                        x=top_importance["importance"][::-1],
                        y=top_importance["feature"][::-1],
                        orientation="h",
                        marker_color="#1f77b4",
                    )
                )
                fig.update_layout(
                    height=440,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title="Importance",
                    yaxis_title="",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                group_fig = go.Figure(
                    go.Bar(
                        x=group_summary["feature_group"],
                        y=group_summary["importance"],
                        marker_color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#7f7f7f"][
                            : len(group_summary)
                        ],
                    )
                )
                group_fig.update_layout(
                    height=440,
                    margin=dict(l=10, r=10, t=20, b=10),
                    yaxis_title="Total importance",
                    xaxis_title="",
                )
                st.plotly_chart(group_fig, use_container_width=True)

            st.dataframe(top_importance, use_container_width=True)
            st.download_button(
                "Download feature importance CSV",
                importance.to_csv(index=False).encode("utf-8"),
                file_name=f"{category.lower()}_feature_importance.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
