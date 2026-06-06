"""Run demand forecasting inference with a trained LSTM or MLP model."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import torch

from model import FeedforwardMLP, SlidingWindowDataset, StackedLSTM


MODELS_DIR = Path("neural_stock/models")
DEFAULT_DATA_PATH = MODELS_DIR / "processed_data.pkl"
DEFAULT_METADATA_PATH = MODELS_DIR / "training_metadata.pkl"


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def load_metadata(metadata_path: Path, data_path: Path) -> dict:
    if metadata_path.exists():
        return load_pickle(metadata_path)

    data = load_pickle(data_path)
    if not isinstance(data, dict):
        raise ValueError("Data pickle must contain metadata or training_metadata.pkl must exist")

    feature_cols = data.get("feature_cols")
    if not feature_cols:
        raise ValueError("Could not find feature_cols in metadata or processed data")

    window_size = 7
    input_size = len(feature_cols)
    return {
        "feature_cols": feature_cols,
        "target_col": "units_sold",
        "window_size": window_size,
        "feature_scaler": data.get("scaler"),
        "demand_scaler": data.get("demand_scaler"),
        "lstm_config": {
            "input_size": input_size,
            "hidden_size": 128,
            "num_layers": 2,
            "output_size": 1,
            "dropout": 0.2,
        },
        "mlp_config": {
            "input_size": input_size * window_size,
            "hidden_sizes": [128, 64],
            "output_size": 1,
            "dropout": 0.2,
        },
    }


def load_dataframe(data_path: Path) -> pd.DataFrame:
    if data_path.suffix.lower() == ".pkl":
        data = load_pickle(data_path)
        if isinstance(data, dict) and "df" in data:
            return data["df"].copy()
        if isinstance(data, pd.DataFrame):
            return data.copy()
        raise ValueError("Pickle data must be a DataFrame or contain a 'df' key")

    return pd.read_csv(data_path)


def load_input_frame(input_json: str | None, data_path: Path) -> pd.DataFrame:
    if not input_json:
        return load_dataframe(data_path)

    input_path = Path(input_json)
    if input_path.exists():
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(input_json)

    if isinstance(payload, dict):
        payload = [payload]
    return pd.DataFrame(payload)


def scaler_input(scaler, df: pd.DataFrame, columns: list[str]):
    if getattr(scaler, "feature_names_in_", None) is not None:
        return df[columns]
    return df[columns].to_numpy()


def build_model(model_name: str, metadata: dict) -> torch.nn.Module:
    if model_name == "lstm":
        return StackedLSTM(**metadata["lstm_config"])
    if model_name == "mlp":
        return FeedforwardMLP(**metadata["mlp_config"])
    raise ValueError("model must be either 'lstm' or 'mlp'")


def load_model(model_name: str, model_path: Path, metadata: dict, device: torch.device) -> torch.nn.Module:
    model = build_model(model_name, metadata)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def latest_window(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window_size: int,
    product_id: str | None,
) -> pd.DataFrame:
    working = df.copy()
    if product_id and "product_id" in working.columns:
        working = working[working["product_id"].astype(str) == str(product_id)]
        if working.empty:
            raise ValueError(f"No rows found for product_id={product_id}")

    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"])
        sort_cols = ["date"]
        if "product_id" in working.columns:
            sort_cols = ["product_id", "date"]
        working = working.sort_values(sort_cols).reset_index(drop=True)

    if len(working) < window_size + 1:
        raise ValueError(f"Need at least {window_size + 1} rows to build one forecast window")

    return working.tail(window_size + 1).copy()


def predict(
    model: torch.nn.Module,
    model_name: str,
    window_df: pd.DataFrame,
    metadata: dict,
    device: torch.device,
) -> float:
    feature_cols = metadata["feature_cols"]
    prepared_window = window_df.copy()
    prepared_window[feature_cols] = prepared_window[feature_cols].apply(pd.to_numeric, errors="coerce")
    prepared_window[feature_cols] = prepared_window[feature_cols].ffill().bfill().fillna(0.0)

    feature_scaler = metadata.get("feature_scaler")
    if feature_scaler is not None:
        prepared_window[feature_cols] = feature_scaler.transform(
            scaler_input(feature_scaler, prepared_window, feature_cols)
        )

    dataset = SlidingWindowDataset(
        prepared_window,
        window_size=metadata["window_size"],
        target_col=metadata["target_col"],
        feature_cols=feature_cols,
        group_cols=[],
    )

    x, _ = dataset[0]
    x = x.unsqueeze(0).to(device)
    if model_name == "mlp":
        x = x.flatten(start_dim=1)

    with torch.no_grad():
        prediction = model(x).cpu().item()

    scaler = metadata.get("demand_scaler")
    if scaler is not None:
        prediction = scaler.inverse_transform([[prediction]])[0][0]

    return max(0.0, float(prediction))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a demand forecast from a trained model.")
    parser.add_argument("--model", choices=["lstm", "mlp"], default="lstm")
    parser.add_argument("--model-path", default=None, help="Path to model weights. Defaults by model type.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Path to processed data pickle or CSV.")
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--product-id", default=None, help="Optional product_id to forecast from the latest rows.")
    parser.add_argument(
        "--input-json",
        default=None,
        help="JSON list/dict or a JSON file path containing rows with the trained feature columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    metadata = load_metadata(Path(args.metadata_path), data_path)

    default_model_path = MODELS_DIR / f"best_{args.model}_model.pth"
    model_path = Path(args.model_path) if args.model_path else default_model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, model_path, metadata, device)

    df = load_input_frame(args.input_json, data_path)
    window_df = latest_window(
        df,
        metadata["feature_cols"],
        metadata["target_col"],
        metadata["window_size"],
        args.product_id,
    )
    prediction = predict(model, args.model, window_df, metadata, device)

    label = f" for product {args.product_id}" if args.product_id else ""
    print(f"Predicted demand{label}: {prediction:.2f} units")


if __name__ == "__main__":
    main()
