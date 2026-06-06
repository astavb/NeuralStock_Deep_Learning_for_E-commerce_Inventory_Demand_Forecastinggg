"""Training script for NeuralStock demand forecasting models."""

from __future__ import annotations # for Python 3.10+ type hinting

import argparse # for command-line argument parsing
import pickle # for saving/loading processed data and metadata
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, random_split
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - dashboard can still use the CSV history.
    SummaryWriter = None

from model import FeedforwardMLP, SlidingWindowDataset, StackedLSTM


TARGET_COLUMN = "units_sold" # the column we want to predict, can be overridden by command-line argument
DEFAULT_WINDOW_SIZE = 7 # number of past days to use for prediction, can be overridden by command-line argument
DEFAULT_MODELS_DIR = Path("neural_stock/models") # default directory for saving models and metadata, can be overridden by command-line argument
DEFAULT_PROCESSED_PATH = DEFAULT_MODELS_DIR / "processed_data.pkl" # default path for processed data pickle, can be overridden by command-line argument
DEFAULT_CSV_PATH = Path("cleaned_data/cleaned_data.csv") # default path for cleaned CSV data, used if processed pickle is not found and no path is provided, can be overridden by command-line argument


# why set_seed is important: ensures reproducibility of results by controlling random number generation across libraries

def set_seed(seed: int) -> None: # for reproducibility of results
    random.seed(seed) # set seed for built-in random module
    np.random.seed(seed) # set seed for numpy random functions
    torch.manual_seed(seed) # set seed for PyTorch CPU
    if torch.cuda.is_available(): # set seed for all CUDA devices (GPUs)
        torch.cuda.manual_seed_all(seed) # set seed for PyTorch CUDA


def resolve_data_path(path: str | None) -> Path: # determine which data file to load based on command-line argument and defaults
    if path:
        return Path(path)
    if DEFAULT_PROCESSED_PATH.exists():
        return DEFAULT_PROCESSED_PATH
    return DEFAULT_CSV_PATH


def load_data(path: Path) -> tuple[pd.DataFrame, list[str] | None, MinMaxScaler | None, MinMaxScaler | None]:
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as file:
            data = pickle.load(file)
        if not isinstance(data, dict) or "df" not in data:
            raise ValueError("Processed pickle must contain a dictionary with a 'df' key")
        return (
            data["df"].copy(),
            list(data.get("feature_cols", [])) or None,
            data.get("scaler"),
            data.get("demand_scaler"),
        )

    df = pd.read_csv(path)
    return df, None, None, None


def infer_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    non_features = {target_col, "date", "product_id", "product_category", "day_of_week", "price_category"}
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [col for col in numeric_cols if col not in non_features]


def scaler_input(scaler: MinMaxScaler, df: pd.DataFrame, columns: list[str]):
    if getattr(scaler, "feature_names_in_", None) is not None:
        return df[columns]
    return df[columns].to_numpy()


def prepare_training_frame(
    df: pd.DataFrame,
    feature_cols: list[str] | None,
    target_col: str,
    feature_scaler: MinMaxScaler | None,
    demand_scaler: MinMaxScaler | None,
) -> tuple[pd.DataFrame, list[str], MinMaxScaler, MinMaxScaler]:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    selected_features = feature_cols or infer_feature_cols(df, target_col)
    df[selected_features] = df[selected_features].apply(pd.to_numeric, errors="coerce")
    df[selected_features] = df[selected_features].ffill().bfill().fillna(0.0)
    x_scaler = feature_scaler or MinMaxScaler()
    if feature_scaler is None:
        x_scaler.fit(df[selected_features].to_numpy())
    df[selected_features] = x_scaler.transform(scaler_input(x_scaler, df, selected_features))

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").ffill().bfill().fillna(0.0)
    y_scaler = demand_scaler or MinMaxScaler()
    if demand_scaler is None:
        y_scaler.fit(df[[target_col]].to_numpy())
    df[target_col] = y_scaler.transform(scaler_input(y_scaler, df, [target_col])).reshape(-1)

    return df, selected_features, x_scaler, y_scaler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item() * x_batch.size(0)

    return total_loss / len(loader.dataset)


def fit_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    output_path: Path,
    writer=None,
) -> tuple[float, list[dict]]:
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_val_loss = float("inf")
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            torch.save(model.state_dict(), output_path.with_suffix(".pt"))

        lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "model": name,
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
                "learning_rate": lr,
            }
        )
        if writer is not None:
            writer.add_scalar(f"{name}/train_mse", train_loss, epoch)
            writer.add_scalar(f"{name}/val_mse", val_loss, epoch)
            writer.add_scalar(f"{name}/learning_rate", lr, epoch)

        print(
            f"{name} | Epoch {epoch:03d}/{epochs} | "
            f"train_mse={train_loss:.6f} | val_mse={val_loss:.6f} | lr={lr:.6f}"
        )

    return best_val_loss, history


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    demand_scaler: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[float] = []
    y_pred: list[float] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            predictions = model(x_batch.to(device)).cpu().numpy()
            y_pred.extend(predictions.reshape(-1).tolist())
            y_true.extend(y_batch.numpy().reshape(-1).tolist())

    y_true_arr = demand_scaler.inverse_transform(np.asarray(y_true).reshape(-1, 1)).reshape(-1)
    y_pred_arr = demand_scaler.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).reshape(-1)
    return y_true_arr, np.clip(y_pred_arr, 0, None)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true), 1e-5)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def save_training_metadata(
    path: Path,
    feature_cols: list[str],
    target_col: str,
    window_size: int,
    demand_scaler: MinMaxScaler,
    feature_scaler: MinMaxScaler,
    lstm_config: dict,
    mlp_config: dict,
) -> None:
    metadata = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "window_size": window_size,
        "feature_scaler": feature_scaler,
        "demand_scaler": demand_scaler,
        "lstm_config": lstm_config,
        "mlp_config": mlp_config,
    }
    with path.open("wb") as file:
        pickle.dump(metadata, file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM and MLP demand forecasting models.")
    parser.add_argument("--data-path", default=None, help="Path to processed_data.pkl or cleaned CSV.")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="Directory for saved model files.")
    parser.add_argument("--target-col", default=TARGET_COLUMN)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--log-dir", default="runs/neural_stock", help="TensorBoard log directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    data_path = resolve_data_path(args.data_path)
    df, feature_cols, feature_scaler, demand_scaler = load_data(data_path)
    train_df, feature_cols, feature_scaler, demand_scaler = prepare_training_frame(
        df,
        feature_cols,
        args.target_col,
        feature_scaler,
        demand_scaler,
    )

    dataset = SlidingWindowDataset(
        train_df,
        window_size=args.window_size,
        target_col=args.target_col,
        feature_cols=feature_cols,
    )

    train_size = int(len(dataset) * args.train_ratio)
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError("Dataset is too small for the requested train/validation split")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    input_size = len(feature_cols)
    lstm_config = {
        "input_size": input_size,
        "hidden_size": args.lstm_hidden_size,
        "num_layers": args.lstm_layers,
        "output_size": 1,
        "dropout": args.dropout,
    }
    mlp_config = {
        "input_size": input_size * args.window_size,
        "hidden_sizes": [128, 64],
        "output_size": 1,
        "dropout": args.dropout,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded {len(dataset)} windows from {data_path}")
    print(f"Training on {device}")
    writer = SummaryWriter(args.log_dir) if SummaryWriter is not None else None

    lstm = StackedLSTM(**lstm_config)
    lstm_loss, lstm_history = fit_model(
        "LSTM",
        lstm,
        train_loader,
        val_loader,
        args.epochs,
        args.learning_rate,
        device,
        models_dir / "best_lstm_model.pth",
        writer,
    )

    mlp = FeedforwardMLP(**mlp_config)
    mlp_loss, mlp_history = fit_model(
        "MLP",
        mlp,
        train_loader,
        val_loader,
        args.epochs,
        args.learning_rate,
        device,
        models_dir / "best_mlp_model.pth",
        writer,
    )

    history = pd.DataFrame(lstm_history + mlp_history)
    history.to_csv(models_dir / "training_history.csv", index=False)

    lstm.load_state_dict(torch.load(models_dir / "best_lstm_model.pth", map_location=device))
    mlp.load_state_dict(torch.load(models_dir / "best_mlp_model.pth", map_location=device))
    lstm_true, lstm_pred = collect_predictions(lstm, val_loader, device, demand_scaler)
    mlp_true, mlp_pred = collect_predictions(mlp, val_loader, device, demand_scaler)
    comparison = pd.DataFrame(
        {
            "model": ["LSTM", "MLP"],
            "validation_mse": [lstm_loss, mlp_loss],
            "validation_mape": [mape(lstm_true, lstm_pred), mape(mlp_true, mlp_pred)],
        }
    )
    comparison.to_csv(models_dir / "model_comparison.csv", index=False)
    if writer is not None:
        for _, row in comparison.iterrows():
            writer.add_scalar(f"{row['model']}/validation_mape", row["validation_mape"], args.epochs)
        writer.flush()
        writer.close()

    save_training_metadata(
        models_dir / "training_metadata.pkl",
        feature_cols,
        args.target_col,
        args.window_size,
        demand_scaler,
        feature_scaler,
        lstm_config,
        mlp_config,
    )

    print(f"Saved best LSTM model to {models_dir / 'best_lstm_model.pth'} (val MSE: {lstm_loss:.6f})")
    print(f"Saved best MLP model to {models_dir / 'best_mlp_model.pth'} (val MSE: {mlp_loss:.6f})")
    print(f"Saved training metadata to {models_dir / 'training_metadata.pkl'}")
    print(f"Saved TensorBoard logs to {args.log_dir}")
    print(f"Saved loss history to {models_dir / 'training_history.csv'}")


if __name__ == "__main__":
    main()
