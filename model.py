"""Model architecture and sliding-window dataset utilities."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """Create sequence samples for one-step-ahead demand forecasting.

    Each item returns:
        x: tensor with shape (window_size, number_of_features)
        y: tensor with shape (1,)
    """

    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        window_size: int,
        target_col: str = "units_sold",
        feature_cols: Sequence[str] | None = None,
        group_cols: str | Sequence[str] | None = None,
        date_col: str = "date",
        horizon: int = 1,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")

        self.window_size = window_size
        self.target_col = target_col
        self.horizon = horizon

        if isinstance(data, pd.DataFrame):
            self.feature_cols = list(feature_cols or self._infer_feature_cols(data, target_col))
            frames = self._split_dataframe(data, group_cols, date_col)
            xs, ys = self._build_from_frames(frames)
        else:
            if feature_cols is None:
                raise ValueError("feature_cols must be provided when data is a numpy array")
            self.feature_cols = list(feature_cols)
            array = np.asarray(data, dtype=np.float32)
            xs, ys = self._build_from_arrays([(array[:, :-1], array[:, -1])])

        if not xs:
            raise ValueError(
                "No sliding-window samples were created. Check window_size, horizon, "
                "and the number of rows in each product/date group."
            )

        self.x = torch.tensor(np.stack(xs), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(ys, dtype=np.float32).reshape(-1, 1), dtype=torch.float32)

    @staticmethod
    def _infer_feature_cols(data: pd.DataFrame, target_col: str) -> list[str]:
        numeric_cols = data.select_dtypes(include=["number", "bool"]).columns.tolist()
        return [col for col in numeric_cols if col != target_col]

    @staticmethod
    def _normalise_group_cols(
        data: pd.DataFrame,
        group_cols: str | Sequence[str] | None,
    ) -> list[str]:
        if group_cols is None:
            return ["product_id"] if "product_id" in data.columns else []
        if isinstance(group_cols, str):
            return [group_cols]
        return list(group_cols)

    def _split_dataframe(
        self,
        data: pd.DataFrame,
        group_cols: str | Sequence[str] | None,
        date_col: str,
    ) -> list[pd.DataFrame]:
        df = data.copy()
        sort_cols = []
        groups = self._normalise_group_cols(df, group_cols)
        sort_cols.extend([col for col in groups if col in df.columns])
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            sort_cols.append(date_col)
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        if groups:
            return [group.copy() for _, group in df.groupby(groups, sort=False)]
        return [df]

    def _build_from_frames(self, frames: Iterable[pd.DataFrame]) -> tuple[list[np.ndarray], list[float]]:
        arrays = []
        for frame in frames:
            feature_frame = frame[self.feature_cols].apply(pd.to_numeric, errors="coerce")
            target_series = pd.to_numeric(frame[self.target_col], errors="coerce")

            feature_frame = feature_frame.ffill().bfill().fillna(0.0)
            target_series = target_series.ffill().bfill().fillna(0.0)

            arrays.append(
                (
                    feature_frame.to_numpy(dtype=np.float32),
                    target_series.to_numpy(dtype=np.float32),
                )
            )
        return self._build_from_arrays(arrays)

    def _build_from_arrays(
        self,
        arrays: Iterable[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[list[np.ndarray], list[float]]:
        xs: list[np.ndarray] = []
        ys: list[float] = []

        for features, target in arrays:
            rows_needed = self.window_size + self.horizon
            if len(features) < rows_needed:
                continue

            last_start = len(features) - rows_needed + 1
            for start in range(last_start):
                target_index = start + self.window_size + self.horizon - 1
                xs.append(features[start : start + self.window_size])
                ys.append(float(target[target_index]))

        return xs, ys

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class StackedLSTM(nn.Module):
    """Stacked LSTM for sequence forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("StackedLSTM expects input with shape (batch, window, features)")
        output, _ = self.lstm(x)
        last_timestep = output[:, -1, :]
        return self.regressor(last_timestep)


class FeedforwardMLP(nn.Module):
    """Feedforward baseline that consumes a flattened sliding window."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] = (128, 64),
        output_size: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 3:
            x = torch.flatten(x, start_dim=1)
        return self.network(x)
