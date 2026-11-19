"""Data access for OTARF and the external evaluator (logic copied unchanged from the research code).

AirData       chronological, train-scaled windows with past-only masks and observation age
CIRFData      adds anchor validity and the real past anchor (OTARF input marks)
TrainingWithPrefix / TrainObserved   training-only anchor before the window and future masks for the loss
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from .io import read_json, sha256


class AirData:
    def __init__(self, directory):
        self.directory = directory
        self.meta = read_json(directory / "metadata.json")
        for name, key in [
            ("data.csv", "csv_sha256"),
            ("observations.npz", "observations_sha256"),
        ]:
            if sha256(directory / name) != self.meta[key]:
                raise ValueError("Data checksum mismatch: " + name)
        frame = pd.read_csv(directory / "data.csv")
        filled = frame[self.meta["columns"]].to_numpy(dtype=np.float64)
        self.scaler = StandardScaler().fit(filled[: self.meta["train_end"]])
        self.values = self.scaler.transform(filled).astype(np.float32)
        with np.load(directory / "observations.npz") as original:
            self.mask = original["observed"].astype(np.float32)
        age = np.empty_like(self.mask)
        previous = np.full(self.mask.shape[1], -1, dtype=np.int64)
        for t in range(len(age)):
            previous = np.where(self.mask[t].astype(bool), t, previous)
            age[t] = np.where(previous >= 0, t - previous, t + 1)
        # Ages use the available prefix, including past train history for val/test.
        self.age = age
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
        self.calendar = np.stack(
            [
                dates.hour / 23 - 0.5,
                dates.dayofweek / 6 - 0.5,
                (dates.day - 1) / 30 - 0.5,
                (dates.dayofyear - 1) / 365 - 0.5,
            ],
            axis=1,
        ).astype(np.float32)
        self.marks = np.concatenate([self.mask, age, self.calendar], axis=-1)

    def dataset(self, split, config, horizon):
        return Windows(self, split, config["seq_len"], config["label_len"], horizon)


class Windows(Dataset):
    def __init__(self, data, split, context, label, horizon):
        self.data, self.context, self.label, self.horizon = (
            data,
            context,
            label,
            horizon,
        )
        meta = data.meta
        self.start, self.end = {
            "train": (context, meta["train_end"]),
            "val": (meta["train_end"], meta["test_start"]),
            "test": (meta["test_start"], meta["rows"]),
        }[split]
        self.scaler = data.scaler
        if self.start < context or self.label > context or len(self) <= 0:
            raise ValueError("Invalid chronological window protocol")

    def __len__(self):
        return self.end - self.start - self.horizon + 1

    def __getitem__(self, index):
        t = self.start + index
        # Future masks/ages are deliberately NOT returned, even for training.
        return (
            self.data.values[t - self.context : t],
            self.data.values[t - self.label : t + self.horizon],
            self.data.marks[t - self.context : t],
            self.data.calendar[t - self.label : t + self.horizon],
        )


class EvaluationAdapter:
    """Only adapts the new model to the unchanged, aligned external evaluator."""

    def __init__(self, model, data, args, config, device):
        self.model, self.data, self.args, self.config, self.device = (
            model,
            data,
            args,
            config,
            device,
        )

    def _get_data(self, split):
        ds = self.data.dataset(split, self.config, self.args.pred_len)
        return ds, DataLoader(
            ds, batch_size=self.args.batch_size, shuffle=False, num_workers=0
        )


class CIRFData(AirData):
    def __init__(self, directory):
        super().__init__(directory)
        valid = np.maximum.accumulate(self.mask.astype(bool), axis=0)
        anchor = np.where(valid, self.values, 0.0).astype(np.float32)
        self.marks = np.concatenate(
            [self.mask, self.age, valid.astype(np.float32), anchor, self.calendar],
            axis=-1,
        )
        self.filler = np.median(self.values[: self.meta["train_end"]], axis=0).astype(
            np.float32
        )


class TrainingWithPrefix:
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, y, mark, ymark = self.base[index]
        data = self.base.data
        c = data.meta["channels"]
        previous = self.base.start + index - self.base.context - 1
        if previous < 0:
            prefix = np.zeros((4, c), dtype=np.float32)
            prefix[0] = data.filler
        else:
            valid = data.marks[previous, 2 * c : 3 * c]
            prefix = np.stack(
                [
                    np.where(valid > 0, data.values[previous], data.filler),
                    data.marks[previous, c : 2 * c],
                    valid,
                    data.marks[previous, 3 * c : 4 * c],
                ]
            ).astype(np.float32)
        return x, y, mark, ymark, prefix


class TrainObserved:
    def __init__(self, windows):
        self.windows = windows
        self.prefixed = TrainingWithPrefix(windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        x, y, mark, ymark, prefix = self.prefixed[index]
        origin = self.windows.start + index
        end = origin + self.windows.horizon
        if end > self.windows.data.meta["train_end"]:
            raise ValueError("Training target exceeds train boundary")
        return x, y, mark, ymark, prefix, self.windows.data.mask[origin:end]
