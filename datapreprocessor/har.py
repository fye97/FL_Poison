import os
from dataclasses import dataclass
from typing import Optional, Tuple

import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset

import urllib.request
import zipfile


@dataclass(frozen=True)
class _HarPaths:
    base_dir: str
    x_train: str
    y_train: str
    x_test: str
    y_test: str


def _find_uci_har_dir(root: str) -> Optional[str]:
    """
    Try common extraction layouts for the UCI HAR Dataset.

    Expected structure (zip default):
      <root>/UCI HAR Dataset/train/X_train.txt
    """
    candidates = [
        os.path.join(root, "UCI HAR Dataset"),
        os.path.join(root, "UCI_HAR_Dataset"),
        os.path.join(root, "uci_har_dataset"),
        os.path.join(root, "har"),
    ]
    for d in candidates:
        if os.path.isdir(os.path.join(d, "train")) and os.path.isdir(os.path.join(d, "test")):
            if os.path.exists(os.path.join(d, "train", "X_train.txt")):
                return d
    return None


_UCI_HAR_ZIP_URLS = [
    # Old-but-stable direct file link (kept for backward compatibility by UCI).
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
    # New UCI website download link (redirect target as of 2026-02).
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
]


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _validate_har_zip(zip_path: str, expected_sha256: Optional[str] = None) -> None:
    if expected_sha256:
        actual = _sha256_file(zip_path)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                f"HAR zip sha256 mismatch. expected={expected_sha256} actual={actual}"
            )

    with zipfile.ZipFile(zip_path, "r") as zf:
        required = [
            "UCI HAR Dataset/train/X_train.txt",
            "UCI HAR Dataset/train/y_train.txt",
            "UCI HAR Dataset/test/X_test.txt",
            "UCI HAR Dataset/test/y_test.txt",
        ]
        names = set(zf.namelist())
        missing = [p for p in required if p not in names]
        if missing:
            raise ValueError(f"HAR zip missing files: {missing}")
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"HAR zip failed CRC check at: {bad}")


def _safe_extract_zip(zip_path: str, extract_to: str) -> None:
    extract_root = os.path.realpath(extract_to)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = os.path.realpath(os.path.join(extract_to, member.filename))
            if not member_path.startswith(extract_root + os.sep) and member_path != extract_root:
                raise ValueError(f"Refusing to extract path outside target dir: {member.filename}")
        zf.extractall(extract_to)


def _download_file(url: str, dst_path: str, timeout: int = 60) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "FL_Poison-HAR/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dst_path, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _download_and_prepare(root: str, expected_sha256: Optional[str] = None) -> None:
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "UCI HAR Dataset.zip")
    tmp_path = zip_path + ".part"

    # If an existing zip validates, reuse it.
    if os.path.exists(zip_path):
        try:
            _validate_har_zip(zip_path, expected_sha256=expected_sha256)
        except Exception:
            try:
                os.remove(zip_path)
            except Exception:
                pass

    if not os.path.exists(zip_path):
        last_err: Optional[Exception] = None
        for url in _UCI_HAR_ZIP_URLS:
            try:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                print(f"Downloading UCI HAR dataset from: {url}")
                _download_file(url, tmp_path)
                os.replace(tmp_path, zip_path)
                _validate_har_zip(zip_path, expected_sha256=expected_sha256)
                break
            except Exception as e:
                last_err = e
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                try:
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                except Exception:
                    pass
        else:
            raise RuntimeError(f"Failed to download HAR dataset. Last error: {last_err}")

    _safe_extract_zip(zip_path, root)


def _resolve_paths(root: str, download: bool = False, expected_sha256: Optional[str] = None) -> _HarPaths:
    base_dir = _find_uci_har_dir(root)
    if base_dir is None and download:
        _download_and_prepare(root, expected_sha256=expected_sha256)
        base_dir = _find_uci_har_dir(root)
    if base_dir is None:
        raise FileNotFoundError(
            "HAR dataset not found.\n"
            "Download the UCI HAR Dataset and extract it under `./data/` so one of these exists:\n"
            "- `./data/UCI HAR Dataset/train/X_train.txt`\n"
            "- `./data/UCI_HAR_Dataset/train/X_train.txt`"
        )
    return _HarPaths(
        base_dir=base_dir,
        x_train=os.path.join(base_dir, "train", "X_train.txt"),
        y_train=os.path.join(base_dir, "train", "y_train.txt"),
        x_test=os.path.join(base_dir, "test", "X_test.txt"),
        y_test=os.path.join(base_dir, "test", "y_test.txt"),
    )


def _load_txt_matrix(path: str, dtype=np.float32) -> np.ndarray:
    # The original files are whitespace-separated.
    return np.loadtxt(path, dtype=dtype)


def _load_txt_vector(path: str, dtype=np.int64) -> np.ndarray:
    return np.loadtxt(path, dtype=dtype).reshape(-1)


def _zscore_fit(x_train: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean, std


def _zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


class HAR(Dataset):
    """
    UCI Human Activity Recognition (HAR) dataset.

    Notes:
    - Expects the dataset to be present locally under `root` (no network download).
    - Returns `(features, label)` where `features` is float32 tensor of shape [561],
      and `label` is int64 tensor in [0..5].
    """

    def __init__(self, root: str = "./data", train: bool = True, download: bool = False, normalize: bool = True):
        self.root = root
        self.train = train
        self.normalize = normalize

        expected_sha256 = os.environ.get("FLPOISON_HAR_SHA256")
        paths = _resolve_paths(root, download=download, expected_sha256=expected_sha256)
        cache_path = os.path.join(root, "uci_har_cache_v1.npz")

        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            x_train = cached["x_train"]
            y_train = cached["y_train"]
            x_test = cached["x_test"]
            y_test = cached["y_test"]
            mean = cached["mean"]
            std = cached["std"]
        else:
            x_train = _load_txt_matrix(paths.x_train, dtype=np.float32)
            y_train = _load_txt_vector(paths.y_train, dtype=np.int64) - 1  # 1..6 -> 0..5
            x_test = _load_txt_matrix(paths.x_test, dtype=np.float32)
            y_test = _load_txt_vector(paths.y_test, dtype=np.int64) - 1

            mean, std = _zscore_fit(x_train)
            if normalize:
                x_train = _zscore_apply(x_train, mean, std).astype(np.float32, copy=False)
                x_test = _zscore_apply(x_test, mean, std).astype(np.float32, copy=False)

            # Best-effort cache for faster restarts.
            try:
                np.savez_compressed(
                    cache_path,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    mean=mean.astype(np.float32, copy=False),
                    std=std.astype(np.float32, copy=False),
                )
            except Exception:
                pass

        x = x_train if train else x_test
        y = y_train if train else y_test

        # Keep internal storage as torch tensors for fast indexing.
        self._x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self._y = torch.from_numpy(np.asarray(y, dtype=np.int64))

        self.targets = self._y  # for partitioning utilities
        self.classes = list(range(6))

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int):
        features = self._x[idx]
        target = self._y[idx]
        return features, target
