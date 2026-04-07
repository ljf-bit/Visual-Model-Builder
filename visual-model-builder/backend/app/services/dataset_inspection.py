"""Dataset inspection and normalization helpers for training datasets."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
KNOWN_SPLIT_NAMES = ("train", "val", "test")


@dataclass(slots=True)
class DatasetInspectionResult:
    """Serializable dataset inspection payload shared by API, validation, and runtime."""

    success: bool
    dataset_mode: str
    resolved_split_mode: str
    task_type: str
    sample_count: int
    num_classes: int
    class_names: list[str] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)
    input_shape: list[int] | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _coerce_numeric_list(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            decoded = [part.strip() for part in stripped.split(",") if part.strip()]
        value = decoded

    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
        numbers: list[float] = []
        for item in value:
            if isinstance(item, (int, float)):
                numbers.append(float(item))
                continue
            if isinstance(item, str) and item.strip():
                try:
                    numbers.append(float(item.strip()))
                except ValueError:
                    continue
        return numbers

    return []


def _coerce_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            decoded = [part.strip() for part in stripped.split(",") if part.strip()]
        value = decoded

    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item).strip() for item in value if str(item).strip()]

    return []


def normalize_dataset_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize Dataset node params so old projects stay compatible."""

    raw = dict(params or {})
    dataset_mode = str(raw.get("datasetMode") or "builtin").strip() or "builtin"
    dataset_name = str(raw.get("datasetName") or "FakeData").strip() or "FakeData"
    image_size = _safe_int(raw.get("imageSize"), 28)
    color_mode = str(raw.get("colorMode") or "grayscale").strip() or "grayscale"
    channel_count = 3 if color_mode == "rgb" else 1
    mean = _coerce_numeric_list(raw.get("mean"))
    std = _coerce_numeric_list(raw.get("std"))

    if not mean:
        mean = [0.5] * channel_count
    if not std:
        std = [0.5] * channel_count

    return {
        "datasetMode": dataset_mode,
        "datasetName": dataset_name,
        "trainSplit": _coerce_bool(raw.get("trainSplit"), True),
        "rootPath": str(raw.get("rootPath") or "").strip(),
        "splitMode": str(raw.get("splitMode") or "predefined").strip() or "predefined",
        "trainRatio": _safe_float(raw.get("trainRatio"), 0.7),
        "valRatio": _safe_float(raw.get("valRatio"), 0.2),
        "testRatio": _safe_float(raw.get("testRatio"), 0.1),
        "shuffleBeforeSplit": _coerce_bool(raw.get("shuffleBeforeSplit"), True),
        "imageSize": image_size,
        "colorMode": color_mode,
        "normalize": _coerce_bool(raw.get("normalize"), False),
        "mean": mean,
        "std": std,
        "augmentationEnabled": _coerce_bool(raw.get("augmentationEnabled"), False),
        "csvPath": str(raw.get("csvPath") or "").strip(),
        "labelColumn": str(raw.get("labelColumn") or "label").strip() or "label",
        "pathColumn": str(raw.get("pathColumn") or "image_path").strip() or "image_path",
        "featureColumns": _coerce_string_list(raw.get("featureColumns")),
        "taskType": str(raw.get("taskType") or "classification").strip() or "classification",
        "numClasses": _safe_int(raw.get("numClasses"), 10 if dataset_name == "MNIST" else 10),
    }


def build_dataset_input_shape(params: Mapping[str, Any] | None) -> list[int] | None:
    """Resolve the dataset-provided input shape after preprocessing."""

    normalized = normalize_dataset_params(params)
    image_size = int(normalized["imageSize"])
    if image_size <= 0:
        return None
    channels = 3 if normalized["colorMode"] == "rgb" else 1
    return [channels, image_size, image_size]


def resolve_normalize_stats(params: Mapping[str, Any] | None) -> tuple[list[float], list[float]]:
    """Return normalized mean/std arrays expanded to the configured channel count."""

    normalized = normalize_dataset_params(params)
    channels = 3 if normalized["colorMode"] == "rgb" else 1
    mean = list(normalized["mean"])
    std = list(normalized["std"])

    if len(mean) == 1 and channels > 1:
        mean = mean * channels
    if len(std) == 1 and channels > 1:
        std = std * channels
    return mean, std


def compute_ratio_split_counts(
    total_count: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, int]:
    """Compute deterministic split counts from train/val/test ratios."""

    raw_counts = [
        max(total_count * train_ratio, 0.0),
        max(total_count * val_ratio, 0.0),
        max(total_count * test_ratio, 0.0),
    ]
    floored = [math.floor(value) for value in raw_counts]
    remainder = total_count - sum(floored)

    indexed_fractions = sorted(
        enumerate(raw_counts),
        key=lambda item: item[1] - floored[item[0]],
        reverse=True,
    )
    for index, _ in indexed_fractions[:remainder]:
        floored[index] += 1

    return {
        "train": floored[0],
        "val": floored[1],
        "test": floored[2],
    }


def _list_visible_dirs(path: Path) -> list[Path]:
    return sorted(
        [child for child in path.iterdir() if child.is_dir() and not child.name.startswith(".")],
        key=lambda item: item.name.lower(),
    )


def _find_image_files(path: Path) -> list[Path]:
    return sorted(
        [
            child
            for child in path.rglob("*")
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS and not any(part.startswith(".") for part in child.parts)
        ],
        key=lambda item: str(item).lower(),
    )


def _scan_image_folder_classes(path: Path) -> tuple[dict[str, int], list[str], list[Path]]:
    class_counts: dict[str, int] = {}
    class_names: list[str] = []
    sample_paths: list[Path] = []

    for class_dir in _list_visible_dirs(path):
        image_files = _find_image_files(class_dir)
        if not image_files:
            continue
        class_counts[class_dir.name] = len(image_files)
        class_names.append(class_dir.name)
        sample_paths.append(image_files[0])

    return class_counts, sorted(class_names), sample_paths


def _verify_readable_image(sample_path: Path, errors: list[str]) -> None:
    try:
        from PIL import Image

        with Image.open(sample_path) as image:
            image.verify()
    except ModuleNotFoundError:
        return
    except Exception as exc:
        errors.append(f"Image sample `{sample_path}` could not be read: {exc}")


def _append_unique(items: list[str], message: str) -> None:
    if message and message not in items:
        items.append(message)


def _validate_image_dataset_config(
    normalized: Mapping[str, Any],
    input_shape: list[int] | None,
    warnings: list[str],
    errors: list[str],
) -> None:
    image_size = int(normalized["imageSize"])
    if image_size <= 0:
        _append_unique(errors, "Dataset `imageSize` must be a positive integer.")

    if normalized["colorMode"] not in {"grayscale", "rgb"}:
        _append_unique(errors, "Dataset `colorMode` must be `grayscale` or `rgb`.")

    if normalized["taskType"] not in {"classification", "regression"}:
        _append_unique(errors, "Dataset `taskType` must be `classification` or `regression`.")

    if normalized["normalize"]:
        mean, std = resolve_normalize_stats(normalized)
        expected_channels = input_shape[0] if input_shape else (3 if normalized["colorMode"] == "rgb" else 1)
        if len(mean) != expected_channels:
            _append_unique(
                errors,
                f"Dataset normalization mean must provide {expected_channels} value(s) for `{normalized['colorMode']}` mode.",
            )
        if len(std) != expected_channels:
            _append_unique(
                errors,
                f"Dataset normalization std must provide {expected_channels} value(s) for `{normalized['colorMode']}` mode.",
            )
        if any(value <= 0 for value in std):
            _append_unique(errors, "Dataset normalization std values must be greater than 0.")

    if normalized["datasetMode"] == "image_folder" and normalized["taskType"] != "classification":
        _append_unique(errors, "`image_folder` currently supports classification datasets only.")
    if normalized["datasetMode"] == "builtin" and normalized["taskType"] != "classification":
        _append_unique(errors, "Builtin teaching datasets currently support classification only.")


def _inspect_builtin_dataset(normalized: Mapping[str, Any]) -> DatasetInspectionResult:
    warnings: list[str] = []
    errors: list[str] = []
    input_shape = build_dataset_input_shape(normalized)
    _validate_image_dataset_config(normalized, input_shape, warnings, errors)

    dataset_name = str(normalized["datasetName"])
    if dataset_name not in {"FakeData", "MNIST"}:
        _append_unique(errors, "Builtin dataset must be `FakeData` or `MNIST`.")

    num_classes = int(normalized["numClasses"])
    if dataset_name == "MNIST":
        if num_classes != 10:
            _append_unique(errors, "MNIST always has 10 classes, so `numClasses` must stay at 10.")
        num_classes = 10
        class_names = [str(index) for index in range(10)]
        warnings.append("Builtin MNIST teaching runs currently cap the runtime dataset to a 256-sample subset.")
    else:
        if num_classes <= 0:
            _append_unique(errors, "Dataset `numClasses` must be a positive integer.")
        class_names = [str(index) for index in range(max(num_classes, 0))]
        warnings.append("Builtin FakeData teaching runs currently use a 256-sample synthetic subset.")

    return DatasetInspectionResult(
        success=len(errors) == 0,
        dataset_mode="builtin",
        resolved_split_mode="predefined",
        task_type=str(normalized["taskType"]),
        sample_count=256 if not errors else 0,
        num_classes=max(num_classes, 0),
        class_names=class_names,
        splits={"train": 256 if not errors else 0, "val": 0, "test": 0},
        input_shape=input_shape,
        warnings=warnings,
        errors=errors,
    )


def _inspect_image_folder_dataset(normalized: Mapping[str, Any]) -> DatasetInspectionResult:
    warnings: list[str] = []
    errors: list[str] = []
    input_shape = build_dataset_input_shape(normalized)
    _validate_image_dataset_config(normalized, input_shape, warnings, errors)

    root_path = Path(str(normalized["rootPath"] or "")).expanduser()
    if not str(normalized["rootPath"]).strip():
        _append_unique(errors, "`rootPath` is required when datasetMode is `image_folder`.")
        return DatasetInspectionResult(
            success=False,
            dataset_mode="image_folder",
            resolved_split_mode=str(normalized["splitMode"]),
            task_type=str(normalized["taskType"]),
            sample_count=0,
            num_classes=0,
            class_names=[],
            splits={"train": 0, "val": 0, "test": 0},
            input_shape=input_shape,
            warnings=warnings,
            errors=errors,
        )

    if not root_path.exists():
        _append_unique(errors, f"Dataset root path does not exist: `{root_path}`.")
    elif not root_path.is_dir():
        _append_unique(errors, f"Dataset root path must be a directory: `{root_path}`.")

    if errors:
        return DatasetInspectionResult(
            success=False,
            dataset_mode="image_folder",
            resolved_split_mode=str(normalized["splitMode"]),
            task_type=str(normalized["taskType"]),
            sample_count=0,
            num_classes=0,
            class_names=[],
            splits={"train": 0, "val": 0, "test": 0},
            input_shape=input_shape,
            warnings=warnings,
            errors=errors,
        )

    split_dirs = {name: root_path / name for name in KNOWN_SPLIT_NAMES if (root_path / name).is_dir()}
    requested_split_mode = str(normalized["splitMode"])
    resolved_split_mode = requested_split_mode
    splits = {"train": 0, "val": 0, "test": 0}
    class_names: list[str] = []
    sample_count = 0

    if split_dirs:
        resolved_split_mode = "predefined"
        if requested_split_mode == "ratio":
            _append_unique(warnings, "Explicit train/val/test folders were found, so `splitMode=ratio` was ignored.")

        train_class_names: list[str] = []
        union_class_names: set[str] = set()

        for split_name, split_dir in split_dirs.items():
            class_counts, split_class_names, sample_paths = _scan_image_folder_classes(split_dir)
            split_total = sum(class_counts.values())
            splits[split_name] = split_total
            sample_count += split_total
            union_class_names.update(split_class_names)

            if sample_paths:
                _verify_readable_image(sample_paths[0], errors)

            if split_name == "train":
                train_class_names = split_class_names
                if split_total == 0:
                    _append_unique(errors, "The `train` split exists but contains no readable image samples.")
                if not split_class_names:
                    _append_unique(errors, "The `train` split must contain at least one class directory with images.")
            elif split_total == 0:
                _append_unique(warnings, f"The `{split_name}` split exists but contains no readable image samples.")

            if split_name != "train" and train_class_names:
                extra_classes = sorted(set(split_class_names) - set(train_class_names))
                if extra_classes:
                    _append_unique(
                        warnings,
                        f"Split `{split_name}` contains classes not present in `train`: {', '.join(extra_classes)}.",
                    )

        if splits["train"] == 0:
            _append_unique(errors, "A predefined image folder dataset must provide a non-empty `train` split.")

        class_names = train_class_names or sorted(union_class_names)
    else:
        resolved_split_mode = "ratio"
        if requested_split_mode == "predefined":
            _append_unique(
                warnings,
                "No explicit train/val/test folders were found, so ratio splitting will be used on the class folders under `rootPath`.",
            )

        class_counts, class_names, sample_paths = _scan_image_folder_classes(root_path)
        sample_count = sum(class_counts.values())

        if sample_paths:
            _verify_readable_image(sample_paths[0], errors)

        if sample_count == 0:
            _append_unique(
                errors,
                "Image folder datasets must contain class subdirectories with readable image files.",
            )
        elif not class_names:
            _append_unique(
                errors,
                "Image folder datasets must expose at least one class directory under `rootPath`.",
            )

        train_ratio = float(normalized["trainRatio"])
        val_ratio = float(normalized["valRatio"])
        test_ratio = float(normalized["testRatio"])
        ratio_sum = train_ratio + val_ratio + test_ratio

        if train_ratio <= 0:
            _append_unique(errors, "`trainRatio` must be greater than 0 when using ratio splitting.")
        if any(value < 0 for value in (train_ratio, val_ratio, test_ratio)):
            _append_unique(errors, "Split ratios must be zero or greater.")
        if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            _append_unique(errors, "Split ratios must add up to 1.0.")

        if sample_count > 0 and not errors:
            splits = compute_ratio_split_counts(sample_count, train_ratio, val_ratio, test_ratio)
            if splits["train"] == 0:
                _append_unique(errors, "The computed training split is empty. Increase `trainRatio` or add more samples.")
            if splits["val"] == 0 and val_ratio > 0:
                _append_unique(warnings, "The computed validation split is empty because the dataset is very small.")
            if splits["test"] == 0 and test_ratio > 0:
                _append_unique(warnings, "The computed test split is empty because the dataset is very small.")

    if len(class_names) == 1:
        _append_unique(warnings, "Only one class was detected. Classification training will run, but it is a weak teaching example.")

    return DatasetInspectionResult(
        success=len(errors) == 0,
        dataset_mode="image_folder",
        resolved_split_mode=resolved_split_mode,
        task_type=str(normalized["taskType"]),
        sample_count=sample_count if len(errors) == 0 else sample_count,
        num_classes=len(class_names),
        class_names=class_names,
        splits=splits,
        input_shape=input_shape,
        warnings=warnings,
        errors=errors,
    )


def _inspect_csv_dataset(normalized: Mapping[str, Any]) -> DatasetInspectionResult:
    warnings: list[str] = [
        "`csv` datasets are inspectable, but training/code generation support is still a placeholder in this phase.",
    ]
    errors: list[str] = []
    csv_path = Path(str(normalized["csvPath"] or "")).expanduser()

    if not str(normalized["csvPath"]).strip():
        _append_unique(errors, "`csvPath` is required when datasetMode is `csv`.")
    elif not csv_path.exists():
        _append_unique(errors, f"CSV file does not exist: `{csv_path}`.")
    elif not csv_path.is_file():
        _append_unique(errors, f"CSV path must point to a file: `{csv_path}`.")

    input_shape: list[int] | None = None
    rows: list[dict[str, str]] = []
    field_names: list[str] = []
    if not errors:
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                field_names = list(reader.fieldnames or [])
                rows = list(reader)
        except Exception as exc:
            _append_unique(errors, f"CSV file could not be read: {exc}")

    path_column = str(normalized["pathColumn"])
    label_column = str(normalized["labelColumn"])
    feature_columns = list(normalized["featureColumns"])
    task_type = str(normalized["taskType"])

    for required_column in [path_column, label_column]:
        if required_column and field_names and required_column not in field_names:
            _append_unique(errors, f"CSV column `{required_column}` was not found.")

    if task_type == "classification":
        labels = sorted({str(row.get(label_column, "")).strip() for row in rows if str(row.get(label_column, "")).strip()})
        class_names = labels
        num_classes = len(class_names)
    else:
        class_names = []
        num_classes = 0
        if feature_columns:
            input_shape = [len(feature_columns)]
        else:
            _append_unique(warnings, "CSV regression preview could not infer inputShape because `featureColumns` is empty.")

    if task_type == "classification" and path_column and field_names and path_column in field_names:
        missing_paths = 0
        first_readable_sample: Path | None = None
        for row in rows:
            raw_path = str(row.get(path_column, "")).strip()
            if not raw_path:
                missing_paths += 1
                continue
            candidate = (csv_path.parent / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path)
            if not candidate.exists():
                missing_paths += 1
                continue
            if first_readable_sample is None:
                first_readable_sample = candidate

        if missing_paths:
            _append_unique(warnings, f"{missing_paths} CSV row(s) reference a missing image path.")
        if first_readable_sample is not None:
            _verify_readable_image(first_readable_sample, errors)
        input_shape = build_dataset_input_shape(normalized)
    elif task_type == "classification":
        input_shape = build_dataset_input_shape(normalized)

    return DatasetInspectionResult(
        success=len(errors) == 0,
        dataset_mode="csv",
        resolved_split_mode="predefined",
        task_type=task_type,
        sample_count=len(rows),
        num_classes=num_classes,
        class_names=class_names,
        splits={"train": len(rows), "val": 0, "test": 0},
        input_shape=input_shape,
        warnings=warnings,
        errors=errors,
    )


def inspect_dataset_config(params: Mapping[str, Any] | None) -> DatasetInspectionResult:
    """Inspect a Dataset node config and return metadata plus readable problems."""

    normalized = normalize_dataset_params(params)
    dataset_mode = str(normalized["datasetMode"])

    if dataset_mode == "builtin":
        return _inspect_builtin_dataset(normalized)
    if dataset_mode == "image_folder":
        return _inspect_image_folder_dataset(normalized)
    if dataset_mode == "csv":
        return _inspect_csv_dataset(normalized)

    return DatasetInspectionResult(
        success=False,
        dataset_mode=dataset_mode,
        resolved_split_mode=str(normalized["splitMode"]),
        task_type=str(normalized["taskType"]),
        sample_count=0,
        num_classes=0,
        class_names=[],
        splits={"train": 0, "val": 0, "test": 0},
        input_shape=build_dataset_input_shape(normalized),
        warnings=[],
        errors=[f"Unsupported datasetMode `{dataset_mode}`."],
    )
