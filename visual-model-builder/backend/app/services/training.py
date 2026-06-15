"""Training graph helpers and runtime execution for Phase 2."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

from app.services.graph_ir import IRGraph, IRNode, topological_sort


@dataclass(slots=True)
class TrainingComponents:
    """Resolved training nodes from the graph."""

    input_node: IRNode | None
    output_node: IRNode | None
    dataset_node: IRNode | None
    dataloader_node: IRNode | None
    loss_node: IRNode | None
    optimizer_node: IRNode | None
    trainer_node: IRNode | None
    metric_node: IRNode | None
    model_nodes: list[IRNode]


def get_node_map(ir_graph: IRGraph) -> dict[str, IRNode]:
    """Return IR nodes keyed by node id."""

    return {node.node_id: node for node in ir_graph.nodes}


def has_training_nodes(ir_graph: IRGraph) -> bool:
    """Whether the graph contains Phase 2 nodes."""

    return any(node.category in {"data", "training"} for node in ir_graph.nodes)


def get_nodes_by_op(ir_graph: IRGraph, op: str) -> list[IRNode]:
    """Return all nodes matching a specific op."""

    return [node for node in ir_graph.nodes if node.op == op]


def get_model_nodes_in_order(ir_graph: IRGraph) -> list[IRNode]:
    """Return model nodes in topological order."""

    sorted_ids = topological_sort(ir_graph) or []
    node_map = get_node_map(ir_graph)
    return [node_map[node_id] for node_id in sorted_ids if node_map[node_id].category == "model"]


def resolve_training_components(ir_graph: IRGraph) -> TrainingComponents:
    """Resolve the single training pipeline supported in Phase 2."""

    def first(op: str) -> IRNode | None:
        matches = get_nodes_by_op(ir_graph, op)
        return matches[0] if matches else None

    return TrainingComponents(
        input_node=first("Input"),
        output_node=first("Output"),
        dataset_node=first("Dataset"),
        dataloader_node=first("DataLoader"),
        loss_node=first("Loss"),
        optimizer_node=first("Optimizer"),
        trainer_node=first("Trainer"),
        metric_node=first("Metric"),
        model_nodes=get_model_nodes_in_order(ir_graph),
    )


def get_input_shape(ir_graph: IRGraph) -> list[int]:
    """Return the model input shape."""

    components = resolve_training_components(ir_graph)
    shape = components.input_node.params.get("inputShape") if components.input_node else None
    if isinstance(shape, list) and len(shape) == 3:
        return [int(value) for value in shape]
    return [1, 28, 28]


def _normalize_sample_output_shape(output_shape: list[int] | None) -> list[int] | None:
    """Mirror runtime class-logit normalization on sample-relative shapes."""

    if not output_shape:
        return output_shape

    normalized_shape = list(output_shape)
    while len(normalized_shape) > 1 and normalized_shape[-1] == 1:
        normalized_shape.pop()
    return normalized_shape


def get_num_classes(ir_graph: IRGraph, node_results: dict[str, Any] | None = None) -> int:
    """Infer effective classifier output size after runtime normalization semantics."""

    components = resolve_training_components(ir_graph)
    output_node = components.output_node
    resolved_node_results = node_results

    if resolved_node_results is None:
        from app.services.shape_infer import infer_graph_shapes

        resolved_node_results = infer_graph_shapes(ir_graph)

    if output_node and output_node.node_id in resolved_node_results:
        output_shape = resolved_node_results[output_node.node_id].output_shape
        normalized_output_shape = _normalize_sample_output_shape(output_shape)
        if normalized_output_shape:
            return int(normalized_output_shape[0])

    linear_nodes = [node for node in get_model_nodes_in_order(ir_graph) if node.op == "Linear"]
    if linear_nodes:
        return int(linear_nodes[-1].params.get("out_features", 10))
    return 10


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_.")
    return slug or "project"


def _get_training_runs_root() -> Path:
    configured_root = os.environ.get("VMB_TRAINING_RUNS_DIR")
    if configured_root:
        return Path(configured_root).resolve()
    return (Path(__file__).resolve().parents[2] / "training_runs").resolve()


def _stable_hash(payload: object) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _load_torch_runtime():
    """Import torch/torchvision lazily so the API can start without training deps."""

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the current backend environment. "
            "Install `torch` and `torchvision` before using /run-training."
        ) from exc

    return torch, nn, DataLoader, Subset, datasets, transforms


def _parse_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "train", "yes", "1"}:
            return True
        if lowered in {"false", "test", "no", "0"}:
            return False
    return default


def _build_torch_layer(node: IRNode, nn: Any):
    params = node.params
    if node.op == "Conv2d":
        return nn.Conv2d(
            int(params.get("in_channels", 1)),
            int(params.get("out_channels", 16)),
            kernel_size=int(params.get("kernel_size", 3)),
            stride=int(params.get("stride", 1)),
            padding=int(params.get("padding", 0)),
        )
    if node.op == "ReLU":
        return nn.ReLU(inplace=bool(params.get("inplace", False)))
    if node.op == "MaxPool2d":
        return nn.MaxPool2d(
            kernel_size=int(params.get("kernel_size", 2)),
            stride=int(params.get("stride", 2)),
            padding=int(params.get("padding", 0)),
        )
    if node.op == "Flatten":
        start_dim = int(params.get("start_dim", 0))
        if start_dim >= 0:
            # Graph params describe a single sample shape; runtime tensors add a batch axis.
            start_dim += 1
        return nn.Flatten(
            start_dim=start_dim,
            end_dim=int(params.get("end_dim", -1)),
        )
    if node.op == "Linear":
        return nn.Linear(
            int(params.get("in_features", 128)),
            int(params.get("out_features", 10)),
            bias=bool(params.get("bias", True)),
        )
    raise ValueError(f"Unsupported model node for runtime: {node.op}")


def build_runtime_model(ir_graph: IRGraph):
    """Instantiate a sequential runtime model from model nodes."""

    _, nn, _, _, _, _ = _load_torch_runtime()
    modules = [
        _build_torch_layer(node, nn)
        for node in get_model_nodes_in_order(ir_graph)
        if node.op not in {"Input", "Output"}
    ]

    class GraphModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(modules)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return GraphModel()


def build_runtime_loss(ir_graph: IRGraph):
    """Instantiate the selected loss."""

    _, nn, _, _, _, _ = _load_torch_runtime()
    components = resolve_training_components(ir_graph)
    loss_type = str(components.loss_node.params.get("lossType", "CrossEntropyLoss")) if components.loss_node else "CrossEntropyLoss"
    if loss_type == "MSELoss":
        return nn.MSELoss()
    return nn.CrossEntropyLoss()


def build_runtime_optimizer(ir_graph: IRGraph, model):
    """Instantiate the selected optimizer."""

    torch, _, _, _, _, _ = _load_torch_runtime()
    components = resolve_training_components(ir_graph)
    params = components.optimizer_node.params if components.optimizer_node else {}
    optimizer_type = str(params.get("optimizerType", "Adam"))
    lr = float(params.get("lr", 0.001))
    weight_decay = float(params.get("weightDecay", 0.0))

    if optimizer_type == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(params.get("momentum", 0.0)),
        )

    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _resolve_device(device_value: object):
    torch, _, _, _, _, _ = _load_torch_runtime()
    device_name = str(device_value or "cpu")
    if device_name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _maybe_accuracy(metric_node: IRNode | None, outputs, labels) -> float | None:
    if metric_node is None:
        return None
    predictions = outputs.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def _normalize_classification_outputs(outputs):
    """Collapse trailing singleton axes so sample-relative class logits stay trainable in batches."""

    while outputs.ndim > 2 and outputs.shape[-1] == 1:
        outputs = outputs.squeeze(-1)
    return outputs


def compute_classification_metrics(
    predictions: list[int],
    labels: list[int],
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict[str, object]:
    """Compute classification metrics without pulling in sklearn."""

    resolved_classes = max(int(num_classes), 1)
    confusion_matrix = [[0 for _ in range(resolved_classes)] for _ in range(resolved_classes)]
    for target, predicted in zip(labels, predictions):
        if 0 <= int(target) < resolved_classes and 0 <= int(predicted) < resolved_classes:
            confusion_matrix[int(target)][int(predicted)] += 1

    total = sum(sum(row) for row in confusion_matrix)
    correct = sum(confusion_matrix[index][index] for index in range(resolved_classes))
    accuracy = correct / total if total else None

    class_metrics: list[dict[str, object]] = []
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    weighted_f1_sum = 0.0
    support_sum = 0

    for class_index in range(resolved_classes):
        true_positive = confusion_matrix[class_index][class_index]
        false_positive = sum(confusion_matrix[row][class_index] for row in range(resolved_classes) if row != class_index)
        false_negative = sum(confusion_matrix[class_index][col] for col in range(resolved_classes) if col != class_index)
        support = sum(confusion_matrix[class_index])

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        weighted_f1_sum += f1 * support
        support_sum += support

        class_metrics.append(
            {
                "classIndex": class_index,
                "className": (class_names or [])[class_index] if class_names and class_index < len(class_names) else str(class_index),
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_precision = precision_sum / resolved_classes
    macro_recall = recall_sum / resolved_classes
    macro_f1 = f1_sum / resolved_classes
    weighted_f1 = weighted_f1_sum / support_sum if support_sum else 0.0

    return {
        "loss": None,
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "macroPrecision": macro_precision,
        "macroRecall": macro_recall,
        "macroF1": macro_f1,
        "weightedF1": weighted_f1,
        "confusionMatrix": confusion_matrix,
        "classMetrics": class_metrics,
        "sampleCount": total,
    }


def _empty_metric_summary() -> dict[str, float | None]:
    return {"loss": None, "accuracy": None, "precision": None, "recall": None, "f1": None}


def _log_to_metric_summary(log: Mapping[str, Any], prefix: str = "") -> dict[str, float | None]:
    def read(key: str) -> float | None:
        value = log.get(f"{prefix}{key}" if prefix else key)
        return float(value) if isinstance(value, (int, float)) else None

    return {
        "loss": read("loss" if not prefix else "Loss"),
        "accuracy": read("accuracy" if not prefix else "Accuracy"),
        "precision": read("precision" if not prefix else "Precision"),
        "recall": read("recall" if not prefix else "Recall"),
        "f1": read("f1" if not prefix else "F1"),
    }


def _runtime_environment(torch) -> dict[str, object]:
    return {
        "pythonVersion": sys.version.split()[0],
        "platform": platform.platform(),
        "torchVersion": getattr(torch, "__version__", "unknown"),
        "cudaAvailable": bool(torch.cuda.is_available()),
    }


def _compute_batch_loss_and_outputs(torch, nn, model, criterion, inputs, labels, num_classes: int):
    outputs = model(inputs)
    if isinstance(criterion, nn.MSELoss):
        targets = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        loss = criterion(outputs, targets)
        normalized_outputs = _normalize_classification_outputs(outputs)
    else:
        normalized_outputs = _normalize_classification_outputs(outputs)
        loss = criterion(normalized_outputs, labels)
    return loss, normalized_outputs


def _summarize_epoch_metrics(
    running_loss: float,
    batch_count: int,
    predictions: list[int],
    labels: list[int],
    num_classes: int,
    class_names: list[str],
    include_metrics: bool,
) -> dict[str, object]:
    metrics = compute_classification_metrics(predictions, labels, num_classes, class_names)
    metrics["loss"] = running_loss / max(batch_count, 1)

    if not include_metrics:
        metrics["accuracy"] = None
        metrics["precision"] = None
        metrics["recall"] = None
        metrics["f1"] = None

    return metrics


def train_one_epoch(
    torch,
    nn,
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    num_classes: int,
    class_names: list[str],
    include_metrics: bool,
    should_cancel: Callable[[], bool],
) -> tuple[dict[str, object], bool]:
    """Run one training epoch and return aggregate metrics."""

    model.train()
    running_loss = 0.0
    batch_count = 0
    predictions: list[int] = []
    labels_seen: list[int] = []

    for inputs, labels in dataloader:
        if should_cancel():
            return {}, True

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, outputs = _compute_batch_loss_and_outputs(torch, nn, model, criterion, inputs, labels, num_classes)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        batch_count += 1
        predictions.extend(int(value) for value in outputs.argmax(dim=1).detach().cpu().tolist())
        labels_seen.extend(int(value) for value in labels.detach().cpu().tolist())

    return _summarize_epoch_metrics(
        running_loss,
        batch_count,
        predictions,
        labels_seen,
        num_classes,
        class_names,
        include_metrics,
    ), False


def evaluate(
    torch,
    nn,
    model,
    dataloader,
    criterion,
    device,
    num_classes: int,
    class_names: list[str],
    include_metrics: bool = True,
) -> dict[str, object]:
    """Evaluate a model on one split and return loss plus classification metrics."""

    model.eval()
    running_loss = 0.0
    batch_count = 0
    predictions: list[int] = []
    labels_seen: list[int] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss, outputs = _compute_batch_loss_and_outputs(torch, nn, model, criterion, inputs, labels, num_classes)
            running_loss += float(loss.item())
            batch_count += 1
            predictions.extend(int(value) for value in outputs.argmax(dim=1).detach().cpu().tolist())
            labels_seen.extend(int(value) for value in labels.detach().cpu().tolist())

    return _summarize_epoch_metrics(
        running_loss,
        batch_count,
        predictions,
        labels_seen,
        num_classes,
        class_names,
        include_metrics,
    )


def _persist_training_artifacts(
    model,
    logs: list[dict[str, float | int | None]],
    metadata: TrainingRunMetadataPayload,
    warnings: list[str],
    diagnostics: dict[str, object] | None = None,
    insights: dict[str, object] | None = None,
    evaluation: dict[str, object] | None = None,
    project_snapshot: dict[str, object] | None = None,
    normalized_config: dict[str, object] | None = None,
    runtime_environment: dict[str, object] | None = None,
) -> TrainingRunMetadataPayload:
    """Persist weights and JSON summaries for a completed run."""

    root = _get_training_runs_root()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}-{_slugify(metadata.project_name)}"
    run_directory = root / run_id
    run_directory.mkdir(parents=True, exist_ok=True)

    weights_path = run_directory / "model_weights.pt"
    logs_path = run_directory / "training_logs.json"
    summary_path = run_directory / "training_summary.json"
    metadata.run_id = run_id

    torch, _, _, _, _, _ = _load_torch_runtime()
    torch.save(model.state_dict(), weights_path)
    logs_path.write_text(json.dumps(logs, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "ok": True,
                "status": "completed",
                "warnings": warnings,
                "logs": logs,
                "diagnostics": diagnostics,
                "insights": insights,
                "evaluation": evaluation,
                "projectSnapshot": project_snapshot,
                "normalizedConfig": normalized_config,
                "runtimeEnvironment": runtime_environment,
                "trainingMetadata": {
                    **asdict(metadata),
                    "run_directory": str(run_directory.resolve()),
                    "weights_path": str(weights_path.resolve()),
                    "logs_path": str(logs_path.resolve()),
                    "summary_path": str(summary_path.resolve()),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metadata.run_directory = str(run_directory.resolve())
    metadata.weights_path = str(weights_path.resolve())
    metadata.logs_path = str(logs_path.resolve())
    metadata.summary_path = str(summary_path.resolve())
    return metadata


# ---------------------------------------------------------------------------
# Dataset-aware runtime for the current training phase.
# ---------------------------------------------------------------------------

from app.services.dataset_inspection import (
    BUILTIN_RUNTIME_SAMPLE_COUNT,
    BUILTIN_RUNTIME_SPLITS,
    build_dataset_input_shape,
    compute_ratio_split_counts,
    inspect_dataset_config,
    normalize_dataset_params,
    resolve_normalize_stats,
)


@dataclass(slots=True)
class RuntimeDatasetBundle:
    """Resolved runtime dataset plus dataset-summary metadata for reporting."""

    dataset: Any
    val_dataset: Any | None
    test_dataset: Any | None
    requested_name: str
    used_name: str
    dataset_mode: str
    image_size: int
    num_classes: int
    class_names: list[str]
    sample_count: int
    splits: dict[str, int]
    input_shape: list[int]
    task_type: str
    train_split: bool
    warnings: list[str]


@dataclass(slots=True)
class TrainingRunMetadataPayload:
    """Serializable summary of a completed training run."""

    run_id: str
    project_name: str
    requested_dataset_name: str
    dataset_used: str
    dataset_size: int
    dataset_mode: str
    sample_count: int
    image_size: int
    num_classes: int
    class_names: list[str]
    splits: dict[str, int]
    input_shape: list[int]
    task_type: str
    train_split: bool
    batch_size: int
    shuffle: bool
    num_workers: int
    drop_last: bool
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    collate_fn_type: str
    epochs: int
    device: str
    loss_type: str
    optimizer_type: str
    learning_rate: float
    weight_decay: float
    momentum: float | None
    metric_type: str | None
    started_at: str
    completed_at: str
    duration_seconds: float
    run_directory: str
    weights_path: str
    logs_path: str
    summary_path: str


def _convert_to_rgb(image):
    return image.convert("RGB")


def _convert_to_grayscale(image):
    return image.convert("L")


def normalize_dataloader_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize DataLoader params so old projects stay compatible."""

    raw = dict(params or {})
    prefetch_factor = raw.get("prefetchFactor", 2)
    try:
        prefetch_factor = int(prefetch_factor) if prefetch_factor not in {None, ""} else None
    except (TypeError, ValueError):
        prefetch_factor = 2

    return {
        "batchSize": int(raw.get("batchSize", 32)),
        "shuffle": bool(raw.get("shuffle", True)),
        "numWorkers": int(raw.get("numWorkers", 0)),
        "dropLast": bool(raw.get("dropLast", False)),
        "pinMemory": bool(raw.get("pinMemory", False)),
        "persistentWorkers": bool(raw.get("persistentWorkers", False)),
        "prefetchFactor": prefetch_factor,
        "collateFnType": str(raw.get("collateFnType", "default")),
    }


def _collect_dataloader_runtime_warnings(params: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    num_workers = int(params["numWorkers"])
    if bool(params["persistentWorkers"]) and num_workers == 0:
        warnings.append("DataLoader `persistentWorkers` was disabled at runtime because `numWorkers` is 0.")
    if params["prefetchFactor"] is not None and num_workers == 0:
        warnings.append("DataLoader `prefetchFactor` was ignored at runtime because `numWorkers` is 0.")
    if params["collateFnType"] == "custom_placeholder":
        warnings.append("DataLoader `collateFnType=custom_placeholder` is not implemented yet, so the default collate function was used.")
    return warnings


def _build_transform_pipeline(dataset_params: Mapping[str, Any], transforms, is_training: bool) -> Any:
    transform_steps: list[object] = []
    image_size = int(dataset_params["imageSize"])

    if dataset_params["colorMode"] == "rgb":
        transform_steps.append(transforms.Lambda(_convert_to_rgb))
    else:
        transform_steps.append(transforms.Lambda(_convert_to_grayscale))

    if bool(dataset_params["augmentationEnabled"]) and is_training:
        transform_steps.append(transforms.RandomHorizontalFlip())
        transform_steps.append(transforms.RandomRotation(10))

    transform_steps.append(transforms.Resize((image_size, image_size)))
    transform_steps.append(transforms.ToTensor())

    if bool(dataset_params["normalize"]):
        mean, std = resolve_normalize_stats(dataset_params)
        transform_steps.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_steps)


def _split_indices(total_count: int, split_counts: Mapping[str, int], seed: int) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(total_count))
    import random

    random.Random(seed).shuffle(indices)
    train_count = max(int(split_counts.get("train", total_count)), 0)
    val_count = max(int(split_counts.get("val", 0)), 0)
    test_count = max(int(split_counts.get("test", 0)), 0)
    train_indices = indices[:train_count]
    val_indices = indices[train_count : train_count + val_count]
    test_indices = indices[train_count + val_count : train_count + val_count + test_count]
    return train_indices, val_indices, test_indices


def _build_builtin_runtime_dataset(ir_graph: IRGraph, normalized: Mapping[str, Any], inspection, seed: int) -> RuntimeDatasetBundle:
    _, _, _, Subset, datasets, transforms = _load_torch_runtime()
    warnings = list(inspection.warnings)
    input_shape = build_dataset_input_shape(normalized) or get_input_shape(ir_graph)
    train_split = _parse_bool(normalized.get("trainSplit", True), True)
    dataset_name = str(normalized["datasetName"])
    train_transform = _build_transform_pipeline(normalized, transforms, is_training=True)
    eval_transform = _build_transform_pipeline(normalized, transforms, is_training=False)
    split_counts = dict(BUILTIN_RUNTIME_SPLITS)

    if dataset_name == "MNIST":
        try:
            train_source = datasets.MNIST(
                root="./data",
                train=train_split,
                download=True,
                transform=train_transform,
            )
            eval_source = datasets.MNIST(
                root="./data",
                train=train_split,
                download=False,
                transform=eval_transform,
            )
            capped_size = min(len(train_source), 256)
            train_indices, val_indices, test_indices = _split_indices(capped_size, split_counts, seed)
            return RuntimeDatasetBundle(
                dataset=Subset(train_source, train_indices),
                val_dataset=Subset(eval_source, val_indices) if val_indices else None,
                test_dataset=Subset(eval_source, test_indices) if test_indices else None,
                requested_name="MNIST",
                used_name="MNIST",
                dataset_mode="builtin",
                image_size=int(normalized["imageSize"]),
                num_classes=10,
                class_names=[str(index) for index in range(10)],
                sample_count=capped_size,
                splits=split_counts,
                input_shape=input_shape,
                task_type=str(normalized["taskType"]),
                train_split=train_split,
                warnings=warnings,
            )
        except Exception as exc:
            raise RuntimeError(
                "MNIST could not be loaded or downloaded, so training was stopped instead of silently using FakeData. "
                "Check network access for the first MNIST download, or switch the Dataset node to FakeData explicitly."
            ) from exc

    train_source = datasets.FakeData(
        size=BUILTIN_RUNTIME_SAMPLE_COUNT,
        image_size=(int(input_shape[0]), int(input_shape[1]), int(input_shape[2])),
        num_classes=max(int(normalized["numClasses"]), 1),
        transform=train_transform,
    )
    eval_source = datasets.FakeData(
        size=BUILTIN_RUNTIME_SAMPLE_COUNT,
        image_size=(int(input_shape[0]), int(input_shape[1]), int(input_shape[2])),
        num_classes=max(int(normalized["numClasses"]), 1),
        transform=eval_transform,
    )
    train_indices, val_indices, test_indices = _split_indices(BUILTIN_RUNTIME_SAMPLE_COUNT, split_counts, seed)
    class_names = [str(index) for index in range(max(int(normalized["numClasses"]), 1))]
    return RuntimeDatasetBundle(
        dataset=Subset(train_source, train_indices),
        val_dataset=Subset(eval_source, val_indices) if val_indices else None,
        test_dataset=Subset(eval_source, test_indices) if test_indices else None,
        requested_name=dataset_name,
        used_name="FakeData",
        dataset_mode="builtin",
        image_size=int(normalized["imageSize"]),
        num_classes=max(int(normalized["numClasses"]), 1),
        class_names=class_names,
        sample_count=BUILTIN_RUNTIME_SAMPLE_COUNT,
        splits=split_counts,
        input_shape=input_shape,
        task_type=str(normalized["taskType"]),
        train_split=train_split,
        warnings=warnings,
    )


def _build_image_folder_runtime_dataset(normalized: Mapping[str, Any], inspection, seed: int) -> RuntimeDatasetBundle:
    _, _, _, Subset, datasets, transforms = _load_torch_runtime()
    root_path = Path(str(normalized["rootPath"])).expanduser()
    train_transform = _build_transform_pipeline(normalized, transforms, is_training=True)
    eval_transform = _build_transform_pipeline(normalized, transforms, is_training=False)
    input_shape = build_dataset_input_shape(normalized) or [1, 28, 28]

    if inspection.resolved_split_mode == "predefined":
        train_root = root_path / "train"
        dataset = datasets.ImageFolder(root=str(train_root), transform=train_transform)
        val_root = root_path / "val"
        test_root = root_path / "test"
        val_dataset = datasets.ImageFolder(root=str(val_root), transform=eval_transform) if val_root.exists() else None
        test_dataset = datasets.ImageFolder(root=str(test_root), transform=eval_transform) if test_root.exists() else None
        return RuntimeDatasetBundle(
            dataset=dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            requested_name=str(root_path),
            used_name="ImageFolder",
            dataset_mode="image_folder",
            image_size=int(normalized["imageSize"]),
            num_classes=len(dataset.classes),
            class_names=list(dataset.classes),
            sample_count=int(inspection.sample_count),
            splits=dict(inspection.splits),
            input_shape=input_shape,
            task_type=str(normalized["taskType"]),
            train_split=True,
            warnings=list(inspection.warnings),
        )

    full_dataset = datasets.ImageFolder(root=str(root_path), transform=train_transform)
    eval_dataset = datasets.ImageFolder(root=str(root_path), transform=eval_transform)

    split_counts = compute_ratio_split_counts(
        total_count=len(full_dataset),
        train_ratio=float(normalized["trainRatio"]),
        val_ratio=float(normalized["valRatio"]),
        test_ratio=float(normalized["testRatio"]),
    )
    if bool(normalized["shuffleBeforeSplit"]):
        train_indices, val_indices, test_indices = _split_indices(len(full_dataset), split_counts, seed)
    else:
        ordered_indices = list(range(len(full_dataset)))
        train_count = split_counts["train"]
        val_count = split_counts["val"]
        test_count = split_counts["test"]
        train_indices = ordered_indices[:train_count]
        val_indices = ordered_indices[train_count : train_count + val_count]
        test_indices = ordered_indices[train_count + val_count : train_count + val_count + test_count]
    dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(eval_dataset, val_indices) if val_indices else None
    test_dataset = Subset(eval_dataset, test_indices) if test_indices else None

    warnings = list(inspection.warnings)
    if split_counts["val"] == 0 and float(normalized["valRatio"]) > 0:
        warnings.append("Validation split was configured but ended up empty at runtime because the dataset is very small.")
    if split_counts["test"] == 0 and float(normalized["testRatio"]) > 0:
        warnings.append("Test split was configured but ended up empty at runtime because the dataset is very small.")

    return RuntimeDatasetBundle(
        dataset=dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        requested_name=str(root_path),
        used_name="ImageFolder",
        dataset_mode="image_folder",
        image_size=int(normalized["imageSize"]),
        num_classes=len(full_dataset.classes),
        class_names=list(full_dataset.classes),
        sample_count=len(full_dataset),
        splits=split_counts,
        input_shape=input_shape,
        task_type=str(normalized["taskType"]),
        train_split=True,
        warnings=warnings,
    )


def build_runtime_dataset(ir_graph: IRGraph, seed: int = 42):
    """Build the configured runtime dataset bundle."""

    components = resolve_training_components(ir_graph)
    dataset_params = normalize_dataset_params(components.dataset_node.params if components.dataset_node else {})
    inspection = inspect_dataset_config(dataset_params)

    if not inspection.success:
        raise RuntimeError("; ".join(inspection.errors))
    if inspection.dataset_mode == "csv":
        raise RuntimeError("CSV runtime training is not implemented yet in this phase.")
    if inspection.dataset_mode == "builtin":
        return _build_builtin_runtime_dataset(ir_graph, dataset_params, inspection, seed)
    if inspection.dataset_mode == "image_folder":
        return _build_image_folder_runtime_dataset(dataset_params, inspection, seed)

    raise RuntimeError(f"Unsupported dataset mode `{inspection.dataset_mode}`.")


def build_runtime_dataloader(ir_graph: IRGraph, dataset, for_evaluation: bool = False, seed: int = 42):
    """Create a DataLoader from the graph params."""

    torch, _, DataLoader, _, _, _ = _load_torch_runtime()
    components = resolve_training_components(ir_graph)
    params = normalize_dataloader_params(components.dataloader_node.params if components.dataloader_node else {})
    num_workers = int(params["numWorkers"])

    dataloader_kwargs: dict[str, Any] = {
        "batch_size": int(params["batchSize"]),
        "shuffle": False if for_evaluation else bool(params["shuffle"]),
        "num_workers": num_workers,
        "drop_last": False if for_evaluation else bool(params["dropLast"]),
        "pin_memory": bool(params["pinMemory"]),
    }
    if not for_evaluation and bool(params["shuffle"]):
        dataloader_kwargs["generator"] = torch.Generator().manual_seed(int(seed))
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(params["persistentWorkers"])
        if params["prefetchFactor"] is not None:
            dataloader_kwargs["prefetch_factor"] = int(params["prefetchFactor"])

    return DataLoader(dataset, **dataloader_kwargs)


def _select_best_epoch(logs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not logs:
        return None
    if any(isinstance(log.get("valAccuracy"), (int, float)) for log in logs):
        return max(
            logs,
            key=lambda log: float(log.get("valAccuracy") if isinstance(log.get("valAccuracy"), (int, float)) else -1.0),
        )
    if any(isinstance(log.get("valLoss"), (int, float)) for log in logs):
        return min(
            logs,
            key=lambda log: float(log.get("valLoss") if isinstance(log.get("valLoss"), (int, float)) else float("inf")),
        )
    return min(logs, key=lambda log: float(log.get("loss") if isinstance(log.get("loss"), (int, float)) else float("inf")))


def _build_evaluation_summary(
    logs: list[dict[str, Any]],
    primary_metrics: dict[str, object],
    test_metrics: dict[str, object] | None,
    class_names: list[str],
    seed: int,
    config_hash: str,
    primary_split: str,
) -> dict[str, object]:
    best_log = _select_best_epoch(logs)
    last_log = logs[-1] if logs else {}
    final_train = _log_to_metric_summary(last_log)
    final_validation = _log_to_metric_summary(last_log, "val") if any("valLoss" in log for log in logs) else None
    best_validation = _log_to_metric_summary(best_log, "val") if best_log and "valLoss" in best_log else None

    return {
        "finalTrain": final_train,
        "finalValidation": final_validation,
        "finalTest": {
            "loss": test_metrics.get("loss"),
            "accuracy": test_metrics.get("accuracy"),
            "precision": test_metrics.get("precision"),
            "recall": test_metrics.get("recall"),
            "f1": test_metrics.get("f1"),
        }
        if test_metrics
        else None,
        "bestValidation": best_validation,
        "bestEpoch": int(best_log["epoch"]) if best_log and "epoch" in best_log else None,
        "primarySplit": primary_split,
        "confusionMatrix": primary_metrics.get("confusionMatrix", []),
        "classMetrics": primary_metrics.get("classMetrics", []),
        "macroPrecision": primary_metrics.get("macroPrecision"),
        "macroRecall": primary_metrics.get("macroRecall"),
        "macroF1": primary_metrics.get("macroF1"),
        "weightedF1": primary_metrics.get("weightedF1"),
        "sampleCount": primary_metrics.get("sampleCount", 0),
        "classNames": class_names,
        "seed": seed,
        "configHash": config_hash,
    }


def run_training(
    ir_graph: IRGraph,
    project_name: str = "Untitled Project",
    diagnostics_payload: Any | None = None,
    project_snapshot: dict[str, object] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[
    str,
    list[dict[str, float | int | None]],
    list[str],
    dict[str, object] | None,
    dict[str, object] | None,
    dict[str, object] | None,
]:
    """Execute the synchronous training loop for builtin and image-folder datasets."""

    torch, nn, _, _, _, _ = _load_torch_runtime()
    components = resolve_training_components(ir_graph)
    dataset_params = normalize_dataset_params(components.dataset_node.params if components.dataset_node else {})
    dataloader_params = normalize_dataloader_params(components.dataloader_node.params if components.dataloader_node else {})
    loss_params = components.loss_node.params if components.loss_node else {}
    optimizer_params = components.optimizer_node.params if components.optimizer_node else {}
    trainer_params = components.trainer_node.params if components.trainer_node else {}
    metric_params = components.metric_node.params if components.metric_node else {}
    epochs = int(trainer_params.get("epochs", 1))
    seed = int(trainer_params.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _resolve_device(trainer_params.get("device", "cpu"))
    started_at = datetime.now(timezone.utc)
    started_at_iso = started_at.isoformat()
    started_perf = perf_counter()
    normalized_config = {
        "dataset": dict(dataset_params),
        "dataloader": dict(dataloader_params),
        "loss": dict(loss_params),
        "optimizer": dict(optimizer_params),
        "trainer": {**dict(trainer_params), "seed": seed},
        "metric": dict(metric_params),
    }
    config_hash = _stable_hash({"project": project_snapshot, "config": normalized_config})

    dataset_bundle = build_runtime_dataset(ir_graph, seed=seed)
    warnings = list(dataset_bundle.warnings)
    warnings.extend(_collect_dataloader_runtime_warnings(dataloader_params))
    dataloader = build_runtime_dataloader(ir_graph, dataset_bundle.dataset, seed=seed)
    val_dataloader = (
        build_runtime_dataloader(ir_graph, dataset_bundle.val_dataset, for_evaluation=True, seed=seed)
        if dataset_bundle.val_dataset is not None
        else None
    )
    test_dataloader = (
        build_runtime_dataloader(ir_graph, dataset_bundle.test_dataset, for_evaluation=True, seed=seed)
        if dataset_bundle.test_dataset is not None
        else None
    )
    model = build_runtime_model(ir_graph).to(device)
    criterion = build_runtime_loss(ir_graph)
    optimizer = build_runtime_optimizer(ir_graph, model)
    num_classes = dataset_bundle.num_classes or get_num_classes(ir_graph)
    class_names = list(dataset_bundle.class_names)
    include_metrics = components.metric_node is not None
    logs: list[dict[str, float | int | None]] = []

    def is_cancelled() -> bool:
        return bool(should_cancel and should_cancel())

    def build_cancelled_response() -> tuple[
        str,
        list[dict[str, float | int | None]],
        list[str],
        None,
        dict[str, object],
        None,
    ]:
        cancel_messages = [*warnings, "Training was cancelled before completion."]
        from app.services.diagnostics import build_training_insights

        insights_payload = build_training_insights(
            diagnostics=diagnostics_payload,
            logs=logs,
            status="cancelled",
            runtime_messages=cancel_messages,
            training_metadata=None,
        )
        return "cancelled", logs, cancel_messages, None, insights_payload.model_dump(by_alias=True), None

    if is_cancelled():
        return build_cancelled_response()

    for epoch in range(1, epochs + 1):
        if is_cancelled():
            return build_cancelled_response()

        train_metrics, cancelled = train_one_epoch(
            torch,
            nn,
            model,
            dataloader,
            criterion,
            optimizer,
            device,
            num_classes,
            class_names,
            include_metrics,
            is_cancelled,
        )
        if cancelled:
            return build_cancelled_response()

        val_metrics = (
            evaluate(torch, nn, model, val_dataloader, criterion, device, num_classes, class_names, include_metrics)
            if val_dataloader is not None
            else None
        )
        epoch_log = {
            "epoch": epoch,
            "loss": train_metrics.get("loss"),
            "accuracy": train_metrics.get("accuracy"),
            "precision": train_metrics.get("precision"),
            "recall": train_metrics.get("recall"),
            "f1": train_metrics.get("f1"),
        }
        if val_metrics is not None:
            epoch_log.update(
                {
                    "valLoss": val_metrics.get("loss"),
                    "valAccuracy": val_metrics.get("accuracy"),
                    "valPrecision": val_metrics.get("precision"),
                    "valRecall": val_metrics.get("recall"),
                    "valF1": val_metrics.get("f1"),
                }
            )
        logs.append(epoch_log)
        if progress_callback:
            progress_callback(
                {
                    "status": "running",
                    "epoch": epoch,
                    "epochs": epochs,
                    "progress": epoch / max(epochs, 1),
                    "log": epoch_log,
                }
            )

    completed_at_iso = datetime.now(timezone.utc).isoformat()
    primary_split = "validation" if val_dataloader is not None else "train"
    primary_metrics = (
        evaluate(torch, nn, model, val_dataloader, criterion, device, num_classes, class_names, include_metrics)
        if val_dataloader is not None
        else evaluate(torch, nn, model, dataloader, criterion, device, num_classes, class_names, include_metrics)
    )
    test_metrics = (
        evaluate(torch, nn, model, test_dataloader, criterion, device, num_classes, class_names, include_metrics)
        if test_dataloader is not None
        else None
    )
    evaluation_payload = _build_evaluation_summary(
        logs,
        primary_metrics,
        test_metrics,
        class_names,
        seed,
        config_hash,
        primary_split,
    )
    metadata = TrainingRunMetadataPayload(
        run_id="",
        project_name=project_name,
        requested_dataset_name=dataset_bundle.requested_name,
        dataset_used=dataset_bundle.used_name,
        dataset_size=len(dataset_bundle.dataset),
        dataset_mode=dataset_bundle.dataset_mode,
        sample_count=dataset_bundle.sample_count,
        image_size=dataset_bundle.image_size,
        num_classes=dataset_bundle.num_classes,
        class_names=list(dataset_bundle.class_names),
        splits=dict(dataset_bundle.splits),
        input_shape=list(dataset_bundle.input_shape),
        task_type=dataset_bundle.task_type,
        train_split=dataset_bundle.train_split,
        batch_size=int(dataloader_params["batchSize"]),
        shuffle=bool(dataloader_params["shuffle"]),
        num_workers=int(dataloader_params["numWorkers"]),
        drop_last=bool(dataloader_params["dropLast"]),
        pin_memory=bool(dataloader_params["pinMemory"]),
        persistent_workers=bool(dataloader_params["persistentWorkers"]) and int(dataloader_params["numWorkers"]) > 0,
        prefetch_factor=int(dataloader_params["prefetchFactor"]) if dataloader_params["prefetchFactor"] is not None and int(dataloader_params["numWorkers"]) > 0 else None,
        collate_fn_type=str(dataloader_params["collateFnType"]),
        epochs=epochs,
        device=str(device),
        loss_type=str(loss_params.get("lossType", "CrossEntropyLoss")),
        optimizer_type=str(optimizer_params.get("optimizerType", "Adam")),
        learning_rate=float(optimizer_params.get("lr", 0.001)),
        weight_decay=float(optimizer_params.get("weightDecay", 0.0)),
        momentum=float(optimizer_params.get("momentum", 0.0)) if str(optimizer_params.get("optimizerType", "Adam")) == "SGD" else None,
        metric_type=str(metric_params.get("metricType")) if components.metric_node else None,
        started_at=started_at_iso,
        completed_at=completed_at_iso,
        duration_seconds=round(perf_counter() - started_perf, 4),
        run_directory="",
        weights_path="",
        logs_path="",
        summary_path="",
    )

    from app.services.diagnostics import build_training_insights

    insights_payload = build_training_insights(
        diagnostics=diagnostics_payload,
        logs=logs,
        status="completed",
        runtime_messages=warnings,
        training_metadata=asdict(metadata),
    )

    try:
        metadata = _persist_training_artifacts(
            model,
            logs,
            metadata,
            warnings,
            diagnostics=diagnostics_payload.model_dump(by_alias=True) if diagnostics_payload is not None else None,
            insights=insights_payload.model_dump(by_alias=True),
            evaluation=evaluation_payload,
            project_snapshot=project_snapshot,
            normalized_config=normalized_config,
            runtime_environment=_runtime_environment(torch),
        )
    except Exception as exc:
        warnings.append(f"Training artifacts could not be fully persisted: {exc}")

    return "completed", logs, warnings, asdict(metadata), insights_payload.model_dump(by_alias=True), evaluation_payload
