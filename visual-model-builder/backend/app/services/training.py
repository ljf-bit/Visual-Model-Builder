"""Training graph helpers and runtime execution for Phase 2."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

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


@dataclass(slots=True)
class RuntimeDatasetBundle:
    """Resolved runtime dataset and the metadata needed for reporting."""

    dataset: Any
    requested_name: str
    used_name: str
    image_size: int
    num_classes: int
    train_split: bool
    warnings: list[str]


@dataclass(slots=True)
class TrainingRunMetadataPayload:
    """Serializable summary of a completed training run."""

    project_name: str
    requested_dataset_name: str
    dataset_used: str
    dataset_size: int
    image_size: int
    num_classes: int
    train_split: bool
    batch_size: int
    shuffle: bool
    num_workers: int
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


def _resize_transform(image_size: int, transforms):
    transform_steps: list[object] = []
    if image_size != 28:
        transform_steps.append(transforms.Resize((image_size, image_size)))
    transform_steps.append(transforms.ToTensor())
    return transforms.Compose(transform_steps)


def build_runtime_dataset(ir_graph: IRGraph):
    """Build a small teaching dataset and return warnings if any."""

    _, _, _, Subset, datasets, transforms = _load_torch_runtime()
    components = resolve_training_components(ir_graph)
    params = components.dataset_node.params if components.dataset_node else {}
    dataset_name = str(params.get("datasetName", "FakeData"))
    image_size = int(params.get("imageSize", get_input_shape(ir_graph)[-1]))
    num_classes = int(params.get("numClasses", get_num_classes(ir_graph)))
    input_shape = get_input_shape(ir_graph)
    train_split = _parse_bool(params.get("trainSplit", True), True)
    warnings: list[str] = []

    if dataset_name == "MNIST":
        transform = _resize_transform(image_size, transforms)
        try:
            dataset = datasets.MNIST(
                root="./data",
                train=train_split,
                download=False,
                transform=transform,
            )
            return RuntimeDatasetBundle(
                dataset=Subset(dataset, range(min(len(dataset), 256))),
                requested_name=dataset_name,
                used_name="MNIST",
                image_size=image_size,
                num_classes=num_classes,
                train_split=train_split,
                warnings=warnings,
            )
        except Exception:
            warnings.append("MNIST was unavailable locally, so training used FakeData instead.")

    return RuntimeDatasetBundle(
        dataset=datasets.FakeData(
            size=256,
            image_size=(int(input_shape[0]), image_size, image_size),
            num_classes=num_classes,
            transform=transforms.ToTensor(),
        ),
        requested_name=dataset_name,
        used_name="FakeData",
        image_size=image_size,
        num_classes=num_classes,
        train_split=train_split,
        warnings=warnings,
    )


def build_runtime_dataloader(ir_graph: IRGraph, dataset) -> DataLoader:
    """Create a DataLoader from graph params."""

    _, _, DataLoader, _, _, _ = _load_torch_runtime()
    components = resolve_training_components(ir_graph)
    params = components.dataloader_node.params if components.dataloader_node else {}
    return DataLoader(
        dataset,
        batch_size=int(params.get("batchSize", 32)),
        shuffle=bool(params.get("shuffle", True)),
        num_workers=int(params.get("numWorkers", 0)),
    )


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


def _persist_training_artifacts(
    model,
    logs: list[dict[str, float | int | None]],
    metadata: TrainingRunMetadataPayload,
    warnings: list[str],
    diagnostics: dict[str, object] | None = None,
    insights: dict[str, object] | None = None,
) -> TrainingRunMetadataPayload:
    """Persist weights and JSON summaries for a completed run."""

    root = _get_training_runs_root()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}-{_slugify(metadata.project_name)}"
    run_directory = root / run_id
    run_directory.mkdir(parents=True, exist_ok=True)

    weights_path = run_directory / "model_weights.pt"
    logs_path = run_directory / "training_logs.json"
    summary_path = run_directory / "training_summary.json"

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


def run_training(
    ir_graph: IRGraph,
    project_name: str = "Untitled Project",
    diagnostics_payload: Any | None = None,
) -> tuple[str, list[dict[str, float | int | None]], list[str], dict[str, object] | None, dict[str, object] | None]:
    """Execute the minimal Phase 2 training loop."""

    torch, nn, _, _, _, _ = _load_torch_runtime()
    components = resolve_training_components(ir_graph)
    dataset_params = components.dataset_node.params if components.dataset_node else {}
    dataloader_params = components.dataloader_node.params if components.dataloader_node else {}
    loss_params = components.loss_node.params if components.loss_node else {}
    optimizer_params = components.optimizer_node.params if components.optimizer_node else {}
    trainer_params = components.trainer_node.params if components.trainer_node else {}
    metric_params = components.metric_node.params if components.metric_node else {}
    epochs = int(trainer_params.get("epochs", 1))
    device = _resolve_device(trainer_params.get("device", "cpu"))
    started_at = datetime.now(timezone.utc)
    started_at_iso = started_at.isoformat()
    started_perf = perf_counter()

    dataset_bundle = build_runtime_dataset(ir_graph)
    warnings = list(dataset_bundle.warnings)
    dataloader = build_runtime_dataloader(ir_graph, dataset_bundle.dataset)
    model = build_runtime_model(ir_graph).to(device)
    criterion = build_runtime_loss(ir_graph)
    optimizer = build_runtime_optimizer(ir_graph, model)
    num_classes = get_num_classes(ir_graph)
    logs: list[dict[str, float | int | None]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        batch_count = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if isinstance(criterion, nn.MSELoss):
                targets = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
                loss = criterion(outputs, targets)
            else:
                outputs = _normalize_classification_outputs(outputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            accuracy = _maybe_accuracy(components.metric_node, outputs, labels)
            if accuracy is not None:
                running_accuracy += accuracy
            batch_count += 1

        logs.append(
            {
                "epoch": epoch,
                "loss": running_loss / max(batch_count, 1),
                "accuracy": (running_accuracy / batch_count) if components.metric_node and batch_count else None,
            }
        )

    completed_at_iso = datetime.now(timezone.utc).isoformat()
    metadata = TrainingRunMetadataPayload(
        project_name=project_name,
        requested_dataset_name=str(dataset_params.get("datasetName", dataset_bundle.requested_name)),
        dataset_used=dataset_bundle.used_name,
        dataset_size=len(dataset_bundle.dataset),
        image_size=dataset_bundle.image_size,
        num_classes=dataset_bundle.num_classes,
        train_split=dataset_bundle.train_split,
        batch_size=int(dataloader_params.get("batchSize", 32)),
        shuffle=bool(dataloader_params.get("shuffle", True)),
        num_workers=int(dataloader_params.get("numWorkers", 0)),
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
        )
    except Exception as exc:
        warnings.append(f"Training artifacts could not be fully persisted: {exc}")

    return "completed", logs, warnings, asdict(metadata), insights_payload.model_dump(by_alias=True)
