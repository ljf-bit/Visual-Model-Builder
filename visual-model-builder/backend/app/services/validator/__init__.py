'''
"""
Graph Validator service.

Validates graph structure: checks for Input/Output nodes,
cycles, disconnected nodes, and connection legality.
"""

from app.services.graph_ir import IRGraph, topological_sort


def validate_graph(ir_graph: IRGraph) -> tuple[list[str], dict[str, list[str]]]:
    """
    Validate the graph and return (global_errors, node_errors).

    TODO: Implement full validation rules in Phase 1.
    """
    global_errors: list[str] = []
    node_errors: dict[str, list[str]] = {}

    # Check for Input node
    has_input = any(n.op == "Input" for n in ir_graph.nodes)
    if not has_input:
        global_errors.append("图中缺少 Input 节点")

    # Check for Output node
    has_output = any(n.op == "Output" for n in ir_graph.nodes)
    if not has_output:
        global_errors.append("图中缺少 Output 节点")

    # Check for cycles
    sorted_ids = topological_sort(ir_graph)
    if sorted_ids is None:
        global_errors.append("图中存在环路")

    # Check for disconnected nodes (no edges)
    for node in ir_graph.nodes:
        if node.op not in ("Input", "Output"):
            if not node.input_refs and not node.output_refs:
                if node.node_id not in node_errors:
                    node_errors[node.node_id] = []
                node_errors[node.node_id].append(f"节点 {node.node_id} 未连接")

    return global_errors, node_errors
'''

"""Validation rules for model graphs and Phase 2 training graphs."""

from app.services.graph_ir import IRGraph, topological_sort
from app.services.training import get_node_map, get_nodes_by_op, has_training_nodes, resolve_training_components


def _add_node_error(node_errors: dict[str, list[str]], node_id: str, message: str) -> None:
    node_errors.setdefault(node_id, []).append(message)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and value > 0


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and value >= 0


def _is_positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and float(value) > 0


def _validate_model_params(node, node_errors: dict[str, list[str]]) -> None:
    params = node.params

    if node.op == "Input":
        input_shape = params.get("inputShape")
        if not isinstance(input_shape, list) or len(input_shape) != 3 or not all(_is_positive_int(v) for v in input_shape):
            _add_node_error(node_errors, node.node_id, "Input requires inputShape as [C, H, W] with positive integers.")
    elif node.op == "Conv2d":
        for key in ("in_channels", "out_channels", "kernel_size", "stride"):
            if not _is_positive_int(params.get(key)):
                _add_node_error(node_errors, node.node_id, f"Conv2d parameter `{key}` must be a positive integer.")
        if not _is_non_negative_int(params.get("padding")):
            _add_node_error(node_errors, node.node_id, "Conv2d parameter `padding` must be a non-negative integer.")
    elif node.op == "MaxPool2d":
        for key in ("kernel_size", "stride"):
            if not _is_positive_int(params.get(key)):
                _add_node_error(node_errors, node.node_id, f"MaxPool2d parameter `{key}` must be a positive integer.")
        if not _is_non_negative_int(params.get("padding")):
            _add_node_error(node_errors, node.node_id, "MaxPool2d parameter `padding` must be a non-negative integer.")
    elif node.op == "Flatten":
        for key in ("start_dim", "end_dim"):
            if not isinstance(params.get(key), int):
                _add_node_error(node_errors, node.node_id, f"Flatten parameter `{key}` must be an integer.")
    elif node.op == "Linear":
        for key in ("in_features", "out_features"):
            if not _is_positive_int(params.get(key)):
                _add_node_error(node_errors, node.node_id, f"Linear parameter `{key}` must be a positive integer.")


def _validate_training_params(node, node_errors: dict[str, list[str]]) -> None:
    params = node.params

    if node.op == "Dataset":
        if params.get("datasetName") not in {"MNIST", "FakeData"}:
            _add_node_error(node_errors, node.node_id, "Dataset must use `MNIST` or `FakeData` in Phase 2.")
        if not _is_positive_int(params.get("imageSize")):
            _add_node_error(node_errors, node.node_id, "Dataset `imageSize` must be a positive integer.")
        if not _is_positive_int(params.get("numClasses")):
            _add_node_error(node_errors, node.node_id, "Dataset `numClasses` must be a positive integer.")
    elif node.op == "DataLoader":
        if not _is_positive_int(params.get("batchSize")):
            _add_node_error(node_errors, node.node_id, "DataLoader `batchSize` must be a positive integer.")
        if not _is_non_negative_int(params.get("numWorkers")):
            _add_node_error(node_errors, node.node_id, "DataLoader `numWorkers` must be a non-negative integer.")
    elif node.op == "Loss":
        if params.get("lossType") not in {"CrossEntropyLoss", "MSELoss"}:
            _add_node_error(node_errors, node.node_id, "Loss `lossType` must be `CrossEntropyLoss` or `MSELoss`.")
    elif node.op == "Optimizer":
        if params.get("optimizerType") not in {"SGD", "Adam"}:
            _add_node_error(node_errors, node.node_id, "Optimizer `optimizerType` must be `SGD` or `Adam`.")
        if not _is_positive_number(params.get("lr")):
            _add_node_error(node_errors, node.node_id, "Optimizer `lr` must be greater than 0.")
        if not isinstance(params.get("weightDecay"), (int, float)) or float(params.get("weightDecay")) < 0:
            _add_node_error(node_errors, node.node_id, "Optimizer `weightDecay` must be zero or greater.")
        if params.get("optimizerType") == "SGD" and (
            not isinstance(params.get("momentum"), (int, float)) or float(params.get("momentum")) < 0
        ):
            _add_node_error(node_errors, node.node_id, "Optimizer `momentum` must be zero or greater for SGD.")
    elif node.op == "Trainer":
        if not _is_positive_int(params.get("epochs")):
            _add_node_error(node_errors, node.node_id, "Trainer `epochs` must be a positive integer.")
        if not _is_positive_int(params.get("logInterval")):
            _add_node_error(node_errors, node.node_id, "Trainer `logInterval` must be a positive integer.")
        if params.get("device") not in {"cpu", "auto"}:
            _add_node_error(node_errors, node.node_id, "Trainer `device` must be `cpu` or `auto`.")
    elif node.op == "Metric":
        if params.get("metricType") != "Accuracy":
            _add_node_error(node_errors, node.node_id, "Metric `metricType` must be `Accuracy` in Phase 2.")


def _validate_common_graph(ir_graph: IRGraph, node_errors: dict[str, list[str]], global_errors: list[str]) -> None:
    node_map = get_node_map(ir_graph)
    inputs = get_nodes_by_op(ir_graph, "Input")
    outputs = get_nodes_by_op(ir_graph, "Output")

    if not inputs:
        global_errors.append("Graph requires at least one Input node.")
    if not outputs:
        global_errors.append("Graph requires at least one Output node.")

    sorted_ids = topological_sort(ir_graph)
    if sorted_ids is None:
        global_errors.append("Graph contains a cycle.")

    for node in ir_graph.nodes:
        if node.category == "unknown":
            _add_node_error(node_errors, node.node_id, f"Unsupported node type `{node.op}`.")
            continue

        if not node.input_refs and not node.output_refs:
            _add_node_error(node_errors, node.node_id, f"Node `{node.node_id}` is disconnected.")

        if node.op == "Input":
            invalid_inputs = [ref for ref in node.input_refs if node_map.get(ref) and node_map[ref].op != "DataLoader"]
            if invalid_inputs:
                _add_node_error(node_errors, node.node_id, "Input may only receive connections from DataLoader.")
        elif node.op == "Output":
            invalid_outputs = [ref for ref in node.output_refs if node_map.get(ref) and node_map[ref].op != "Loss"]
            if invalid_outputs:
                _add_node_error(node_errors, node.node_id, "Output may only connect to Loss in Phase 2.")
        elif node.category == "model":
            if len([ref for ref in node.input_refs if node_map.get(ref) and node_map[ref].category == "model"]) > 1:
                _add_node_error(node_errors, node.node_id, "Phase 2 still supports only a single upstream model connection.")
            if len([ref for ref in node.output_refs if node_map.get(ref) and node_map[ref].category == "model"]) > 1:
                _add_node_error(node_errors, node.node_id, "Phase 2 still supports only a single downstream model connection.")

        if node.category == "model":
            _validate_model_params(node, node_errors)
        else:
            _validate_training_params(node, node_errors)


def _validate_training_graph(ir_graph: IRGraph, node_errors: dict[str, list[str]], global_errors: list[str]) -> None:
    node_map = get_node_map(ir_graph)
    dataset_nodes = get_nodes_by_op(ir_graph, "Dataset")
    dataloader_nodes = get_nodes_by_op(ir_graph, "DataLoader")
    loss_nodes = get_nodes_by_op(ir_graph, "Loss")
    optimizer_nodes = get_nodes_by_op(ir_graph, "Optimizer")
    trainer_nodes = get_nodes_by_op(ir_graph, "Trainer")
    metric_nodes = get_nodes_by_op(ir_graph, "Metric")
    components = resolve_training_components(ir_graph)

    if len(dataset_nodes) != 1:
        global_errors.append("Phase 2 supports exactly one Dataset node.")
    if len(dataloader_nodes) != 1:
        global_errors.append("Phase 2 supports exactly one DataLoader node.")
    if len(loss_nodes) != 1:
        global_errors.append("Phase 2 supports exactly one Loss node.")
    if len(optimizer_nodes) != 1:
        global_errors.append("Phase 2 supports exactly one Optimizer node.")
    if len(trainer_nodes) != 1:
        global_errors.append("Phase 2 requires exactly one Trainer node.")
    if len(metric_nodes) > 1:
        global_errors.append("Phase 2 supports at most one Metric node.")

    if components.dataset_node and components.dataloader_node:
        if components.dataloader_node.node_id not in components.dataset_node.output_refs:
            _add_node_error(node_errors, components.dataset_node.node_id, "Dataset must connect to DataLoader.")
        if any(node_map.get(ref) and node_map[ref].op != "Dataset" for ref in components.dataloader_node.input_refs):
            _add_node_error(node_errors, components.dataloader_node.node_id, "DataLoader may only receive input from Dataset.")

    if components.dataloader_node and components.input_node:
        if components.input_node.node_id not in components.dataloader_node.output_refs:
            _add_node_error(node_errors, components.dataloader_node.node_id, "DataLoader must connect to the model Input node.")

    if components.output_node and components.loss_node:
        if components.loss_node.node_id not in components.output_node.output_refs:
            _add_node_error(node_errors, components.output_node.node_id, "Model Output must connect to Loss.")
        if any(node_map.get(ref) and node_map[ref].op != "Output" for ref in components.loss_node.input_refs):
            _add_node_error(node_errors, components.loss_node.node_id, "Loss may only receive input from the model Output node.")

    if components.trainer_node:
        trainer_inputs = {node_map[ref].op for ref in components.trainer_node.input_refs if ref in node_map}
        if "Loss" not in trainer_inputs:
            _add_node_error(node_errors, components.trainer_node.node_id, "Trainer must receive Loss.")
        if "Optimizer" not in trainer_inputs:
            _add_node_error(node_errors, components.trainer_node.node_id, "Trainer must receive Optimizer.")
        if metric_nodes and "Metric" not in trainer_inputs:
            _add_node_error(node_errors, components.trainer_node.node_id, "Trainer must receive Metric when Metric exists.")

    if components.optimizer_node and components.trainer_node and components.optimizer_node.output_refs != [components.trainer_node.node_id]:
        _add_node_error(node_errors, components.optimizer_node.node_id, "Optimizer must connect to Trainer.")
    if components.metric_node and components.trainer_node and components.metric_node.output_refs != [components.trainer_node.node_id]:
        _add_node_error(node_errors, components.metric_node.node_id, "Metric must connect to Trainer.")
    if components.dataset_node and components.dataset_node.input_refs:
        _add_node_error(node_errors, components.dataset_node.node_id, "Dataset must not have upstream inputs.")
    if components.optimizer_node and components.optimizer_node.input_refs:
        _add_node_error(node_errors, components.optimizer_node.node_id, "Optimizer does not accept upstream inputs in Phase 2.")
    if components.metric_node and components.metric_node.input_refs:
        _add_node_error(node_errors, components.metric_node.node_id, "Metric does not accept upstream inputs in Phase 2.")
    if components.trainer_node and components.trainer_node.output_refs:
        _add_node_error(node_errors, components.trainer_node.node_id, "Trainer is the terminal node of the training pipeline.")


def validate_graph(ir_graph: IRGraph, require_training: bool = False) -> tuple[list[str], dict[str, list[str]]]:
    """Validate graph structure and training constraints."""

    global_errors: list[str] = []
    node_errors: dict[str, list[str]] = {}

    _validate_common_graph(ir_graph, node_errors, global_errors)

    if require_training:
        _validate_training_graph(ir_graph, node_errors, global_errors)

    return global_errors, node_errors


# ---------------------------------------------------------------------------
# Dataset-aware validation overrides for the real-data training phase.
# The latest definitions below intentionally replace the lightweight Phase 2
# versions above while preserving the module path for existing imports.
# ---------------------------------------------------------------------------

from app.services.dataset_inspection import inspect_dataset_config, normalize_dataset_params
from app.services.training import get_input_shape, get_num_classes


def _append_unique(items: list[str], message: str) -> None:
    if message and message not in items:
        items.append(message)


def _add_node_error(node_errors: dict[str, list[str]], node_id: str, message: str) -> None:
    bucket = node_errors.setdefault(node_id, [])
    if message not in bucket:
        bucket.append(message)


def _validate_dataloader_params(node, node_errors: dict[str, list[str]], warnings: list[str]) -> None:
    params = node.params
    batch_size = params.get("batchSize")
    num_workers = params.get("numWorkers")
    prefetch_factor = params.get("prefetchFactor")
    collate_fn_type = params.get("collateFnType", "default")
    persistent_workers = bool(params.get("persistentWorkers", False))

    if not _is_positive_int(batch_size):
        _add_node_error(node_errors, node.node_id, "DataLoader `batchSize` must be a positive integer.")
    if not _is_non_negative_int(num_workers):
        _add_node_error(node_errors, node.node_id, "DataLoader `numWorkers` must be a non-negative integer.")
    if prefetch_factor not in {None, ""}:
        if not _is_positive_int(prefetch_factor):
            _add_node_error(node_errors, node.node_id, "DataLoader `prefetchFactor` must be a positive integer when provided.")
        elif int(num_workers or 0) == 0:
            _append_unique(warnings, "DataLoader `prefetchFactor` is ignored when `numWorkers` is 0.")
    if persistent_workers and int(num_workers or 0) == 0:
        _append_unique(
            warnings,
            "DataLoader `persistentWorkers` only takes effect when `numWorkers` is greater than 0.",
        )
    if collate_fn_type not in {"default", "custom_placeholder"}:
        _add_node_error(node_errors, node.node_id, "DataLoader `collateFnType` must be `default` or `custom_placeholder`.")
    elif collate_fn_type == "custom_placeholder":
        _append_unique(
            warnings,
            "DataLoader `collateFnType=custom_placeholder` is a reserved extension point, so the default collate function will still be used.",
        )


def _validate_training_params(node, node_errors: dict[str, list[str]], warnings: list[str]) -> None:
    params = node.params

    if node.op == "Dataset":
        normalized = normalize_dataset_params(params)
        if normalized["datasetMode"] not in {"builtin", "image_folder", "csv"}:
            _add_node_error(node_errors, node.node_id, "Dataset `datasetMode` must be `builtin`, `image_folder`, or `csv`.")
        if not _is_positive_int(normalized["imageSize"]):
            _add_node_error(node_errors, node.node_id, "Dataset `imageSize` must be a positive integer.")
    elif node.op == "DataLoader":
        _validate_dataloader_params(node, node_errors, warnings)
    elif node.op == "Loss":
        if params.get("lossType") not in {"CrossEntropyLoss", "MSELoss"}:
            _add_node_error(node_errors, node.node_id, "Loss `lossType` must be `CrossEntropyLoss` or `MSELoss`.")
    elif node.op == "Optimizer":
        if params.get("optimizerType") not in {"SGD", "Adam"}:
            _add_node_error(node_errors, node.node_id, "Optimizer `optimizerType` must be `SGD` or `Adam`.")
        if not _is_positive_number(params.get("lr")):
            _add_node_error(node_errors, node.node_id, "Optimizer `lr` must be greater than 0.")
        if not isinstance(params.get("weightDecay"), (int, float)) or float(params.get("weightDecay")) < 0:
            _add_node_error(node_errors, node.node_id, "Optimizer `weightDecay` must be zero or greater.")
        if params.get("optimizerType") == "SGD" and (
            not isinstance(params.get("momentum"), (int, float)) or float(params.get("momentum")) < 0
        ):
            _add_node_error(node_errors, node.node_id, "Optimizer `momentum` must be zero or greater for SGD.")
    elif node.op == "Trainer":
        if not _is_positive_int(params.get("epochs")):
            _add_node_error(node_errors, node.node_id, "Trainer `epochs` must be a positive integer.")
        if not _is_positive_int(params.get("logInterval")):
            _add_node_error(node_errors, node.node_id, "Trainer `logInterval` must be a positive integer.")
        if params.get("device") not in {"cpu", "auto"}:
            _add_node_error(node_errors, node.node_id, "Trainer `device` must be `cpu` or `auto`.")
    elif node.op == "Metric":
        if params.get("metricType") != "Accuracy":
            _add_node_error(node_errors, node.node_id, "Metric `metricType` must be `Accuracy` in the current phase.")


def _validate_common_graph(
    ir_graph: IRGraph,
    node_errors: dict[str, list[str]],
    global_errors: list[str],
    warnings: list[str],
) -> None:
    node_map = get_node_map(ir_graph)
    inputs = get_nodes_by_op(ir_graph, "Input")
    outputs = get_nodes_by_op(ir_graph, "Output")

    if not inputs:
        global_errors.append("Graph requires at least one Input node.")
    if not outputs:
        global_errors.append("Graph requires at least one Output node.")

    sorted_ids = topological_sort(ir_graph)
    if sorted_ids is None:
        global_errors.append("Graph contains a cycle.")

    for node in ir_graph.nodes:
        if node.category == "unknown":
            _add_node_error(node_errors, node.node_id, f"Unsupported node type `{node.op}`.")
            continue

        if not node.input_refs and not node.output_refs:
            _add_node_error(node_errors, node.node_id, f"Node `{node.node_id}` is disconnected.")

        if node.op == "Input":
            invalid_inputs = [ref for ref in node.input_refs if node_map.get(ref) and node_map[ref].op != "DataLoader"]
            if invalid_inputs:
                _add_node_error(node_errors, node.node_id, "Input may only receive connections from DataLoader.")
        elif node.op == "Output":
            invalid_outputs = [ref for ref in node.output_refs if node_map.get(ref) and node_map[ref].op != "Loss"]
            if invalid_outputs:
                _add_node_error(node_errors, node.node_id, "Output may only connect to Loss in the training pipeline.")
        elif node.category == "model":
            if len([ref for ref in node.input_refs if node_map.get(ref) and node_map[ref].category == "model"]) > 1:
                _add_node_error(node_errors, node.node_id, "The current model graph supports only a single upstream model connection.")
            if len([ref for ref in node.output_refs if node_map.get(ref) and node_map[ref].category == "model"]) > 1:
                _add_node_error(node_errors, node.node_id, "The current model graph supports only a single downstream model connection.")

        if node.category == "model":
            _validate_model_params(node, node_errors)
        else:
            _validate_training_params(node, node_errors, warnings)


def _validate_training_graph(
    ir_graph: IRGraph,
    node_errors: dict[str, list[str]],
    global_errors: list[str],
    warnings: list[str],
) -> None:
    node_map = get_node_map(ir_graph)
    dataset_nodes = get_nodes_by_op(ir_graph, "Dataset")
    dataloader_nodes = get_nodes_by_op(ir_graph, "DataLoader")
    loss_nodes = get_nodes_by_op(ir_graph, "Loss")
    optimizer_nodes = get_nodes_by_op(ir_graph, "Optimizer")
    trainer_nodes = get_nodes_by_op(ir_graph, "Trainer")
    metric_nodes = get_nodes_by_op(ir_graph, "Metric")
    components = resolve_training_components(ir_graph)

    if len(dataset_nodes) != 1:
        global_errors.append("The training graph supports exactly one Dataset node.")
    if len(dataloader_nodes) != 1:
        global_errors.append("The training graph supports exactly one DataLoader node.")
    if len(loss_nodes) != 1:
        global_errors.append("The training graph supports exactly one Loss node.")
    if len(optimizer_nodes) != 1:
        global_errors.append("The training graph supports exactly one Optimizer node.")
    if len(trainer_nodes) != 1:
        global_errors.append("The training graph requires exactly one Trainer node.")
    if len(metric_nodes) > 1:
        global_errors.append("The training graph supports at most one Metric node.")

    if components.dataset_node and components.dataloader_node:
        if components.dataloader_node.node_id not in components.dataset_node.output_refs:
            _add_node_error(node_errors, components.dataset_node.node_id, "Dataset must connect to DataLoader.")
        if any(node_map.get(ref) and node_map[ref].op != "Dataset" for ref in components.dataloader_node.input_refs):
            _add_node_error(node_errors, components.dataloader_node.node_id, "DataLoader may only receive input from Dataset.")

    if components.dataloader_node and components.input_node:
        if components.input_node.node_id not in components.dataloader_node.output_refs:
            _add_node_error(node_errors, components.dataloader_node.node_id, "DataLoader must connect to the model Input node.")

    if components.output_node and components.loss_node:
        if components.loss_node.node_id not in components.output_node.output_refs:
            _add_node_error(node_errors, components.output_node.node_id, "Model Output must connect to Loss.")
        if any(node_map.get(ref) and node_map[ref].op != "Output" for ref in components.loss_node.input_refs):
            _add_node_error(node_errors, components.loss_node.node_id, "Loss may only receive input from the model Output node.")

    if components.trainer_node:
        trainer_inputs = {node_map[ref].op for ref in components.trainer_node.input_refs if ref in node_map}
        if "Loss" not in trainer_inputs:
            _add_node_error(node_errors, components.trainer_node.node_id, "Trainer must receive Loss.")
        if "Optimizer" not in trainer_inputs:
            _add_node_error(node_errors, components.trainer_node.node_id, "Trainer must receive Optimizer.")
        if metric_nodes and "Metric" not in trainer_inputs:
            _add_node_error(node_errors, components.trainer_node.node_id, "Trainer must receive Metric when Metric exists.")

    if components.optimizer_node and components.trainer_node and components.optimizer_node.output_refs != [components.trainer_node.node_id]:
        _add_node_error(node_errors, components.optimizer_node.node_id, "Optimizer must connect to Trainer.")
    if components.metric_node and components.trainer_node and components.metric_node.output_refs != [components.trainer_node.node_id]:
        _add_node_error(node_errors, components.metric_node.node_id, "Metric must connect to Trainer.")
    if components.dataset_node and components.dataset_node.input_refs:
        _add_node_error(node_errors, components.dataset_node.node_id, "Dataset must not have upstream inputs.")
    if components.optimizer_node and components.optimizer_node.input_refs:
        _add_node_error(node_errors, components.optimizer_node.node_id, "Optimizer does not accept upstream inputs in the current phase.")
    if components.metric_node and components.metric_node.input_refs:
        _add_node_error(node_errors, components.metric_node.node_id, "Metric does not accept upstream inputs in the current phase.")
    if components.trainer_node and components.trainer_node.output_refs:
        _add_node_error(node_errors, components.trainer_node.node_id, "Trainer is the terminal node of the training pipeline.")

    if not components.dataset_node:
        return

    inspection = inspect_dataset_config(components.dataset_node.params)
    for message in inspection.errors:
        _add_node_error(node_errors, components.dataset_node.node_id, message)
    for message in inspection.warnings:
        _append_unique(warnings, f"Dataset: {message}")

    if inspection.dataset_mode == "csv":
        _add_node_error(
            node_errors,
            components.dataset_node.node_id,
            "CSV datasets are not yet runnable in this phase. Use `builtin` or `image_folder` for actual training.",
        )

    train_count = inspection.splits.get("train", inspection.sample_count)
    if inspection.success and train_count <= 0:
        _add_node_error(node_errors, components.dataset_node.node_id, "The training split is empty.")

    if inspection.success and components.input_node and inspection.input_shape:
        model_input_shape = get_input_shape(ir_graph)
        if model_input_shape != inspection.input_shape:
            _add_node_error(
                node_errors,
                components.input_node.node_id,
                f"Model Input shape {model_input_shape} does not match dataset input shape {inspection.input_shape}.",
            )

    if inspection.success and inspection.task_type == "classification":
        if inspection.num_classes <= 0:
            _add_node_error(node_errors, components.dataset_node.node_id, "Classification datasets must expose at least one class.")
        else:
            output_classes = get_num_classes(ir_graph)
            if output_classes != inspection.num_classes:
                target_node_id = components.output_node.node_id if components.output_node else components.dataset_node.node_id
                _add_node_error(
                    node_errors,
                    target_node_id,
                    f"Model output dimension {output_classes} does not match dataset classes {inspection.num_classes}.",
                )

        if components.loss_node and components.loss_node.params.get("lossType") != "CrossEntropyLoss":
            _append_unique(
                warnings,
                "Classification datasets are usually paired with CrossEntropyLoss. The current loss can run, but it is less aligned with the teaching goal.",
            )
    elif inspection.success and inspection.task_type == "regression":
        if components.loss_node and components.loss_node.params.get("lossType") != "MSELoss":
            _add_node_error(
                node_errors,
                components.loss_node.node_id,
                "Regression datasets should use MSELoss in the current training runtime.",
            )


def validate_graph(
    ir_graph: IRGraph,
    require_training: bool = False,
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Validate graph structure, dataset config, and training constraints."""

    global_errors: list[str] = []
    node_errors: dict[str, list[str]] = {}
    warnings: list[str] = []

    _validate_common_graph(ir_graph, node_errors, global_errors, warnings)

    if require_training:
        _validate_training_graph(ir_graph, node_errors, global_errors, warnings)

    return global_errors, node_errors, warnings
