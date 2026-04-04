"""Training diagnostics and teaching-oriented run interpretation for Phase 3."""

from __future__ import annotations

from math import ceil
from typing import Any

from app.schemas.responses import (
    TrainingConfigStats,
    TrainingDiagnosticsResponse,
    TrainingGraphStats,
    TrainingInsightsResponse,
    TrainingModelStats,
)
from app.services.shape_infer import infer_graph_shapes
from app.services.training import (
    get_input_shape,
    get_node_map,
    get_num_classes,
    get_model_nodes_in_order,
    resolve_training_components,
)
from app.services.validator import validate_graph

LEARNABLE_MODEL_OPS = {"Conv2d", "Linear"}


def _append_unique(items: list[str], message: str) -> None:
    if message and message not in items:
        items.append(message)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _estimate_dataset_size(dataset_name: str) -> int:
    if dataset_name in {"MNIST", "FakeData"}:
        # Phase 2 runtime currently caps the teaching dataset to a small subset.
        return 256
    return 0


def _estimate_model_parameters(ir_graph, node_results: dict[str, Any]) -> int:
    total = 0
    for node in get_model_nodes_in_order(ir_graph):
        params = node.params
        if node.op == "Conv2d":
            out_channels = _safe_int(params.get("out_channels"), 0)
            in_channels = _safe_int(params.get("in_channels"), 0)
            kernel_size = _safe_int(params.get("kernel_size"), 0)
            if out_channels > 0 and in_channels > 0 and kernel_size > 0:
                total += out_channels * in_channels * kernel_size * kernel_size
                total += out_channels  # Conv2d uses bias=True in the current runtime/codegen.
        elif node.op == "Linear":
            out_features = _safe_int(params.get("out_features"), 0)
            in_features = _safe_int(params.get("in_features"), 0)
            bias = bool(params.get("bias", True))
            if out_features > 0 and in_features > 0:
                total += out_features * in_features
                if bias:
                    total += out_features
    return total


def _complexity_label(parameter_count: int, learnable_layers: int) -> str:
    if learnable_layers == 0 or parameter_count == 0:
        return "no_learnable_layers"
    if parameter_count < 2_000:
        return "tiny"
    if parameter_count < 50_000:
        return "simple"
    if parameter_count < 500_000:
        return "moderate"
    return "deep"


def _build_graph_stats(ir_graph) -> TrainingGraphStats:
    model_nodes = [node for node in ir_graph.nodes if node.category == "model"]
    data_nodes = [node for node in ir_graph.nodes if node.category == "data"]
    training_nodes = [node for node in ir_graph.nodes if node.category == "training"]
    learnable_layers = [node for node in model_nodes if node.op in LEARNABLE_MODEL_OPS]
    components = resolve_training_components(ir_graph)

    return TrainingGraphStats(
        totalNodes=len(ir_graph.nodes),
        totalEdges=len(ir_graph.edges),
        modelNodeCount=len(model_nodes),
        dataNodeCount=len(data_nodes),
        trainingNodeCount=len(training_nodes),
        learnableLayerCount=len(learnable_layers),
        modelDepth=len(get_model_nodes_in_order(ir_graph)),
        hasMetric=components.metric_node is not None,
        hasTrainer=components.trainer_node is not None,
    )


def _build_model_stats(ir_graph, node_results: dict[str, Any]) -> TrainingModelStats:
    model_nodes = get_model_nodes_in_order(ir_graph)
    learnable_layers = [node for node in model_nodes if node.op in LEARNABLE_MODEL_OPS]
    parameter_count = _estimate_model_parameters(ir_graph, node_results)
    flatten_features = None

    for node in model_nodes:
        if node.op == "Flatten":
            output_shape = node_results.get(node.node_id).output_shape if node.node_id in node_results else None
            if output_shape:
                flatten_features = output_shape[-1]

    return TrainingModelStats(
        parameterCount=parameter_count,
        trainableParameterCount=parameter_count,
        learnableLayerCount=len(learnable_layers),
        outputClasses=get_num_classes(ir_graph, node_results),
        flattenFeatures=flatten_features,
        inputShape=get_input_shape(ir_graph),
        complexityLabel=_complexity_label(parameter_count, len(learnable_layers)),
    )


def _build_training_stats(ir_graph) -> TrainingConfigStats:
    components = resolve_training_components(ir_graph)
    dataset_params = components.dataset_node.params if components.dataset_node else {}
    dataloader_params = components.dataloader_node.params if components.dataloader_node else {}
    loss_params = components.loss_node.params if components.loss_node else {}
    optimizer_params = components.optimizer_node.params if components.optimizer_node else {}
    trainer_params = components.trainer_node.params if components.trainer_node else {}
    metric_params = components.metric_node.params if components.metric_node else {}

    dataset_name = str(dataset_params.get("datasetName", "Unknown"))
    estimated_dataset_size = _estimate_dataset_size(dataset_name)
    batch_size = max(_safe_int(dataloader_params.get("batchSize"), 0), 0)

    return TrainingConfigStats(
        datasetName=dataset_name,
        estimatedDatasetSize=estimated_dataset_size,
        numClasses=_safe_int(dataset_params.get("numClasses"), get_num_classes(ir_graph)),
        batchSize=batch_size,
        estimatedBatchesPerEpoch=ceil(estimated_dataset_size / batch_size) if estimated_dataset_size and batch_size else 0,
        epochs=_safe_int(trainer_params.get("epochs"), 0),
        optimizerType=str(optimizer_params.get("optimizerType", "")),
        learningRate=_safe_float(optimizer_params.get("lr"), 0.0),
        weightDecay=_safe_float(optimizer_params.get("weightDecay"), 0.0),
        momentum=_safe_float(optimizer_params.get("momentum"), 0.0) if components.optimizer_node else None,
        lossType=str(loss_params.get("lossType", "")),
        metricType=str(metric_params.get("metricType")) if components.metric_node else None,
        device=str(trainer_params.get("device", "")),
    )


def diagnose_training_graph(ir_graph) -> TrainingDiagnosticsResponse:
    """Return teaching-oriented diagnostics for the current training graph."""

    global_errors, node_errors = validate_graph(ir_graph, require_training=True)
    node_results = infer_graph_shapes(ir_graph)
    node_map = get_node_map(ir_graph)
    components = resolve_training_components(ir_graph)
    graph_stats = _build_graph_stats(ir_graph)
    model_stats = _build_model_stats(ir_graph, node_results)
    training_stats = _build_training_stats(ir_graph)

    warnings: list[str] = []
    errors: list[str] = []
    suggestions: list[str] = []

    for message in global_errors:
        _append_unique(errors, message)

    for node_id, node_messages in node_errors.items():
        node_name = node_map.get(node_id).op if node_id in node_map else node_id
        for message in node_messages:
            _append_unique(errors, f"{node_name}: {message}")

    for node_id, result in node_results.items():
        node_name = node_map.get(node_id).op if node_id in node_map else node_id
        for message in result.errors:
            _append_unique(errors, f"{node_name}: {message}")

    output_classes = model_stats.output_classes
    dataset_classes = training_stats.num_classes

    if output_classes < dataset_classes:
        _append_unique(
            errors,
            f"Model predicts {output_classes} classes, but the dataset is configured for {dataset_classes} classes.",
        )
        _append_unique(
            suggestions,
            "Increase the last Linear layer `out_features` so it can represent every dataset class.",
        )
    elif output_classes != dataset_classes:
        _append_unique(
            warnings,
            f"Model predicts {output_classes} classes while the dataset expects {dataset_classes}. Training may still run, but the teaching story is less clear.",
        )
        _append_unique(
            suggestions,
            "Match the final Linear `out_features` to the dataset `numClasses` for a cleaner classification example.",
        )

    for node in get_model_nodes_in_order(ir_graph):
        if node.op != "Linear":
            continue

        input_shape = node_results.get(node.node_id).input_shape if node.node_id in node_results else None
        in_features = _safe_int(node.params.get("in_features"), 0)
        if input_shape is None:
            _append_unique(
                warnings,
                f"{node.node_id} could not confirm its incoming shape. Linear layers are fragile when the upstream shape is still unknown.",
            )
            _append_unique(
                suggestions,
                "Resolve upstream shape errors before trusting the current Linear `in_features` value.",
            )
            continue

        if input_shape[-1] != in_features:
            _append_unique(
                errors,
                f"{node.node_id} expects in_features={in_features}, but the inferred upstream feature size is {input_shape[-1]}.",
            )
            _append_unique(
                suggestions,
                "Update the Linear `in_features` to match the Flatten output shown by shape inference.",
            )

        upstream_model_inputs = [ref for ref in node.input_refs if ref in node_map and node_map[ref].category == "model"]
        if len(input_shape) > 1 and upstream_model_inputs:
            upstream_node = node_map[upstream_model_inputs[0]]
            if upstream_node.op != "Flatten":
                _append_unique(
                    warnings,
                    f"{node.node_id} receives a rank-{len(input_shape)} tensor from {upstream_node.op}. Add Flatten when you want a classic fully connected classifier head.",
                )
                _append_unique(
                    suggestions,
                    "Insert a Flatten node before Linear if you want to collapse spatial dimensions explicitly.",
                )

    for node in get_model_nodes_in_order(ir_graph):
        if node.op != "Flatten":
            continue

        start_dim = _safe_int(node.params.get("start_dim"), 0)
        end_dim = _safe_int(node.params.get("end_dim"), -1)
        if start_dim != 0 or end_dim != -1:
            _append_unique(
                warnings,
                f"{node.node_id} uses Flatten(start_dim={start_dim}, end_dim={end_dim}). In this project, Flatten dimensions are sample-relative and non-negative start_dim is shifted at runtime to skip the batch axis.",
            )
            _append_unique(
                suggestions,
                "Prefer the default Flatten settings unless you intentionally want a partial flatten and understand the sample-relative semantics.",
            )

    learning_rate = training_stats.learning_rate
    optimizer_type = training_stats.optimizer_type
    if learning_rate > 0:
        if optimizer_type == "Adam" and learning_rate > 0.01:
            _append_unique(
                warnings,
                f"Learning rate {learning_rate:g} is high for Adam and may make the loss curve unstable in a short teaching run.",
            )
            _append_unique(suggestions, "Try Adam with a learning rate around 0.001 for a smoother teaching example.")
        elif learning_rate > 0.1:
            _append_unique(
                warnings,
                f"Learning rate {learning_rate:g} is very high and may cause exploding or oscillating loss.",
            )
            _append_unique(suggestions, "Reduce the learning rate and rerun the training demo.")
        elif learning_rate < 1e-5:
            _append_unique(
                warnings,
                f"Learning rate {learning_rate:g} is very small, so the demo may show little visible progress.",
            )
            _append_unique(
                suggestions,
                "Increase the learning rate slightly if you want the loss curve to change more clearly during class.",
            )

    batch_size = training_stats.batch_size
    estimated_dataset_size = training_stats.estimated_dataset_size
    if batch_size == 1:
        _append_unique(
            warnings,
            "Batch size 1 often produces very noisy curves, which can distract from the teaching goal.",
        )
        _append_unique(suggestions, "Try a batch size between 16 and 64 for a steadier teaching demo.")
    elif batch_size > 0 and estimated_dataset_size and batch_size > estimated_dataset_size:
        _append_unique(
            warnings,
            f"Batch size {batch_size} is larger than the current teaching dataset size {estimated_dataset_size}. Each epoch will have only one step.",
        )
        _append_unique(
            suggestions,
            "Use a smaller batch size so students can see multiple optimization steps per epoch.",
        )
    elif batch_size > 128:
        _append_unique(
            warnings,
            f"Batch size {batch_size} is large for the current 256-sample teaching dataset, so each epoch will produce very few updates.",
        )
        _append_unique(suggestions, "Consider a smaller batch size if you want more visible learning dynamics.")

    if training_stats.epochs < 2:
        _append_unique(
            warnings,
            f"Only {training_stats.epochs} epoch is configured. That is enough to prove the loop runs, but usually too short for a meaningful teaching curve.",
        )
        _append_unique(
            suggestions,
            "Use at least 2-3 epochs when you want students to observe a trend instead of a single point.",
        )

    if components.metric_node is None:
        _append_unique(
            warnings,
            "No Metric node is connected, so students will only see loss and not model quality.",
        )
        _append_unique(
            suggestions,
            "Add a Metric node (Accuracy) and connect it to Trainer so the run reports both loss and quality.",
        )

    loss_type = training_stats.loss_type
    if loss_type == "CrossEntropyLoss":
        if output_classes <= 1:
            _append_unique(
                errors,
                "CrossEntropyLoss expects class logits, but the model currently exposes only one output feature.",
            )
            _append_unique(
                suggestions,
                "Increase the final Linear `out_features` above 1 when using CrossEntropyLoss for classification.",
            )
    elif loss_type == "MSELoss":
        _append_unique(
            warnings,
            "MSELoss can run, but it is less intuitive than CrossEntropyLoss for a classification-focused teaching example.",
        )
        _append_unique(
            suggestions,
            "Switch to CrossEntropyLoss if the lesson is about multi-class classification.",
        )

    if optimizer_type == "SGD" and training_stats.momentum is not None and training_stats.momentum <= 0:
        _append_unique(
            warnings,
            "SGD is configured without momentum, so convergence may look unnecessarily slow in a short demo.",
        )
        _append_unique(
            suggestions,
            "Set SGD momentum to around 0.9 if you want a faster and clearer teaching curve.",
        )

    if graph_stats.learnable_layer_count == 0:
        _append_unique(
            errors,
            "The model currently has no learnable layer. Add Conv2d or Linear layers before expecting optimization to do anything interesting.",
        )
        _append_unique(
            suggestions,
            "Insert at least one Conv2d or Linear node before Output so the model can actually learn.",
        )
    elif graph_stats.learnable_layer_count <= 1 or model_stats.parameter_count < 500:
        _append_unique(
            warnings,
            "The model is extremely small, so it may underfit and leave students with a weak example of representation learning.",
        )
        _append_unique(
            suggestions,
            "Add another learnable layer or increase channel/feature width if you want a richer teaching example.",
        )
    elif graph_stats.learnable_layer_count >= 6 or model_stats.parameter_count > 1_000_000:
        _append_unique(
            warnings,
            "The model is relatively deep for a teaching-first demo, so it may hide the core ideas behind too much complexity.",
        )
        _append_unique(
            suggestions,
            "Simplify the model if the lesson focuses on understanding the training pipeline rather than scaling depth.",
        )

    if errors:
        summary = f"Training is blocked by {len(errors)} issue(s). Fix the blocking items before running the loop."
    elif warnings:
        summary = f"Training can run, but {len(warnings)} warning(s) suggest the current setup may be hard to explain or tune."
    else:
        summary = "The training graph looks healthy for a small teaching run."

    return TrainingDiagnosticsResponse(
        ok=len(errors) == 0,
        summary=summary,
        warnings=warnings,
        errors=errors,
        suggestions=suggestions,
        graphStats=graph_stats,
        modelStats=model_stats,
        trainingStats=training_stats,
    )


def build_training_insights(
    diagnostics: TrainingDiagnosticsResponse | None,
    logs: list[dict[str, float | int | None]],
    status: str,
    runtime_messages: list[str],
    training_metadata: dict[str, object] | None,
) -> TrainingInsightsResponse:
    """Turn raw metrics plus diagnostics into teaching-oriented text."""

    possible_causes: list[str] = []
    suggested_fixes: list[str] = []
    if diagnostics is not None:
        for message in diagnostics.warnings:
            _append_unique(possible_causes, message)
        for message in diagnostics.suggestions:
            _append_unique(suggested_fixes, message)

    metadata = training_metadata or {}
    training_stats = diagnostics.training_stats if diagnostics is not None else None
    model_stats = diagnostics.model_stats if diagnostics is not None else None

    dataset_name = str(metadata.get("datasetUsed") or metadata.get("requestedDatasetName") or (training_stats.dataset_name if training_stats else "unknown"))
    epochs = _safe_int(metadata.get("epochs"), training_stats.epochs if training_stats else 0)
    batch_size = _safe_int(metadata.get("batchSize"), training_stats.batch_size if training_stats else 0)
    optimizer_type = str(metadata.get("optimizerType") or (training_stats.optimizer_type if training_stats else "unknown"))
    learning_rate = _safe_float(metadata.get("learningRate"), training_stats.learning_rate if training_stats else 0.0)
    device = str(metadata.get("device") or (training_stats.device if training_stats else "unknown"))
    metric_type = metadata.get("metricType") or (training_stats.metric_type if training_stats else None)
    configuration_summary = (
        f"This run used {dataset_name} on {device} with {optimizer_type}, learning rate {learning_rate:g}, "
        f"batch size {batch_size}, and {epochs} epoch(s)."
    )

    if model_stats is not None:
        model_summary = (
            f"The model has {model_stats.parameter_count:,} trainable parameters across "
            f"{model_stats.learnable_layer_count} learnable layer(s), which makes it a "
            f"{model_stats.complexity_label} teaching example."
        )
    else:
        model_summary = "Model complexity statistics were unavailable for this run."

    failure_explanation: str | None = None
    trend_summary = "No training metrics are available yet."
    quality_summary = "Run training to generate a curve interpretation."

    if status != "completed":
        failure_explanation = runtime_messages[0] if runtime_messages else (
            diagnostics.summary if diagnostics is not None else "Training did not complete."
        )
        trend_summary = "Training did not complete, so there is no loss/accuracy trend to analyze."
        quality_summary = "Resolve the blocking issue, rerun training, and then inspect the curves."
        if not suggested_fixes:
            _append_unique(suggested_fixes, "Fix the blocking issue reported above and rerun training.")
        if not possible_causes and runtime_messages:
            for message in runtime_messages:
                _append_unique(possible_causes, message)
    elif logs:
        first_log = logs[0]
        last_log = logs[-1]
        first_loss = _safe_float(first_log.get("loss"))
        last_loss = _safe_float(last_log.get("loss"))
        first_accuracy = first_log.get("accuracy")
        last_accuracy = last_log.get("accuracy")

        if len(logs) == 1:
            trend_summary = (
                f"Only one epoch ran, so the result mainly proves the training loop works. "
                f"Loss ended at {last_loss:.4f}."
            )
        elif last_loss < first_loss * 0.8:
            trend_summary = (
                f"Loss decreased clearly from {first_loss:.4f} to {last_loss:.4f}, which suggests the optimizer is moving in the right direction."
            )
        elif last_loss < first_loss * 0.98:
            trend_summary = (
                f"Loss decreased slightly from {first_loss:.4f} to {last_loss:.4f}. The model is learning, but the signal is still modest."
            )
        elif last_loss <= first_loss * 1.02:
            trend_summary = (
                f"Loss stayed almost flat ({first_loss:.4f} -> {last_loss:.4f}), so the current setup is not producing a strong teaching curve yet."
            )
            _append_unique(possible_causes, "The training run may be too short or the learning rate may be poorly tuned.")
            _append_unique(suggested_fixes, "Increase epochs or retune the learning rate to produce a clearer curve.")
        else:
            trend_summary = (
                f"Loss increased from {first_loss:.4f} to {last_loss:.4f}, which usually means the optimization setup is unstable for this demo."
            )
            _append_unique(possible_causes, "An aggressive learning rate or a configuration mismatch may be destabilizing training.")
            _append_unique(suggested_fixes, "Lower the learning rate and double-check output classes, loss type, and Flatten settings.")

        if isinstance(last_accuracy, (int, float)):
            if isinstance(first_accuracy, (int, float)) and len(logs) > 1:
                trend_summary += f" Accuracy moved from {first_accuracy * 100:.2f}% to {last_accuracy * 100:.2f}%."
            else:
                trend_summary += f" Final accuracy was {last_accuracy * 100:.2f}%."

            num_classes = _safe_int(metadata.get("numClasses"), training_stats.num_classes if training_stats else 0)
            weak_accuracy_threshold = 1 / max(num_classes, 2)
            if last_accuracy >= 0.8:
                quality_summary = "The run ended with strong accuracy, so the configuration is already a solid teaching example."
            elif last_accuracy >= 0.5:
                quality_summary = "The model is learning, but there is still room to improve clarity or final performance."
            elif last_accuracy <= weak_accuracy_threshold * 1.5:
                quality_summary = "Accuracy remained low for the configured task, so students may mostly learn about failure modes unless the setup is adjusted."
                _append_unique(possible_causes, "The model may be too small, the run may be too short, or the output configuration may not match the dataset well.")
                _append_unique(suggested_fixes, "Try adding capacity, increasing epochs, or aligning the classifier head with the dataset classes.")
            else:
                quality_summary = "Accuracy improved somewhat, but the final quality is still modest for a clean teaching demonstration."
        else:
            quality_summary = "Loss is available, but model quality is harder to judge because no metric result was produced."
            _append_unique(possible_causes, "Without a Metric node, students must infer quality from loss alone.")
            _append_unique(suggested_fixes, "Add an Accuracy metric so the lesson includes both optimization and evaluation signals.")
    else:
        quality_summary = "Training finished without curve data, so there is nothing to interpret yet."

    return TrainingInsightsResponse(
        configurationSummary=configuration_summary,
        modelSummary=model_summary,
        trendSummary=trend_summary,
        qualitySummary=quality_summary,
        failureExplanation=failure_explanation,
        possibleCauses=possible_causes,
        suggestedFixes=suggested_fixes,
    )
