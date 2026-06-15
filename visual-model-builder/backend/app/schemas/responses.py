"""Response schemas for API endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class NodeShapeResult(BaseModel):
    """Shape inference result for a single node."""

    input_shape: list[int] | None = Field(None, alias="inputShape")
    output_shape: list[int] | None = Field(None, alias="outputShape")
    errors: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class ValidateGraphResponse(BaseModel):
    """Response from graph validation endpoints."""

    ok: bool
    global_errors: list[str] = Field(default_factory=list, alias="globalErrors")
    node_errors: dict[str, list[str]] = Field(default_factory=dict, alias="nodeErrors")
    warnings: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class InferShapesResponse(BaseModel):
    """Response from /infer-shapes."""

    ok: bool
    nodes: dict[str, NodeShapeResult] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class GenerateCodeResponse(BaseModel):
    """Response from code generation endpoints."""

    ok: bool
    code: str = ""
    errors: list[str] = Field(default_factory=list)


class InspectDatasetResponse(BaseModel):
    """Response from /inspect-dataset."""

    success: bool
    dataset_mode: str = Field("", alias="datasetMode")
    resolved_split_mode: str = Field("", alias="resolvedSplitMode")
    task_type: str = Field("", alias="taskType")
    sample_count: int = Field(0, alias="sampleCount")
    num_classes: int = Field(0, alias="numClasses")
    class_names: list[str] = Field(default_factory=list, alias="classNames")
    splits: dict[str, int] = Field(default_factory=dict)
    input_shape: list[int] | None = Field(default=None, alias="inputShape")
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class TrainingEpochLog(BaseModel):
    """Metrics captured for a single training epoch."""

    epoch: int
    loss: float
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    val_loss: float | None = Field(default=None, alias="valLoss")
    val_accuracy: float | None = Field(default=None, alias="valAccuracy")
    val_precision: float | None = Field(default=None, alias="valPrecision")
    val_recall: float | None = Field(default=None, alias="valRecall")
    val_f1: float | None = Field(default=None, alias="valF1")

    model_config = {"populate_by_name": True}


class TrainingMetricSnapshot(BaseModel):
    """Compact metric snapshot for one split."""

    loss: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class TrainingClassMetric(BaseModel):
    """Per-class classification metrics."""

    class_index: int = Field(0, alias="classIndex")
    class_name: str = Field("", alias="className")
    support: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    model_config = {"populate_by_name": True}


class TrainingEvaluationSummary(BaseModel):
    """Post-run evaluation metrics and reproducibility metadata."""

    final_train: TrainingMetricSnapshot = Field(default_factory=TrainingMetricSnapshot, alias="finalTrain")
    final_validation: TrainingMetricSnapshot | None = Field(default=None, alias="finalValidation")
    final_test: TrainingMetricSnapshot | None = Field(default=None, alias="finalTest")
    best_validation: TrainingMetricSnapshot | None = Field(default=None, alias="bestValidation")
    best_epoch: int | None = Field(default=None, alias="bestEpoch")
    primary_split: str = Field("train", alias="primarySplit")
    confusion_matrix: list[list[int]] = Field(default_factory=list, alias="confusionMatrix")
    class_metrics: list[TrainingClassMetric] = Field(default_factory=list, alias="classMetrics")
    macro_precision: float | None = Field(default=None, alias="macroPrecision")
    macro_recall: float | None = Field(default=None, alias="macroRecall")
    macro_f1: float | None = Field(default=None, alias="macroF1")
    weighted_f1: float | None = Field(default=None, alias="weightedF1")
    sample_count: int = Field(0, alias="sampleCount")
    class_names: list[str] = Field(default_factory=list, alias="classNames")
    seed: int = 42
    config_hash: str = Field("", alias="configHash")

    model_config = {"populate_by_name": True}


class TrainingRunMetadata(BaseModel):
    """Serializable training metadata and artifact locations."""

    run_id: str = Field("", alias="runId")
    project_name: str = Field("", alias="projectName")
    requested_dataset_name: str = Field("", alias="requestedDatasetName")
    dataset_used: str = Field("", alias="datasetUsed")
    dataset_size: int = Field(0, alias="datasetSize")
    dataset_mode: str = Field("", alias="datasetMode")
    sample_count: int = Field(0, alias="sampleCount")
    image_size: int = Field(0, alias="imageSize")
    num_classes: int = Field(0, alias="numClasses")
    class_names: list[str] = Field(default_factory=list, alias="classNames")
    splits: dict[str, int] = Field(default_factory=dict)
    input_shape: list[int] = Field(default_factory=list, alias="inputShape")
    task_type: str = Field("classification", alias="taskType")
    train_split: bool = Field(True, alias="trainSplit")
    batch_size: int = Field(0, alias="batchSize")
    shuffle: bool = True
    num_workers: int = Field(0, alias="numWorkers")
    drop_last: bool = Field(False, alias="dropLast")
    pin_memory: bool = Field(False, alias="pinMemory")
    persistent_workers: bool = Field(False, alias="persistentWorkers")
    prefetch_factor: int | None = Field(default=None, alias="prefetchFactor")
    collate_fn_type: str = Field("default", alias="collateFnType")
    epochs: int = 0
    device: str = ""
    loss_type: str = Field("", alias="lossType")
    optimizer_type: str = Field("", alias="optimizerType")
    learning_rate: float = Field(0.0, alias="learningRate")
    weight_decay: float = Field(0.0, alias="weightDecay")
    momentum: float | None = None
    metric_type: str | None = Field(default=None, alias="metricType")
    started_at: str = Field("", alias="startedAt")
    completed_at: str = Field("", alias="completedAt")
    duration_seconds: float = Field(0.0, alias="durationSeconds")
    run_directory: str = Field("", alias="runDirectory")
    weights_path: str = Field("", alias="weightsPath")
    logs_path: str = Field("", alias="logsPath")
    summary_path: str = Field("", alias="summaryPath")

    model_config = {"populate_by_name": True}


class TrainingGraphStats(BaseModel):
    """High-level structural statistics for a training graph."""

    total_nodes: int = Field(0, alias="totalNodes")
    total_edges: int = Field(0, alias="totalEdges")
    model_node_count: int = Field(0, alias="modelNodeCount")
    data_node_count: int = Field(0, alias="dataNodeCount")
    training_node_count: int = Field(0, alias="trainingNodeCount")
    learnable_layer_count: int = Field(0, alias="learnableLayerCount")
    model_depth: int = Field(0, alias="modelDepth")
    has_metric: bool = Field(False, alias="hasMetric")
    has_trainer: bool = Field(False, alias="hasTrainer")

    model_config = {"populate_by_name": True}


class TrainingModelStats(BaseModel):
    """Model complexity summary used by diagnostics and insights."""

    parameter_count: int = Field(0, alias="parameterCount")
    trainable_parameter_count: int = Field(0, alias="trainableParameterCount")
    learnable_layer_count: int = Field(0, alias="learnableLayerCount")
    output_classes: int = Field(0, alias="outputClasses")
    flatten_features: int | None = Field(default=None, alias="flattenFeatures")
    input_shape: list[int] = Field(default_factory=list, alias="inputShape")
    complexity_label: str = Field("unknown", alias="complexityLabel")

    model_config = {"populate_by_name": True}


class TrainingConfigStats(BaseModel):
    """Training configuration summary extracted from the graph."""

    dataset_name: str = Field("", alias="datasetName")
    estimated_dataset_size: int = Field(0, alias="estimatedDatasetSize")
    num_classes: int = Field(0, alias="numClasses")
    batch_size: int = Field(0, alias="batchSize")
    estimated_batches_per_epoch: int = Field(0, alias="estimatedBatchesPerEpoch")
    epochs: int = 0
    optimizer_type: str = Field("", alias="optimizerType")
    learning_rate: float = Field(0.0, alias="learningRate")
    weight_decay: float = Field(0.0, alias="weightDecay")
    momentum: float | None = None
    loss_type: str = Field("", alias="lossType")
    metric_type: str | None = Field(default=None, alias="metricType")
    device: str = ""

    model_config = {"populate_by_name": True}


class TrainingDiagnosticsResponse(BaseModel):
    """Pre-run training diagnostics."""

    ok: bool
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    graph_stats: TrainingGraphStats = Field(default_factory=TrainingGraphStats, alias="graphStats")
    model_stats: TrainingModelStats = Field(default_factory=TrainingModelStats, alias="modelStats")
    training_stats: TrainingConfigStats = Field(default_factory=TrainingConfigStats, alias="trainingStats")

    model_config = {"populate_by_name": True}


class TrainingInsightsResponse(BaseModel):
    """Post-run interpretation shown in the analysis panel and export."""

    configuration_summary: str = Field("", alias="configurationSummary")
    model_summary: str = Field("", alias="modelSummary")
    trend_summary: str = Field("", alias="trendSummary")
    quality_summary: str = Field("", alias="qualitySummary")
    failure_explanation: str | None = Field(default=None, alias="failureExplanation")
    possible_causes: list[str] = Field(default_factory=list, alias="possibleCauses")
    suggested_fixes: list[str] = Field(default_factory=list, alias="suggestedFixes")

    model_config = {"populate_by_name": True}


class RunTrainingResponse(BaseModel):
    """Response from /run-training."""

    ok: bool
    status: str = ""
    logs: list[TrainingEpochLog] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    diagnostics: TrainingDiagnosticsResponse | None = None
    insights: TrainingInsightsResponse | None = None
    evaluation: TrainingEvaluationSummary | None = None
    training_metadata: TrainingRunMetadata | None = Field(None, alias="trainingMetadata")

    model_config = {"populate_by_name": True}


class TrainingJobResponse(BaseModel):
    """Serializable state for an asynchronous training job."""

    job_id: str = Field("", alias="jobId")
    ok: bool = False
    status: str = "queued"
    progress: float = 0.0
    cancel_requested: bool = Field(False, alias="cancelRequested")
    logs: list[TrainingEpochLog] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    diagnostics: TrainingDiagnosticsResponse | None = None
    insights: TrainingInsightsResponse | None = None
    evaluation: TrainingEvaluationSummary | None = None
    training_metadata: TrainingRunMetadata | None = Field(None, alias="trainingMetadata")
    created_at: str = Field("", alias="createdAt")
    updated_at: str = Field("", alias="updatedAt")

    model_config = {"populate_by_name": True}


class TrainingRunRecord(BaseModel):
    """One persisted run listed by /training-runs."""

    run_id: str = Field("", alias="runId")
    ok: bool = False
    status: str = ""
    project_name: str = Field("", alias="projectName")
    created_at: str = Field("", alias="createdAt")
    completed_at: str = Field("", alias="completedAt")
    dataset_used: str = Field("", alias="datasetUsed")
    dataset_mode: str = Field("", alias="datasetMode")
    duration_seconds: float | None = Field(default=None, alias="durationSeconds")
    final_loss: float | None = Field(default=None, alias="finalLoss")
    final_accuracy: float | None = Field(default=None, alias="finalAccuracy")
    macro_f1: float | None = Field(default=None, alias="macroF1")
    weighted_f1: float | None = Field(default=None, alias="weightedF1")
    summary_path: str = Field("", alias="summaryPath")

    model_config = {"populate_by_name": True}


class TrainingRunListResponse(BaseModel):
    """Response from /training-runs."""

    ok: bool = True
    runs: list[TrainingRunRecord] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class TrainingRunDetailResponse(BaseModel):
    """Response from /training-runs/{run_id}."""

    ok: bool = True
    run_id: str = Field("", alias="runId")
    summary: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}
