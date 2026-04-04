"""Response schemas for API endpoints."""

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


class TrainingEpochLog(BaseModel):
    """Metrics captured for a single training epoch."""

    epoch: int
    loss: float
    accuracy: float | None = None


class TrainingRunMetadata(BaseModel):
    """Serializable training metadata and artifact locations."""

    project_name: str = Field("", alias="projectName")
    requested_dataset_name: str = Field("", alias="requestedDatasetName")
    dataset_used: str = Field("", alias="datasetUsed")
    dataset_size: int = Field(0, alias="datasetSize")
    image_size: int = Field(0, alias="imageSize")
    num_classes: int = Field(0, alias="numClasses")
    train_split: bool = Field(True, alias="trainSplit")
    batch_size: int = Field(0, alias="batchSize")
    shuffle: bool = True
    num_workers: int = Field(0, alias="numWorkers")
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


class RunTrainingResponse(BaseModel):
    """Response from /run-training."""

    ok: bool
    status: str = ""
    logs: list[TrainingEpochLog] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    training_metadata: TrainingRunMetadata | None = Field(None, alias="trainingMetadata")

    model_config = {"populate_by_name": True}
