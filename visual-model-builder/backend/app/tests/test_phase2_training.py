import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def build_phase2_payload(
    include_trainer: bool = True,
    flatten_start_dim: int = 0,
    linear_in_features: int = 3136,
    linear_out_features: int = 10,
    use_two_convs: bool = False,
) -> dict:
    nodes = [
        {
            "id": "Dataset_1",
            "type": "Dataset",
            "position": {"x": 0, "y": 0},
            "data": {
                "label": "Dataset",
                "params": {
                    "datasetName": "FakeData",
                    "trainSplit": True,
                    "imageSize": 28,
                    "numClasses": 10,
                },
            },
        },
        {
            "id": "DataLoader_1",
            "type": "DataLoader",
            "position": {"x": 150, "y": 0},
            "data": {
                "label": "DataLoader",
                "params": {
                    "batchSize": 32,
                    "shuffle": True,
                    "numWorkers": 0,
                },
            },
        },
        {
            "id": "Input_1",
            "type": "Input",
            "position": {"x": 300, "y": 0},
            "data": {"label": "Input", "params": {"inputShape": [1, 28, 28]}},
        },
        {
            "id": "Conv2d_1",
            "type": "Conv2d",
            "position": {"x": 450, "y": 0},
            "data": {
                "label": "Conv2d",
                "params": {
                    "in_channels": 1,
                    "out_channels": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
            },
        },
        {
            "id": "ReLU_1",
            "type": "ReLU",
            "position": {"x": 600, "y": 0},
            "data": {"label": "ReLU", "params": {"inplace": False}},
        },
        {
            "id": "MaxPool2d_1",
            "type": "MaxPool2d",
            "position": {"x": 750, "y": 0},
            "data": {
                "label": "MaxPool2d",
                "params": {"kernel_size": 2, "stride": 2, "padding": 0},
            },
        },
        {
            "id": "Flatten_1",
            "type": "Flatten",
            "position": {"x": 900, "y": 0},
            "data": {"label": "Flatten", "params": {"start_dim": flatten_start_dim, "end_dim": -1}},
        },
        {
            "id": "Linear_1",
            "type": "Linear",
            "position": {"x": 1050, "y": 0},
            "data": {
                "label": "Linear",
                "params": {"in_features": linear_in_features, "out_features": linear_out_features, "bias": True},
            },
        },
        {
            "id": "Output_1",
            "type": "Output",
            "position": {"x": 1200, "y": 0},
            "data": {"label": "Output", "params": {}},
        },
        {
            "id": "Loss_1",
            "type": "Loss",
            "position": {"x": 1350, "y": 0},
            "data": {"label": "Loss", "params": {"lossType": "CrossEntropyLoss"}},
        },
        {
            "id": "Optimizer_1",
            "type": "Optimizer",
            "position": {"x": 1350, "y": 150},
            "data": {
                "label": "Optimizer",
                "params": {
                    "optimizerType": "Adam",
                    "lr": 0.001,
                    "weightDecay": 0.0,
                    "momentum": 0.9,
                },
            },
        },
        {
            "id": "Metric_1",
            "type": "Metric",
            "position": {"x": 1350, "y": 300},
            "data": {"label": "Metric", "params": {"metricType": "Accuracy"}},
        },
    ]

    if use_two_convs:
        nodes[3]["data"]["params"]["padding"] = 0
        nodes.insert(
            4,
            {
                "id": "Conv2d_2",
                "type": "Conv2d",
                "position": {"x": 525, "y": 0},
                "data": {
                    "label": "Conv2d",
                    "params": {
                        "in_channels": 16,
                        "out_channels": 10,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 0,
                    },
                },
            },
        )
        nodes = [node for node in nodes if node["id"] != "MaxPool2d_1"]

    if include_trainer:
        nodes.append(
            {
                "id": "Trainer_1",
                "type": "Trainer",
                "position": {"x": 1500, "y": 150},
                "data": {
                    "label": "Trainer",
                    "params": {
                        "epochs": 1,
                        "device": "cpu",
                        "logInterval": 1,
                        "validateEveryEpoch": False,
                    },
                },
            }
        )

    if use_two_convs:
        edges = [
            {"id": "e1", "source": "Dataset_1", "target": "DataLoader_1"},
            {"id": "e2", "source": "DataLoader_1", "target": "Input_1"},
            {"id": "e3", "source": "Input_1", "target": "Conv2d_1"},
            {"id": "e4", "source": "Conv2d_1", "target": "Conv2d_2"},
            {"id": "e5", "source": "Conv2d_2", "target": "ReLU_1"},
            {"id": "e6", "source": "ReLU_1", "target": "Flatten_1"},
            {"id": "e7", "source": "Flatten_1", "target": "Linear_1"},
            {"id": "e8", "source": "Linear_1", "target": "Output_1"},
            {"id": "e9", "source": "Output_1", "target": "Loss_1"},
        ]
    else:
        edges = [
            {"id": "e1", "source": "Dataset_1", "target": "DataLoader_1"},
            {"id": "e2", "source": "DataLoader_1", "target": "Input_1"},
            {"id": "e3", "source": "Input_1", "target": "Conv2d_1"},
            {"id": "e4", "source": "Conv2d_1", "target": "ReLU_1"},
            {"id": "e5", "source": "ReLU_1", "target": "MaxPool2d_1"},
            {"id": "e6", "source": "MaxPool2d_1", "target": "Flatten_1"},
            {"id": "e7", "source": "Flatten_1", "target": "Linear_1"},
            {"id": "e8", "source": "Linear_1", "target": "Output_1"},
            {"id": "e9", "source": "Output_1", "target": "Loss_1"},
        ]

    if include_trainer:
        edges.extend(
            [
                {"id": "e10", "source": "Loss_1", "target": "Trainer_1"},
                {"id": "e11", "source": "Optimizer_1", "target": "Trainer_1"},
                {"id": "e12", "source": "Metric_1", "target": "Trainer_1"},
            ]
        )

    return {
        "project": {
            "version": "2.0.0",
            "metadata": {
                "name": "Phase 2 Training Example",
                "createdAt": "2026-04-03T00:00:00Z",
                "updatedAt": "2026-04-03T00:00:00Z",
            },
            "nodes": nodes,
            "edges": edges,
        }
    }


def test_validate_training_graph_accepts_minimal_phase2_pipeline():
    response = client.post("/validate-training-graph", json=build_phase2_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["globalErrors"] == []


def test_generate_training_code_contains_training_sections():
    response = client.post("/generate-training-code", json=build_phase2_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "DataLoader" in data["code"]
    assert "def train()" in data["code"]
    assert "CrossEntropyLoss" in data["code"]


def test_generate_model_code_still_works_when_training_nodes_exist():
    response = client.post("/generate-code", json=build_phase2_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "class Model(nn.Module):" in data["code"]
    assert "def train()" not in data["code"]


def test_generate_model_code_shifts_flatten_start_dim_for_batched_runtime():
    response = client.post(
        "/generate-code",
        json=build_phase2_payload(
            flatten_start_dim=1,
            linear_in_features=576,
            linear_out_features=1,
            use_two_convs=True,
        ),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "nn.Flatten(start_dim=2, end_dim=-1)" in data["code"]


def test_run_training_returns_epoch_metrics(monkeypatch):
    runs_dir = Path(__file__).resolve().parents[2] / ".test_training_runs" / "epoch_metrics"
    runs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VMB_TRAINING_RUNS_DIR", str(runs_dir))
    response = client.post("/run-training", json=build_phase2_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["status"] == "completed"
    assert len(data["logs"]) == 1
    assert data["logs"][0]["loss"] >= 0
    assert data["logs"][0]["accuracy"] is not None
    assert data["diagnostics"]["ok"] is True
    assert "summary" in data["diagnostics"]
    assert "trendSummary" in data["insights"]
    assert data["trainingMetadata"]["projectName"] == "Phase 2 Training Example"
    assert data["trainingMetadata"]["datasetUsed"] == "FakeData"
    assert Path(data["trainingMetadata"]["weightsPath"]).exists()
    assert Path(data["trainingMetadata"]["logsPath"]).exists()
    assert Path(data["trainingMetadata"]["summaryPath"]).exists()
    summary_payload = json.loads(Path(data["trainingMetadata"]["summaryPath"]).read_text(encoding="utf-8"))
    assert summary_payload["diagnostics"]["summary"]
    assert summary_payload["insights"]["qualitySummary"]


def test_run_training_accepts_sample_relative_flatten_dimensions(monkeypatch):
    runs_dir = Path(__file__).resolve().parents[2] / ".test_training_runs" / "sample_relative_flatten"
    runs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VMB_TRAINING_RUNS_DIR", str(runs_dir))
    response = client.post(
        "/run-training",
        json=build_phase2_payload(
            flatten_start_dim=1,
            linear_in_features=576,
            linear_out_features=1,
            use_two_convs=True,
        ),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["status"] == "completed"
    assert len(data["logs"]) == 1
    assert data["logs"][0]["loss"] >= 0
    assert Path(data["trainingMetadata"]["weightsPath"]).exists()


def test_validate_training_graph_reports_missing_trainer():
    response = client.post("/validate-training-graph", json=build_phase2_payload(include_trainer=False))

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert any("Trainer" in message for message in data["globalErrors"])


def test_diagnose_training_graph_returns_teaching_oriented_summary():
    response = client.post("/diagnose-training-graph", json=build_phase2_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "summary" in data
    assert "warnings" in data
    assert "suggestions" in data
    assert data["graphStats"]["hasTrainer"] is True
    assert data["modelStats"]["parameterCount"] > 0
    assert data["trainingStats"]["datasetName"] == "FakeData"
    assert any("epoch" in message.lower() for message in data["warnings"])


def test_diagnose_training_graph_flags_output_class_mismatch():
    response = client.post(
        "/diagnose-training-graph",
        json=build_phase2_payload(linear_out_features=1),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert any("classes" in message.lower() for message in data["errors"])
    assert any("Linear" in message or "out_features" in message for message in data["suggestions"])


def test_diagnose_training_graph_respects_flatten_runtime_class_normalization():
    response = client.post(
        "/diagnose-training-graph",
        json=build_phase2_payload(
            flatten_start_dim=1,
            linear_in_features=576,
            linear_out_features=1,
            use_two_convs=True,
        ),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["modelStats"]["outputClasses"] == 10
    assert not any("predicts 1 classes" in message for message in data["errors"])


def test_run_training_blocks_when_diagnostics_fail():
    response = client.post(
        "/run-training",
        json=build_phase2_payload(linear_out_features=1),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert data["status"] == "diagnostics_failed"
    assert data["logs"] == []
    assert data["trainingMetadata"] is None
    assert data["diagnostics"]["ok"] is False
    assert data["insights"]["failureExplanation"]
