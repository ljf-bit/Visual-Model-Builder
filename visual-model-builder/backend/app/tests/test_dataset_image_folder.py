from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.tests.test_phase2_training import build_phase2_payload

client = TestClient(app)


def create_image_folder_dataset(root: Path, with_predefined_splits: bool) -> Path:
    if with_predefined_splits:
        split_layout = {
            "train": 3,
            "val": 1,
        }
        for split_name, samples_per_class in split_layout.items():
            for class_name, color in (("class_a", 32), ("class_b", 224)):
                class_dir = root / split_name / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                for index in range(samples_per_class):
                    Image.new("L", (28, 28), color=color).save(class_dir / f"{split_name}_{class_name}_{index}.png")
        return root

    for class_name, color in (("class_a", 32), ("class_b", 224)):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            Image.new("L", (28, 28), color=color).save(class_dir / f"{class_name}_{index}.png")
    return root


def build_image_folder_payload(root_path: Path, split_mode: str = "ratio") -> dict:
    payload = build_phase2_payload(linear_out_features=2)
    dataset_params = payload["project"]["nodes"][0]["data"]["params"]
    dataset_params.clear()
    dataset_params.update(
        {
            "datasetMode": "image_folder",
            "rootPath": str(root_path),
            "splitMode": split_mode,
            "trainRatio": 0.5,
            "valRatio": 0.25,
            "testRatio": 0.25,
            "shuffleBeforeSplit": True,
            "imageSize": 28,
            "colorMode": "grayscale",
            "normalize": False,
            "mean": [0.5],
            "std": [0.5],
            "augmentationEnabled": False,
            "taskType": "classification",
        }
    )

    dataloader_params = payload["project"]["nodes"][1]["data"]["params"]
    dataloader_params.update(
        {
            "batchSize": 2,
            "shuffle": True,
            "numWorkers": 0,
            "dropLast": False,
            "pinMemory": False,
            "persistentWorkers": False,
            "prefetchFactor": 2,
            "collateFnType": "default",
        }
    )

    return payload


def test_inspect_dataset_accepts_predefined_image_folder(tmp_path):
    dataset_root = create_image_folder_dataset(tmp_path / "predefined_dataset", with_predefined_splits=True)
    response = client.post(
        "/inspect-dataset",
        json={
            "config": {
                "datasetMode": "image_folder",
                "rootPath": str(dataset_root),
                "splitMode": "predefined",
                "imageSize": 28,
                "colorMode": "grayscale",
                "normalize": False,
                "taskType": "classification",
            }
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["datasetMode"] == "image_folder"
    assert data["resolvedSplitMode"] == "predefined"
    assert data["sampleCount"] == 8
    assert data["numClasses"] == 2
    assert data["classNames"] == ["class_a", "class_b"]
    assert data["splits"]["train"] == 6
    assert data["splits"]["val"] == 2
    assert data["inputShape"] == [1, 28, 28]


def test_inspect_dataset_rejects_empty_image_folder(tmp_path):
    empty_root = tmp_path / "empty_dataset"
    empty_root.mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/inspect-dataset",
        json={
            "config": {
                "datasetMode": "image_folder",
                "rootPath": str(empty_root),
                "splitMode": "ratio",
                "trainRatio": 0.7,
                "valRatio": 0.2,
                "testRatio": 0.1,
                "imageSize": 28,
                "colorMode": "grayscale",
                "normalize": False,
                "taskType": "classification",
            }
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert any("class subdirectories" in message.lower() for message in data["errors"])


def test_validate_training_graph_flags_output_class_mismatch_for_image_folder(tmp_path):
    dataset_root = create_image_folder_dataset(tmp_path / "ratio_dataset", with_predefined_splits=False)
    payload = build_image_folder_payload(dataset_root, split_mode="ratio")
    payload["project"]["nodes"][7]["data"]["params"]["out_features"] = 3

    response = client.post("/validate-training-graph", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert any("output dimension" in message.lower() for messages in data["nodeErrors"].values() for message in messages)


def test_validate_training_graph_reports_dataloader_param_errors_and_warnings(tmp_path):
    dataset_root = create_image_folder_dataset(tmp_path / "loader_dataset", with_predefined_splits=False)
    payload = build_image_folder_payload(dataset_root, split_mode="ratio")
    dataloader_params = payload["project"]["nodes"][1]["data"]["params"]
    dataloader_params["batchSize"] = 0
    dataloader_params["persistentWorkers"] = True
    dataloader_params["prefetchFactor"] = 4

    response = client.post("/validate-training-graph", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert any("batchsize" in message.lower() for message in data["nodeErrors"]["DataLoader_1"])
    assert any("persistentworkers" in message.lower() for message in data["warnings"])


def test_generate_training_code_supports_image_folder(tmp_path):
    dataset_root = create_image_folder_dataset(tmp_path / "codegen_dataset", with_predefined_splits=False)
    payload = build_image_folder_payload(dataset_root, split_mode="ratio")

    response = client.post("/generate-training-code", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "ImageFolder" in data["code"]
    assert "build_datasets" in data["code"]
    assert str(dataset_root) in data["code"]


def test_run_training_supports_image_folder(monkeypatch, tmp_path):
    dataset_root = create_image_folder_dataset(tmp_path / "runtime_dataset", with_predefined_splits=False)
    payload = build_image_folder_payload(dataset_root, split_mode="ratio")
    runs_dir = tmp_path / "training_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VMB_TRAINING_RUNS_DIR", str(runs_dir))

    response = client.post("/run-training", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["status"] == "completed"
    assert len(data["logs"]) == 1
    assert data["logs"][0]["loss"] >= 0
    assert data["trainingMetadata"]["datasetMode"] == "image_folder"
    assert data["trainingMetadata"]["sampleCount"] == 8
    assert data["trainingMetadata"]["splits"]["train"] == 4
    assert data["trainingMetadata"]["splits"]["val"] == 2
    assert data["evaluation"]["primarySplit"] == "validation"
    assert data["evaluation"]["sampleCount"] == 2
    assert len(data["evaluation"]["confusionMatrix"]) == 2
    assert data["trainingMetadata"]["numClasses"] == 2
    assert data["trainingMetadata"]["classNames"] == ["class_a", "class_b"]
    assert Path(data["trainingMetadata"]["weightsPath"]).exists()
