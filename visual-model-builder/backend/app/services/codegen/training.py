"""Training script generation for Phase 2."""

from __future__ import annotations

from app.services.graph_ir import IRGraph
from app.services.training import get_input_shape, get_num_classes, resolve_training_components


def _indent_block(block: str, prefix: str = "    ") -> str:
    return "\n".join(f"{prefix}{line}" if line else "" for line in block.splitlines())


def _dataset_block(dataset_name: str, image_size: int, num_classes: int, input_shape: list[int], train_split: object) -> str:
    train_value = "True" if str(train_split).lower() not in {"false", "test"} else "False"
    if dataset_name == "MNIST":
        resize_line = f"transforms.Resize(({image_size}, {image_size})), " if image_size != 28 else ""
        return (
            "transform = transforms.Compose(["
            f"{resize_line}transforms.ToTensor()])\n"
            "dataset = datasets.MNIST(\n"
            "    root='./data',\n"
            f"    train={train_value},\n"
            "    download=True,\n"
            "    transform=transform,\n"
            ")\n"
            "dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), 256)))"
        )

    channels = input_shape[0] if input_shape else 1
    return (
        "dataset = datasets.FakeData(\n"
        "    size=256,\n"
        f"    image_size=({channels}, {image_size}, {image_size}),\n"
        f"    num_classes={num_classes},\n"
        "    transform=transforms.ToTensor(),\n"
        ")"
    )


def _generate_model_class(ir_graph: IRGraph) -> str:
    from app.services.codegen import generate_model_code

    return generate_model_code(ir_graph).strip()


def _classification_helpers_block() -> str:
    return """
def normalize_classification_outputs(outputs):
    while outputs.ndim > 2 and outputs.shape[-1] == 1:
        outputs = outputs.squeeze(-1)
    return outputs
""".strip()


def generate_training_code(ir_graph: IRGraph) -> str:
    """Generate a complete, readable training script."""

    components = resolve_training_components(ir_graph)
    input_shape = get_input_shape(ir_graph)
    num_classes = get_num_classes(ir_graph)

    dataset_params = components.dataset_node.params if components.dataset_node else {}
    dataloader_params = components.dataloader_node.params if components.dataloader_node else {}
    loss_params = components.loss_node.params if components.loss_node else {}
    optimizer_params = components.optimizer_node.params if components.optimizer_node else {}
    trainer_params = components.trainer_node.params if components.trainer_node else {}
    metric_params = components.metric_node.params if components.metric_node else {}

    dataset_name = str(dataset_params.get("datasetName", "FakeData"))
    image_size = int(dataset_params.get("imageSize", input_shape[-1]))
    batch_size = int(dataloader_params.get("batchSize", 32))
    shuffle = str(bool(dataloader_params.get("shuffle", True)))
    num_workers = int(dataloader_params.get("numWorkers", 0))
    loss_type = str(loss_params.get("lossType", "CrossEntropyLoss"))
    optimizer_type = str(optimizer_params.get("optimizerType", "Adam"))
    lr = float(optimizer_params.get("lr", 0.001))
    weight_decay = float(optimizer_params.get("weightDecay", 0.0))
    momentum = float(optimizer_params.get("momentum", 0.0))
    epochs = int(trainer_params.get("epochs", 1))
    device_setting = str(trainer_params.get("device", "cpu"))
    metric_enabled = metric_params.get("metricType") == "Accuracy"

    model_code = _generate_model_class(ir_graph)
    dataset_code = _dataset_block(
        dataset_name=dataset_name,
        image_size=image_size,
        num_classes=int(dataset_params.get("numClasses", num_classes)),
        input_shape=input_shape,
        train_split=dataset_params.get("trainSplit", True),
    )

    criterion_line = f"criterion = nn.{loss_type}()"
    if optimizer_type == "SGD":
        optimizer_line = (
            "optimizer = torch.optim.SGD("
            f"model.parameters(), lr={lr}, weight_decay={weight_decay}, momentum={momentum})"
        )
    else:
        optimizer_line = (
            "optimizer = torch.optim.Adam("
            f"model.parameters(), lr={lr}, weight_decay={weight_decay})"
        )

    metric_block = ""
    metric_log_line = "epoch_accuracy = None"
    if metric_enabled:
        metric_block = "running_correct = 0\nrunning_total = 0\n"
        metric_log_line = "epoch_accuracy = running_correct / running_total if running_total else 0.0"

    accuracy_update = ""
    if metric_enabled:
        accuracy_update = (
            "predictions = outputs.argmax(dim=1)\n"
            "running_correct += (predictions == labels).sum().item()\n"
            "running_total += labels.size(0)\n"
        )

    loss_target_block = "loss = criterion(outputs, labels)"
    if loss_type == "MSELoss":
        loss_target_block = (
            f"targets = torch.nn.functional.one_hot(labels, num_classes={num_classes}).float()\n"
            "loss = criterion(outputs, targets)"
        )
    else:
        loss_target_block = (
            "outputs = normalize_classification_outputs(outputs)\n"
            "loss = criterion(outputs, labels)"
        )

    training_function = f"""
def train():
    device = torch.device('cuda' if '{device_setting}' == 'auto' and torch.cuda.is_available() else 'cpu')
    model = Model().to(device)

{_indent_block(dataset_code)}
    dataloader = DataLoader(
        dataset,
        batch_size={batch_size},
        shuffle={shuffle},
        num_workers={num_workers},
    )

    {criterion_line}
    {optimizer_line}

    history = []

    for epoch in range(1, {epochs} + 1):
        model.train()
        running_loss = 0.0
{_indent_block(metric_block, "        ") if metric_block else ""}
        batch_count = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
{_indent_block(loss_target_block, "            ")}
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
{_indent_block(accuracy_update, "            ") if accuracy_update else ""}
            batch_count += 1

        epoch_loss = running_loss / max(batch_count, 1)
        {metric_log_line}
        history.append({{"epoch": epoch, "loss": epoch_loss, "accuracy": epoch_accuracy}})
        print(f"Epoch {{epoch}}: loss={{epoch_loss:.4f}}" + (f", accuracy={{epoch_accuracy:.4f}}" if epoch_accuracy is not None else ""))

    return history


if __name__ == "__main__":
    train()
""".strip()

    return (
        "import torch\n"
        "import torch.nn as nn\n"
        "from torch.utils.data import DataLoader\n"
        "from torchvision import datasets, transforms\n\n\n"
        f"{model_code}\n\n\n"
        f"{_classification_helpers_block()}\n\n\n"
        f"{training_function}\n"
    )
