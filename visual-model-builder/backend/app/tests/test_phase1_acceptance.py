from app.schemas.graph import GraphEdge, GraphNode, GraphNodeData, Position, ProjectGraph, ProjectMetadata
from app.services.codegen import generate_model_code
from app.services.graph_ir import project_to_ir
from app.services.shape_infer import infer_graph_shapes


def build_phase1_graph() -> ProjectGraph:
    return ProjectGraph(
        version="1.0.0",
        metadata=ProjectMetadata(
            name="Phase 1 Example",
            createdAt="2026-04-03T00:00:00Z",
            updatedAt="2026-04-03T00:00:00Z",
        ),
        nodes=[
            GraphNode(
                id="Input_1",
                type="Input",
                position=Position(x=0, y=0),
                data=GraphNodeData(label="Input", params={"inputShape": [1, 28, 28]}),
            ),
            GraphNode(
                id="Conv2d_1",
                type="Conv2d",
                position=Position(x=100, y=0),
                data=GraphNodeData(
                    label="Conv2d",
                    params={
                        "in_channels": 1,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    },
                ),
            ),
            GraphNode(
                id="ReLU_1",
                type="ReLU",
                position=Position(x=200, y=0),
                data=GraphNodeData(label="ReLU", params={"inplace": True}),
            ),
            GraphNode(
                id="MaxPool2d_1",
                type="MaxPool2d",
                position=Position(x=300, y=0),
                data=GraphNodeData(
                    label="MaxPool2d",
                    params={"kernel_size": 2, "stride": 2, "padding": 0},
                ),
            ),
            GraphNode(
                id="Flatten_1",
                type="Flatten",
                position=Position(x=400, y=0),
                data=GraphNodeData(label="Flatten", params={"start_dim": 0, "end_dim": -1}),
            ),
            GraphNode(
                id="Linear_1",
                type="Linear",
                position=Position(x=500, y=0),
                data=GraphNodeData(
                    label="Linear",
                    params={"in_features": 3136, "out_features": 10, "bias": False},
                ),
            ),
            GraphNode(
                id="Output_1",
                type="Output",
                position=Position(x=600, y=0),
                data=GraphNodeData(label="Output", params={}),
            ),
        ],
        edges=[
            GraphEdge(id="e1", source="Input_1", target="Conv2d_1"),
            GraphEdge(id="e2", source="Conv2d_1", target="ReLU_1"),
            GraphEdge(id="e3", source="ReLU_1", target="MaxPool2d_1"),
            GraphEdge(id="e4", source="MaxPool2d_1", target="Flatten_1"),
            GraphEdge(id="e5", source="Flatten_1", target="Linear_1"),
            GraphEdge(id="e6", source="Linear_1", target="Output_1"),
        ],
    )


def test_phase1_shape_inference_matches_reference_cnn():
    ir_graph = project_to_ir(build_phase1_graph())

    results = infer_graph_shapes(ir_graph)

    assert results["Conv2d_1"].output_shape == [16, 28, 28]
    assert results["MaxPool2d_1"].output_shape == [16, 14, 14]
    assert results["Flatten_1"].output_shape == [3136]
    assert results["Linear_1"].output_shape == [10]


def test_codegen_includes_editable_relu_and_linear_params():
    ir_graph = project_to_ir(build_phase1_graph())

    code = generate_model_code(ir_graph)

    assert "nn.ReLU(inplace=True)" in code
    assert "nn.Linear(3136, 10, bias=False)" in code
