from app.services.graph_ir import IRGraph, IRNode
from app.services.codegen import generate_model_code

nodes = [
    IRNode(node_id="Input_1", op="Input", params={"inputShape": [1, 28, 28]}, output_refs=["Conv2d_1"]),
    IRNode(node_id="Conv2d_1", op="Conv2d", params={"in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 0}, input_refs=["Input_1"], output_refs=["Output_1"]),
    IRNode(node_id="Output_1", op="Output", input_refs=["Conv2d_1"])
]

ir_graph = IRGraph(version="1.0.0", nodes=nodes, edges=[])
print(generate_model_code(ir_graph))
