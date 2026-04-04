from app.services.shape_infer import infer_flatten

res = infer_flatten([16, 28, 28], {"start_dim": 1, "end_dim": -1})
print("Result 1:", res.output_shape)

res = infer_flatten([16, 28, 28], {"start_dim": 0, "end_dim": 1})
print("Result 2:", res.output_shape)

res = infer_flatten([16, 28, 28], {"start_dim": -1, "end_dim": -1})
print("Result 3:", res.output_shape)

res = infer_flatten([16, 28], {"start_dim": 1, "end_dim": -1})
print("Result 4:", res.output_shape)
