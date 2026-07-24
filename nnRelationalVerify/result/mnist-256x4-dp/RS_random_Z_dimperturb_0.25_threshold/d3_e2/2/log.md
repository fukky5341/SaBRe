## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00017725


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041454, -0.0041304, -0.0041454, -0.0041304, -0.0000126, 0.0000126)
1: (-0.0079028, -0.0073407, -0.0079028, -0.0073407, -0.0004727, 0.0004727)
2: (0.9669797, 0.9676542, 0.9669797, 0.9676542, -0.0005672, 0.0005672)
3: (0.0027544, 0.0077292, 0.0027544, 0.0077292, -0.0041837, 0.0041837)
4: (-0.0012809, -0.0009025, -0.0012809, -0.0009025, -0.0003182, 0.0003182)
5: (0.0159758, 0.0163582, 0.0159758, 0.0163582, -0.0003216, 0.0003216)
6: (0.0039502, 0.0041362, 0.0039502, 0.0041362, -0.0001564, 0.0001564)
7: (-0.0097813, -0.0084921, -0.0097813, -0.0084921, -0.0010842, 0.0010842)
8: (0.0089691, 0.0099919, 0.0089691, 0.0099919, -0.0008602, 0.0008602)
9: (0.0138564, 0.0156961, 0.0138564, 0.0156961, -0.0015471, 0.0015471)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.52 + 1.25 = 2.77 seconds
status: Status.ADV_EXAMPLE
