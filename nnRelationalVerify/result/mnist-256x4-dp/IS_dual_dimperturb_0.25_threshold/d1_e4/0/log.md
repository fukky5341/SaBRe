## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 8.477e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0038393, -0.0023169, -0.0038393, -0.0023169, -0.0009134, 0.0009134)
1: (-0.0045972, -0.0043072, -0.0045972, -0.0043072, -0.0001599, 0.0001599)
2: (0.0096786, 0.0116191, 0.0096786, 0.0116191, -0.0011513, 0.0011513)
3: (1.0086075, 1.0090556, 1.0086075, 1.0090556, -0.0002942, 0.0002942)
4: (-0.0035076, -0.0032066, -0.0035076, -0.0032066, -0.0001762, 0.0001762)
5: (0.0010123, 0.0021769, 0.0010123, 0.0021769, -0.0006975, 0.0006975)
6: (-0.0025333, -0.0024793, -0.0025333, -0.0024793, -0.0000439, 0.0000439)
7: (-0.0097005, -0.0070777, -0.0097005, -0.0070777, -0.0016816, 0.0016816)
8: (-0.0054946, -0.0023418, -0.0054946, -0.0023418, -0.0018281, 0.0018281)
9: (-0.0030212, -0.0015210, -0.0030212, -0.0015210, -0.0008583, 0.0008583)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 1.31 = 2.70 seconds
status: Status.ADV_EXAMPLE
