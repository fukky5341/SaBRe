## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0002144


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9914006, 0.9922203, 0.9914006, 0.9922203, -0.0004234, 0.0004234)
1: (-0.0034067, -0.0032024, -0.0034067, -0.0032024, -0.0001055, 0.0001055)
2: (0.0069173, 0.0079998, 0.0069173, 0.0079998, -0.0005591, 0.0005591)
3: (-0.0049143, -0.0044216, -0.0049143, -0.0044216, -0.0002545, 0.0002545)
4: (0.0018667, 0.0020762, 0.0018667, 0.0020762, -0.0001082, 0.0001082)
5: (0.0076596, 0.0090210, 0.0076596, 0.0090210, -0.0007032, 0.0007032)
6: (-0.0007488, -0.0004033, -0.0007488, -0.0004033, -0.0001785, 0.0001785)
7: (-0.0050750, -0.0041810, -0.0050750, -0.0041810, -0.0004618, 0.0004618)
8: (-0.0022330, -0.0017629, -0.0022330, -0.0017629, -0.0002429, 0.0002429)
9: (0.0001803, 0.0007255, 0.0001803, 0.0007255, -0.0002816, 0.0002816)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.37 + 1.22 = 2.59 seconds
status: Status.ADV_EXAMPLE
