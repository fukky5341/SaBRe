## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00027335


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0065773, 0.0072417, 0.0065773, 0.0072417, -0.0003958, 0.0003958)
1: (0.0008218, 0.0021085, 0.0008218, 0.0021085, -0.0007666, 0.0007666)
2: (-0.0000793, 0.0102997, -0.0000793, 0.0102997, -0.0061837, 0.0061837)
3: (-0.0034325, -0.0025055, -0.0034325, -0.0025055, -0.0005523, 0.0005523)
4: (0.0046824, 0.0091800, 0.0046824, 0.0091800, -0.0026796, 0.0026796)
5: (-0.0018901, -0.0012187, -0.0018901, -0.0012187, -0.0004000, 0.0004000)
6: (0.9924883, 0.9937197, 0.9924883, 0.9937197, -0.0007337, 0.0007337)
7: (-0.0049069, 0.0032346, -0.0049069, 0.0032346, -0.0048506, 0.0048506)
8: (-0.0005490, 0.0020017, -0.0005490, 0.0020017, -0.0015197, 0.0015197)
9: (-0.0113243, -0.0062335, -0.0113243, -0.0062335, -0.0030330, 0.0030330)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.38 + 1.55 = 2.94 seconds
status: Status.ADV_EXAMPLE
