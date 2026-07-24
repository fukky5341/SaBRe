## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00104286


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041982, -0.0041769, -0.0041982, -0.0041769, -0.0000213, 0.0000213)
1: (-0.0098795, -0.0095077, -0.0098795, -0.0095077, -0.0003718, 0.0003718)
2: (0.9646077, 0.9650539, 0.9646077, 0.9650539, -0.0004462, 0.0004462)
3: (-0.0147420, -0.0114508, -0.0147420, -0.0114508, -0.0024787, 0.0024787)
4: (0.0001779, 0.0004282, 0.0001779, 0.0004282, -0.0002503, 0.0002503)
5: (0.0174501, 0.0177790, 0.0174501, 0.0177790, -0.0003289, 0.0003289)
6: (0.0030937, 0.0034191, 0.0030937, 0.0034191, -0.0003253, 0.0003253)
7: (-0.0048107, -0.0038376, -0.0048107, -0.0038376, -0.0009731, 0.0009731)
8: (0.0129126, 0.0135893, 0.0129126, 0.0135893, -0.0006767, 0.0006767)
9: (0.0209491, 0.0221662, 0.0209491, 0.0221662, -0.0011273, 0.0011273)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 1.40 = 2.58 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0002646, upper bound: 0.0002646
