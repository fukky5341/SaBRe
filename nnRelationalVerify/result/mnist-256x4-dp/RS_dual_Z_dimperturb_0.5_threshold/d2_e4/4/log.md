## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0148662


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0012924, 0.0192246, 0.0012924, 0.0192246, -0.0179323, 0.0179323)
1: (-0.0051972, 0.0028626, -0.0051972, 0.0028626, -0.0080598, 0.0080598)
2: (0.0012261, 0.0136453, 0.0012261, 0.0136453, -0.0124192, 0.0124192)
3: (-0.0038829, 0.0055447, -0.0038829, 0.0055447, -0.0094276, 0.0094276)
4: (-0.0041578, -0.0003871, -0.0041578, -0.0003871, -0.0037706, 0.0037706)
5: (-0.0007937, 0.0073520, -0.0007937, 0.0073520, -0.0081457, 0.0081457)
6: (-0.0089086, 0.0068701, -0.0089086, 0.0068701, -0.0157787, 0.0157787)
7: (-0.0184032, 0.0036075, -0.0184032, 0.0036075, -0.0220107, 0.0220107)
8: (0.9793511, 0.9928450, 0.9793511, 0.9928450, -0.0134939, 0.0134939)
9: (-0.0084031, 0.0044923, -0.0084031, 0.0044923, -0.0128954, 0.0128954)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 2.49 = 3.92 seconds
status: Status.VERIFIED
relational distance
Output dim: 8, lower bound: -0.0089803, upper bound: 0.0089803
