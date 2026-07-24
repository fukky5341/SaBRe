## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00226296


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0022649, -0.0007072, -0.0022649, -0.0007072, -0.0015577, 0.0015577)
1: (0.9982450, 1.0015125, 0.9982450, 1.0015125, -0.0024286, 0.0024286)
2: (-0.0014067, 0.0012077, -0.0014067, 0.0012077, -0.0026144, 0.0026144)
3: (0.0010803, 0.0020717, 0.0010803, 0.0020717, -0.0006960, 0.0006960)
4: (-0.0011270, 0.0008067, -0.0011270, 0.0008067, -0.0017568, 0.0017568)
5: (-0.0002518, 0.0019579, -0.0002518, 0.0019579, -0.0022097, 0.0022097)
6: (-0.0001066, 0.0012507, -0.0001066, 0.0012507, -0.0013574, 0.0013574)
7: (-0.0050219, -0.0031106, -0.0050219, -0.0031106, -0.0015538, 0.0015538)
8: (-0.0090417, -0.0041230, -0.0090417, -0.0041230, -0.0041446, 0.0041446)
9: (0.0020818, 0.0049673, 0.0020818, 0.0049673, -0.0028855, 0.0028855)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.72 + 1.57 = 3.29 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.0019659, upper bound: 0.0019659
