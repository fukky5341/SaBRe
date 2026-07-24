## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000228159


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0002998, 0.0010674, -0.0002998, 0.0010674, -0.0013672, 0.0013672)
1: (-0.0032214, -0.0026700, -0.0032214, -0.0026700, -0.0005514, 0.0005514)
2: (0.0326852, 0.0335734, 0.0326852, 0.0335734, -0.0008882, 0.0008882)
3: (-0.0028954, -0.0018713, -0.0028954, -0.0018713, -0.0010241, 0.0010241)
4: (-0.0020915, -0.0013340, -0.0020915, -0.0013340, -0.0006531, 0.0006531)
5: (0.0122435, 0.0135173, 0.0122435, 0.0135173, -0.0012738, 0.0012738)
6: (-0.0033418, -0.0025041, -0.0033418, -0.0025041, -0.0006869, 0.0006869)
7: (0.9759021, 0.9764220, 0.9759021, 0.9764220, -0.0005199, 0.0005199)
8: (-0.0134976, -0.0099444, -0.0134976, -0.0099444, -0.0035532, 0.0035532)
9: (0.0017533, 0.0038100, 0.0017533, 0.0038100, -0.0020568, 0.0020568)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.22 + 1.18 = 2.40 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0002163, upper bound: 0.0002163
