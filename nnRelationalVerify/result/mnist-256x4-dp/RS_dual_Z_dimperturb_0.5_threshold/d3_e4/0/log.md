## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.044602215


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0121992, 0.0032295, -0.0121992, 0.0032295, -0.0154288, 0.0154288)
1: (-0.0083614, 0.0072660, -0.0083614, 0.0072660, -0.0156274, 0.0156274)
2: (0.9414485, 0.9809133, 0.9414485, 0.9809133, -0.0394648, 0.0394648)
3: (0.0071238, 0.0539079, 0.0071238, 0.0539079, -0.0467841, 0.0467841)
4: (-0.0243771, 0.0305014, -0.0243771, 0.0305014, -0.0548786, 0.0548786)
5: (0.0058533, 0.0327526, 0.0058533, 0.0327526, -0.0268993, 0.0268993)
6: (-0.0159816, 0.0166476, -0.0159816, 0.0166476, -0.0326293, 0.0326293)
7: (-0.0355408, 0.0004055, -0.0355408, 0.0004055, -0.0359463, 0.0359463)
8: (-0.0156895, 0.0283861, -0.0156895, 0.0283861, -0.0440756, 0.0440756)
9: (-0.0191583, 0.0140803, -0.0191583, 0.0140803, -0.0332386, 0.0332386)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.67 + 1.94 = 3.61 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0294885, upper bound: 0.0294885
