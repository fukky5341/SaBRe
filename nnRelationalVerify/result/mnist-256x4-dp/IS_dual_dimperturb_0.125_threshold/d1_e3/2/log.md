## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00035784


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0043085, -0.0042660, -0.0043085, -0.0042660, -0.0000242, 0.0000242)
1: (0.0033686, 0.0036042, 0.0033686, 0.0036042, -0.0001341, 0.0001341)
2: (0.0069139, 0.0074404, 0.0069139, 0.0074404, -0.0002996, 0.0002996)
3: (0.0041990, 0.0044208, 0.0041990, 0.0044208, -0.0001262, 0.0001262)
4: (1.0130407, 1.0139015, 1.0130407, 1.0139015, -0.0004898, 0.0004898)
5: (0.0047783, 0.0049458, 0.0047783, 0.0049458, -0.0000953, 0.0000953)
6: (-0.0121792, -0.0119612, -0.0121792, -0.0119612, -0.0001240, 0.0001240)
7: (-0.0103569, -0.0103291, -0.0103569, -0.0103291, -0.0000158, 0.0000158)
8: (-0.0025899, -0.0024393, -0.0025899, -0.0024393, -0.0000857, 0.0000857)
9: (-0.0059593, -0.0052055, -0.0059593, -0.0052055, -0.0004289, 0.0004289)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 1.21 = 2.60 seconds
status: Status.VERIFIED
relational distance
Output dim: 4, lower bound: -0.0002736, upper bound: 0.0002736
