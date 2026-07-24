## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0181036


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0043831, -0.0011446, -0.0043831, -0.0011446, -0.0032385, 0.0032385)
1: (-0.0028760, 0.0031920, -0.0028760, 0.0031920, -0.0059985, 0.0059985)
2: (0.0078349, 0.0213914, 0.0078349, 0.0213914, -0.0135565, 0.0135565)
3: (-0.0026529, 0.0040327, -0.0026529, 0.0040327, -0.0066856, 0.0066856)
4: (0.9886541, 1.0123956, 0.9886541, 1.0123956, -0.0237415, 0.0237415)
5: (-0.0040351, 0.0056843, -0.0040351, 0.0056843, -0.0097194, 0.0097194)
6: (-0.0117979, -0.0061870, -0.0117979, -0.0061870, -0.0056109, 0.0056109)
7: (-0.0103083, -0.0020216, -0.0103083, -0.0020216, -0.0082867, 0.0082867)
8: (-0.0065794, -0.0027027, -0.0065794, -0.0027027, -0.0038767, 0.0038767)
9: (-0.0046407, 0.0189527, -0.0046407, 0.0189527, -0.0197602, 0.0197602)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.74 + 2.93 = 4.67 seconds
status: Status.VERIFIED
relational distance
Output dim: 4, lower bound: -0.0149200, upper bound: 0.0149200
