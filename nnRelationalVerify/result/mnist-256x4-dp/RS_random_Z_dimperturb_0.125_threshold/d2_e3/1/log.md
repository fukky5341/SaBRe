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
Threshold: 9.720972e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0069907, 0.0071183, 0.0069907, 0.0071183, -0.0000660, 0.0000660)
1: (0.0016224, 0.0018696, 0.0016224, 0.0018696, -0.0001278, 0.0001278)
2: (0.0018483, 0.0038420, 0.0018483, 0.0038420, -0.0010307, 0.0010307)
3: (-0.0028557, -0.0026776, -0.0028557, -0.0026776, -0.0000921, 0.0000921)
4: (0.0074808, 0.0083447, 0.0074808, 0.0083447, -0.0004467, 0.0004467)
5: (-0.0017654, -0.0016365, -0.0017654, -0.0016365, -0.0000667, 0.0000667)
6: (0.9932545, 0.9934910, 0.9932545, 0.9934910, -0.0001223, 0.0001223)
7: (0.0001586, 0.0017225, 0.0001586, 0.0017225, -0.0008085, 0.0008085)
8: (0.0010380, 0.0015280, 0.0010380, 0.0015280, -0.0002533, 0.0002533)
9: (-0.0103788, -0.0094009, -0.0103788, -0.0094009, -0.0005056, 0.0005056)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.44 + 1.28 = 2.73 seconds
status: Status.VERIFIED
relational distance
Output dim: 6, lower bound: -0.0000845, upper bound: 0.0000845
