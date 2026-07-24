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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0069676, 0.0071111, 0.0069676, 0.0071111, -0.0000723, 0.0000723)
1: (0.0015778, 0.0018557, 0.0015778, 0.0018557, -0.0001399, 0.0001399)
2: (0.0019597, 0.0042016, 0.0019597, 0.0042016, -0.0011288, 0.0011288)
3: (-0.0028878, -0.0026876, -0.0028878, -0.0026876, -0.0001008, 0.0001008)
4: (0.0073249, 0.0082964, 0.0073249, 0.0082964, -0.0004892, 0.0004892)
5: (-0.0017582, -0.0016132, -0.0017582, -0.0016132, -0.0000730, 0.0000730)
6: (0.9932118, 0.9934778, 0.9932118, 0.9934778, -0.0001339, 0.0001339)
7: (-0.0001235, 0.0016351, -0.0001235, 0.0016351, -0.0008855, 0.0008855)
8: (0.0009497, 0.0015006, 0.0009497, 0.0015006, -0.0002774, 0.0002774)
9: (-0.0103241, -0.0092245, -0.0103241, -0.0092245, -0.0005537, 0.0005537)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 1.28 = 2.68 seconds
status: Status.VERIFIED
relational distance
Output dim: 6, lower bound: -0.0000967, upper bound: 0.0000967
